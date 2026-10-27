"""Password hashing, session tokens and the current-user dependency."""

from __future__ import annotations
import base64
import hashlib
import hmac
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any

import jwt
from bson import ObjectId
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from backend.app import db
from backend.app.config import settings

_SCRYPT_N, _SCRYPT_R, _SCRYPT_P = 2**14, 8, 1
_bearer = HTTPBearer(auto_error=False)


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, n=_SCRYPT_N, r=_SCRYPT_R, p=_SCRYPT_P)
    b64 = lambda b: base64.b64encode(b).decode()
    return f"scrypt${_SCRYPT_N}${_SCRYPT_R}${_SCRYPT_P}${b64(salt)}${b64(digest)}"


def verify_password(password: str, stored: str) -> bool:
    try:
        _, n, r, p, salt, digest = stored.split("$")
        candidate = hashlib.scrypt(password.encode(), salt=base64.b64decode(salt), n=int(n), r=int(r), p=int(p))
        return hmac.compare_digest(candidate, base64.b64decode(digest))
    except (ValueError, TypeError):
        return False


def create_session(user_id: ObjectId, user_agent: str = "") -> str:
    """Store a session in MongoDB and return a JWT that references it, so sign-out can revoke it."""
    now = datetime.now(timezone.utc)
    jti = secrets.token_urlsafe(24)
    expires = now + timedelta(hours=settings.session_ttl_hours)
    db.sessions().insert_one({"_id": jti, "user_id": user_id, "created_at": now, "expires_at": expires, "user_agent": user_agent[:200]})
    payload = {"sub": str(user_id), "jti": jti, "iat": now, "exp": expires}
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def revoke_session(jti: str) -> None:
    db.sessions().delete_one({"_id": jti})


def public_user(user: dict[str, Any]) -> dict[str, Any]:
    return {"id": str(user["_id"]), "name": user["name"], "email": user["email"], "created_at": user["created_at"]}


_unauthorized = HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not signed in", headers={"WWW-Authenticate": "Bearer"})


def get_current_user(request: Request, creds: HTTPAuthorizationCredentials | None = Depends(_bearer)) -> dict[str, Any]:
    if creds is None:
        raise _unauthorized
    try:
        payload = jwt.decode(creds.credentials, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
    except jwt.PyJWTError:
        raise _unauthorized
    session = db.sessions().find_one({"_id": payload.get("jti")})
    if not session:
        raise _unauthorized
    expires = session["expires_at"]
    if expires.tzinfo is None:
        expires = expires.replace(tzinfo=timezone.utc)
    if expires < datetime.now(timezone.utc):
        raise _unauthorized
    user = db.users().find_one({"_id": session["user_id"]})
    if not user:
        raise _unauthorized
    request.state.session_id = session["_id"]
    return user


def authorize_match(request: Request, user: dict[str, Any] = Depends(get_current_user)) -> None:
    """Allow a match route only for demo matches or completed matches the user owns."""
    match_id = request.path_params.get("match_id")
    if match_id is None or match_id in settings.demo_match_ids:
        return
    if not db.jobs().find_one({"_id": match_id, "owner_id": user["_id"], "status": "completed"}, {"_id": 1}):
        raise HTTPException(status_code=404, detail=f"Match '{match_id}' not found")
