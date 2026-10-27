"""Sign up, sign in, sign out and profile routes."""

from __future__ import annotations
import re
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pymongo.errors import DuplicateKeyError

from backend.app import db
from backend.app.schemas.models import AuthResponse, LoginRequest, RegisterRequest, UserOut
from backend.app.security import create_session, get_current_user, hash_password, public_user, revoke_session, verify_password

router = APIRouter(prefix="/api/auth", tags=["Authentication"])

_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


@router.post("/register", response_model=AuthResponse, status_code=201, summary="Create an account and sign in")
def register(payload: RegisterRequest, request: Request):
    email = payload.email.strip().lower()
    if not _EMAIL.match(email):
        raise HTTPException(status_code=422, detail="Enter a valid email address")
    user = {
        "name": payload.name.strip(),
        "email": email,
        "password_hash": hash_password(payload.password),
        "created_at": datetime.now(timezone.utc),
    }
    try:
        user["_id"] = db.users().insert_one(user).inserted_id
    except DuplicateKeyError:
        raise HTTPException(status_code=409, detail="An account with this email already exists")
    token = create_session(user["_id"], request.headers.get("user-agent", ""))
    return {"access_token": token, "user": public_user(user)}


@router.post("/login", response_model=AuthResponse, summary="Sign in with email and password")
def login(payload: LoginRequest, request: Request):
    user = db.users().find_one({"email": payload.email.strip().lower()})
    if not user or not verify_password(payload.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Incorrect email or password")
    token = create_session(user["_id"], request.headers.get("user-agent", ""))
    return {"access_token": token, "user": public_user(user)}


@router.post("/logout", status_code=204, summary="Sign out and revoke the current session")
def logout(request: Request, user: dict[str, Any] = Depends(get_current_user)):
    revoke_session(request.state.session_id)


@router.get("/me", response_model=UserOut, summary="Get the signed-in user")
def me(user: dict[str, Any] = Depends(get_current_user)):
    return public_user(user)
