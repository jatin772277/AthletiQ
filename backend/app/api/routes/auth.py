"""Sign up, sign in, sign out and profile routes."""

from __future__ import annotations
import re
import hashlib
import secrets
import smtplib
from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pymongo.errors import DuplicateKeyError

from backend.app import db
from backend.app.config import settings
from backend.app.schemas.models import (
    AuthResponse, LoginRequest, PasswordResetConfirm, PasswordResetRequest,
    PasswordResetRequestResponse, RegisterRequest, UserOut,
)
from backend.app.security import create_session, get_current_user, hash_password, public_user, revoke_session, verify_password
from backend.app.services.email_service import gmail_is_configured, send_password_reset_code

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


@router.post("/password-reset/request", response_model=PasswordResetRequestResponse, summary="Request a password reset")
def request_password_reset(payload: PasswordResetRequest):
    """Create a one-time reset token without revealing whether an account exists."""
    email = payload.email.strip().lower()
    message = "If an account exists for this email, a password reset code has been created."
    user = db.users().find_one({"email": email}, {"_id": 1})
    if not user:
        return {"message": message}

    db.password_reset_tokens().delete_many({"user_id": user["_id"]})
    token = secrets.token_urlsafe(32)
    now = datetime.now(timezone.utc)
    reset = db.password_reset_tokens().insert_one({
        "user_id": user["_id"],
        "token_hash": hashlib.sha256(token.encode()).hexdigest(),
        "created_at": now,
        "expires_at": now + timedelta(hours=1),
    })
    if gmail_is_configured():
        try:
            send_password_reset_code(email, token)
        except (OSError, RuntimeError, smtplib.SMTPException):
            db.password_reset_tokens().delete_one({"_id": reset.inserted_id})
            raise HTTPException(status_code=503, detail="Unable to send the password reset email. Please try again later.")
        return {"message": "Check your email for a password reset code."}

    # Local development keeps a usable no-email fallback. Production configuration
    # is rejected at startup when Gmail credentials have not been supplied.
    return {"message": message, "reset_token": token if settings.app_environment != "production" else None}


@router.post("/password-reset/confirm", status_code=204, summary="Reset a password with a one-time code")
def confirm_password_reset(payload: PasswordResetConfirm):
    token_hash = hashlib.sha256(payload.token.encode()).hexdigest()
    reset = db.password_reset_tokens().find_one_and_delete({
        "token_hash": token_hash,
        "expires_at": {"$gt": datetime.now(timezone.utc)},
    })
    if not reset:
        raise HTTPException(status_code=400, detail="This password reset code is invalid or has expired")
    db.users().update_one({"_id": reset["user_id"]}, {"$set": {"password_hash": hash_password(payload.password)}})
    db.sessions().delete_many({"user_id": reset["user_id"]})


@router.post("/logout", status_code=204, summary="Sign out and revoke the current session")
def logout(request: Request, user: dict[str, Any] = Depends(get_current_user)):
    revoke_session(request.state.session_id)


@router.get("/me", response_model=UserOut, summary="Get the signed-in user")
def me(user: dict[str, Any] = Depends(get_current_user)):
    return public_user(user)
