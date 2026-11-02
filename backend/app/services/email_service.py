"""Transactional email delivery for account-security messages."""

from __future__ import annotations

import smtplib
from email.message import EmailMessage

from backend.app.config import settings


def gmail_is_configured() -> bool:
    return bool(settings.gmail_smtp_email and settings.gmail_smtp_app_password)


def send_password_reset_code(recipient: str, code: str) -> None:
    """Send a one-time reset code through Gmail's SSL SMTP endpoint."""
    if not gmail_is_configured():
        raise RuntimeError("Gmail SMTP is not configured")

    message = EmailMessage()
    message["Subject"] = "Your AthletiQ password reset code"
    message["From"] = f"{settings.gmail_from_name} <{settings.gmail_smtp_email}>"
    message["To"] = recipient
    message.set_content(
        f"Use this one-time code to reset your AthletiQ password:\n\n{code}\n\n"
        "It expires in 1 hour. If you did not request this, you can ignore this email."
    )
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=15) as smtp:
        smtp.login(settings.gmail_smtp_email, settings.gmail_smtp_app_password)
        smtp.send_message(message)
