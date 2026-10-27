"""Shared test helpers."""

from __future__ import annotations


def register(client, email: str = "coach@athletiq.test", password: str = "supersecret1", name: str = "Coach") -> str:
    res = client.post("/api/auth/register", json={"name": name, "email": email, "password": password})
    if res.status_code == 409:
        res = client.post("/api/auth/login", json={"email": email, "password": password})
    assert res.status_code in (200, 201), res.text
    return res.json()["access_token"]
