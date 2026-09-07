"""
Create or reset the admin user.

Usage (run from the `backend` folder):

    python create_admin.py                      # uses ADMIN_EMAIL/ADMIN_PASSWORD from .env
    python create_admin.py you@example.com S3cret!Pass

Creates the admin if missing, or resets the password if the account already
exists. The new account can log in immediately -- no forced password change.
"""
import asyncio
import sys

from app.config import settings
from app.core.security import validate_password
from app.services.auth import get_password_hash
from app.database import get_database


async def main() -> int:
    email = sys.argv[1] if len(sys.argv) > 1 else settings.ADMIN_EMAIL
    password = sys.argv[2] if len(sys.argv) > 2 else settings.ADMIN_PASSWORD

    if not email or not password:
        print("ERROR: no email/password given and none set in .env")
        print("Usage: python create_admin.py <email> <password>")
        return 1

    ok, message = validate_password(password)
    if not ok:
        print(f"ERROR: password rejected: {message}")
        return 1

    db = await get_database()

    existing = await db.get_user_by_email(email)
    if existing:
        await db.update_user(str(existing["user_id"]), {
            "password_hash": get_password_hash(password),
            "role": "admin",
            "is_active": True,
            "must_change_password": False,
        })
        print(f"Password reset for existing user: {email}")
    else:
        await db.create_user({
            "email": email,
            "name": "Admin",
            "password_hash": get_password_hash(password),
            "role": "admin",
            "is_active": True,
            "must_change_password": False,
        })
        print(f"Admin created: {email}")

    print(f"\n  Email:    {email}")
    print(f"  Password: {password}")
    print("\nYou can log in at http://localhost:8000/login")

    await db.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
