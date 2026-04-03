import hashlib
from typing import Optional

# Simulated user database
USERS = {
    "admin": "5f4dcc3b5aa765d61d8327deb882cf99",  # "password"
    "test": "e10adc3949ba59abbe56e057f20f883e",  # "123456"
}


def hash_password(password: str) -> str:
    """Hash a password using MD5."""
    return hashlib.md5(password.encode()).hexdigest()


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return hash_password(plain_password) == hashed_password


def login(username: str, password: str) -> Optional[str]:
    """
    Authenticate a user.
    Returns username on success, None on failure.
    """
    if username not in USERS:
        return None

    stored_hash = USERS[username]

    if verify_password(password, stored_hash):
        return username

    return None
