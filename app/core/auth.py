"""FastAPI authentication dependencies."""

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from app.core.security import decode_access_token, verify_password
from database.models import User, UserRole
from database.session import get_db

bearer_scheme = HTTPBearer(auto_error=False)


def get_user_by_login(db: Session, username_or_email: str) -> User | None:
    """Find a user by username or email."""

    value = username_or_email.strip().lower()
    return db.scalar(
        select(User).where(
            or_(
                User.username == value,
                User.email == value,
            )
        )
    )


def authenticate_user(db: Session, username_or_email: str, password: str) -> User | None:
    """Validate login credentials."""

    user = get_user_by_login(db, username_or_email)
    if user is None or not user.is_active:
        return None
    if not verify_password(password, user.password_hash):
        return None
    return user


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    db: Session = Depends(get_db),
) -> User:
    """Return the currently authenticated user."""

    if credentials is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")

    subject = decode_access_token(credentials.credentials)
    if subject is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

    user = db.get(User, int(subject))
    if user is None or not user.is_active:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found or inactive")

    return user


def require_roles(*roles: UserRole):
    """Create a dependency that allows only selected roles."""

    def dependency(current_user: User = Depends(get_current_user)) -> User:
        if current_user.role not in roles:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient permissions")
        return current_user

    return dependency
