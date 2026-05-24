"""Admin-only user management endpoints."""

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.auth import require_roles
from app.core.security import hash_password
from app.schemas.auth import UserRead
from app.schemas.users import UserCreate, UserUpdate
from database.models import User, UserRole
from database.session import get_db

router = APIRouter(prefix="/users", tags=["users"])


def _normalize(value: str) -> str:
    return value.strip().lower()


def _role_from_string(value: str) -> UserRole:
    try:
        return UserRole(value)
    except ValueError as exc:
        allowed = ", ".join(role.value for role in UserRole)
        raise HTTPException(status_code=400, detail=f"Invalid role. Allowed roles: {allowed}") from exc


def _ensure_unique_login(db: Session, username: str, email: str, exclude_user_id: int | None = None) -> None:
    stmt = select(User).where(or_(User.username == username, User.email == email))
    if exclude_user_id is not None:
        stmt = stmt.where(User.id != exclude_user_id)

    existing = db.scalar(stmt)
    if existing is not None:
        raise HTTPException(status_code=409, detail="Username or email already exists")


@router.get("", response_model=list[UserRead])
def list_users(
    active_only: bool = False,
    limit: int = Query(default=100, ge=1, le=500),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(UserRole.ADMIN)),
) -> list[User]:
    """List users for admin management."""

    stmt = select(User).order_by(User.id).limit(limit)
    if active_only:
        stmt = stmt.where(User.is_active.is_(True))
    return list(db.scalars(stmt).all())


@router.post("", response_model=UserRead, status_code=status.HTTP_201_CREATED)
def create_user(
    payload: UserCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(UserRole.ADMIN)),
) -> User:
    """Create a manually managed user."""

    username = _normalize(payload.username)
    email = _normalize(payload.email)
    _ensure_unique_login(db, username, email)

    user = User(
        username=username,
        email=email,
        password_hash=hash_password(payload.password),
        full_name=payload.full_name,
        role=_role_from_string(payload.role),
        is_active=payload.is_active,
    )
    db.add(user)
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(status_code=409, detail="Username or email already exists") from exc

    db.refresh(user)
    return user


@router.patch("/{user_id}", response_model=UserRead)
def update_user(
    user_id: int,
    payload: UserUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(UserRole.ADMIN)),
) -> User:
    """Update a user, including password, role, and active state."""

    user = db.get(User, user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")

    username = _normalize(payload.username) if payload.username is not None else user.username
    email = _normalize(payload.email) if payload.email is not None else user.email
    _ensure_unique_login(db, username, email, exclude_user_id=user.id)

    user.username = username
    user.email = email
    if payload.password is not None:
        user.password_hash = hash_password(payload.password)
    if payload.full_name is not None:
        user.full_name = payload.full_name
    if payload.role is not None:
        user.role = _role_from_string(payload.role)
    if payload.is_active is not None:
        user.is_active = payload.is_active

    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(status_code=409, detail="Username or email already exists") from exc

    db.refresh(user)
    return user

