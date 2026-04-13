# backend/app/services/auth_service.py
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from fastapi import HTTPException, status
from app.models.user import User, UserRole
from app.models.tenant import Tenant
from app.core.security import hash_password, verify_password, create_access_token, create_refresh_token
from app.schemas.auth import TenantRegisterRequest, LoginRequest, TokenResponse
import re


def slugify(text: str) -> str:
    """Convert 'Acme Corp' → 'acme-corp' for URL-friendly tenant slug."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_-]+", "-", text)
    return text


async def register_tenant(data: TenantRegisterRequest, db: AsyncSession) -> TokenResponse:
    """
    Creates a new tenant (company) + owner user in one transaction.
    This is what happens when someone signs up on your SaaS.
    """
    # Check email not already used
    existing = await db.execute(select(User).where(User.email == data.email))
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="Email already registered")

    # Create unique slug — if "acme-corp" exists, try "acme-corp-2" etc.
    base_slug = slugify(data.company_name)
    slug = base_slug
    counter = 1
    while True:
        existing_tenant = await db.execute(select(Tenant).where(Tenant.slug == slug))
        if not existing_tenant.scalar_one_or_none():
            break
        slug = f"{base_slug}-{counter}"
        counter += 1

    # Create tenant
    tenant = Tenant(name=data.company_name, slug=slug)
    db.add(tenant)
    await db.flush()  # flush to get tenant.id without committing yet

    # Create owner user
    user = User(
        email=data.email,
        full_name=data.full_name,
        hashed_password=hash_password(data.password),
        role=UserRole.OWNER,
        tenant_id=tenant.id,
        is_verified=True,  # skip email verification for now
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)

    # Return tokens immediately — user is logged in after signup
    return TokenResponse(
        access_token=create_access_token(str(user.id)),
        refresh_token=create_refresh_token(str(user.id)),
    )


async def login_user(data: LoginRequest, db: AsyncSession) -> TokenResponse:
    """Verify credentials and return JWT tokens."""
    result = await db.execute(select(User).where(User.email == data.email))
    user = result.scalar_one_or_none()

    if not user or not verify_password(data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password"
        )
    if not user.is_active:
        raise HTTPException(status_code=400, detail="Account is deactivated")

    return TokenResponse(
        access_token=create_access_token(str(user.id)),
        refresh_token=create_refresh_token(str(user.id)),
    )