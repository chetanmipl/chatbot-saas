# backend/app/schemas/auth.py
from uuid import UUID
from pydantic import BaseModel, EmailStr, field_validator
import re


class TenantRegisterRequest(BaseModel):
    """What the user sends when signing up."""
    company_name: str
    email:        EmailStr
    password:     str
    full_name:    str

    @field_validator("password")
    @classmethod
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters")
        if not re.search(r"[A-Z]", v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not re.search(r"\d", v):
            raise ValueError("Password must contain at least one number")
        return v

    @field_validator("company_name")
    @classmethod
    def validate_company_name(cls, v):
        if len(v.strip()) < 2:
            raise ValueError("Company name too short")
        return v.strip()


class LoginRequest(BaseModel):
    email:    EmailStr
    password: str


class TokenResponse(BaseModel):
    """What we send back after successful login."""
    access_token:  str
    refresh_token: str
    token_type:    str = "bearer"


class RefreshRequest(BaseModel):
    refresh_token: str


class UserResponse(BaseModel):
    """Safe user data to return in API responses — never include password."""
    id:        UUID
    email:     str
    full_name: str
    role:      str
    is_active: bool
    tenant_id: UUID

    model_config = {"from_attributes": True}