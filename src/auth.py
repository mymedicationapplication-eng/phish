from __future__ import annotations

import re
from typing import Dict, Optional, Tuple

from .database import Database
from .text_utils import is_valid_email


SPECIAL_CHAR_PATTERN = re.compile(r'[^A-Za-z0-9]')


def assess_password_strength(password: str) -> tuple[bool, str]:
    if len(password) < 8:
        return False, 'Password must be at least 8 characters long.'
    if password.lower() in {'password', '12345678', 'qwertyui', 'admin1234'}:
        return False, 'Please choose a stronger password.'
    if not re.search(r'[A-Z]', password):
        return False, 'Password must contain at least one uppercase letter.'
    if not re.search(r'[a-z]', password):
        return False, 'Password must contain at least one lowercase letter.'
    if not re.search(r'\d', password):
        return False, 'Password must contain at least one number.'
    if not SPECIAL_CHAR_PATTERN.search(password):
        return False, 'Password must contain at least one special character.'
    return True, 'Strong password.'


def validate_registration(full_name: str, email: str, password: str, confirm_password: str) -> Tuple[bool, str]:
    if not full_name.strip():
        return False, 'Full name is required.'
    if len(full_name.strip()) < 3:
        return False, 'Full name must be at least 3 characters long.'
    if not is_valid_email(email):
        return False, 'Please enter a valid email address.'
    password_ok, password_message = assess_password_strength(password)
    if not password_ok:
        return False, password_message
    if password != confirm_password:
        return False, 'Passwords do not match.'
    return True, 'Valid'


def register_user(
    db: Database,
    full_name: str,
    email: str,
    password: str,
    confirm_password: str,
    institution: str = '',
    bio: str = '',
    role: str = 'user',
    actor_user_id: int | None = None,
) -> Tuple[bool, str, Optional[Dict]]:
    is_valid, message = validate_registration(full_name, email, password, confirm_password)
    if not is_valid:
        return False, message, None

    existing = db.get_user_by_email(email)
    if existing:
        return False, 'An account with this email already exists.', None

    user_id = db.create_user(
        full_name=full_name,
        email=email,
        password=password,
        institution=institution,
        bio=bio,
        role=role,
    )
    user = db.get_user_by_id(user_id)
    db.log_event(
        actor_user_id=actor_user_id or user_id,
        action='register_user',
        target_type='user',
        target_id=user_id,
        description=f'User account created for {email.strip().lower()} with role {role}.',
        severity='info',
    )
    return True, 'Account created successfully.', user


def login_user(db: Database, email: str, password: str) -> Tuple[bool, str, Optional[Dict]]:
    if not email.strip() or not password:
        return False, 'Email and password are required.', None
    user, message = db.verify_user(email, password)
    if not user:
        return False, message, None
    db.log_event(
        actor_user_id=user['id'],
        action='login',
        target_type='user',
        target_id=user['id'],
        description=f"Successful login for {user['email']}.",
        severity='info',
    )
    return True, message, user
