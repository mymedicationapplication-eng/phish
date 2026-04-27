from pathlib import Path

from src.auth import login_user, register_user
from src.database import Database


def test_default_admin_is_created(tmp_path: Path):
    db = Database(tmp_path / 'test.db')
    admin = db.get_user_by_email('admin@phishguard.local')
    assert admin is not None
    assert admin['role'] == 'admin'


def test_registration_and_login_flow(tmp_path: Path):
    db = Database(tmp_path / 'test.db')
    ok, message, user = register_user(
        db,
        full_name='Test Student',
        email='student@example.com',
        password='StrongPass@1',
        confirm_password='StrongPass@1',
        institution='Demo University',
        bio='Testing registration flow',
    )
    assert ok is True
    assert user is not None
    ok, message, user = login_user(db, 'student@example.com', 'StrongPass@1')
    assert ok is True
    assert user is not None
    assert user['email'] == 'student@example.com'
