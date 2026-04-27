from __future__ import annotations

import hashlib
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import (
    DB_PATH,
    DEFAULT_ADMIN_EMAIL,
    DEFAULT_ADMIN_NAME,
    DEFAULT_ADMIN_PASSWORD,
    STORAGE_DIR,
)


UTC_FORMAT = '%Y-%m-%dT%H:%M:%S%z'


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class Database:
    def __init__(self, db_path: Path = DB_PATH):
        STORAGE_DIR.mkdir(parents=True, exist_ok=True)
        self.db_path = Path(db_path)
        self.initialize()

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute('PRAGMA foreign_keys = ON')
        return conn

    def initialize(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    full_name TEXT NOT NULL,
                    email TEXT NOT NULL UNIQUE,
                    password_hash TEXT NOT NULL,
                    salt TEXT NOT NULL,
                    role TEXT NOT NULL DEFAULT 'user',
                    bio TEXT NOT NULL DEFAULT '',
                    institution TEXT NOT NULL DEFAULT '',
                    is_active INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    last_login_at TEXT
                );

                CREATE TABLE IF NOT EXISTS scan_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    message_text TEXT NOT NULL,
                    normalized_text TEXT NOT NULL,
                    predicted_label INTEGER NOT NULL,
                    predicted_name TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    phishing_probability REAL NOT NULL,
                    legitimate_probability REAL NOT NULL,
                    heuristic_risk_score INTEGER NOT NULL,
                    risk_level TEXT NOT NULL,
                    suspicious_keywords TEXT NOT NULL,
                    detected_urls TEXT NOT NULL,
                    signal_count INTEGER NOT NULL DEFAULT 0,
                    signal_details TEXT NOT NULL DEFAULT '',
                    ml_top_terms TEXT NOT NULL DEFAULT '',
                    recommendation TEXT NOT NULL DEFAULT '',
                    source_type TEXT NOT NULL DEFAULT 'manual',
                    source_name TEXT NOT NULL DEFAULT '',
                    user_feedback TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    actor_user_id INTEGER,
                    action TEXT NOT NULL,
                    target_type TEXT NOT NULL DEFAULT '',
                    target_id INTEGER,
                    severity TEXT NOT NULL DEFAULT 'info',
                    description TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(actor_user_id) REFERENCES users(id) ON DELETE SET NULL
                );

                CREATE TABLE IF NOT EXISTS training_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    actor_user_id INTEGER,
                    accuracy REAL NOT NULL,
                    precision_score REAL NOT NULL,
                    recall_score REAL NOT NULL,
                    f1_score REAL NOT NULL,
                    roc_auc REAL NOT NULL,
                    train_size INTEGER NOT NULL,
                    test_size INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(actor_user_id) REFERENCES users(id) ON DELETE SET NULL
                );

                CREATE INDEX IF NOT EXISTS idx_scan_history_user_created
                    ON scan_history(user_id, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_scan_history_risk
                    ON scan_history(risk_level, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_users_email
                    ON users(email);
                CREATE INDEX IF NOT EXISTS idx_users_role_active
                    ON users(role, is_active);
                CREATE INDEX IF NOT EXISTS idx_audit_created
                    ON audit_log(created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_training_created
                    ON training_runs(created_at DESC);
                """
            )
            self._ensure_column(conn, 'users', 'role', "TEXT NOT NULL DEFAULT 'user'")
            self._ensure_column(conn, 'users', 'bio', "TEXT NOT NULL DEFAULT ''")
            self._ensure_column(conn, 'users', 'institution', "TEXT NOT NULL DEFAULT ''")
            self._ensure_column(conn, 'users', 'is_active', 'INTEGER NOT NULL DEFAULT 1')
            self._ensure_column(conn, 'users', 'last_login_at', 'TEXT')
            self._ensure_column(conn, 'scan_history', 'risk_level', "TEXT NOT NULL DEFAULT 'Low'")
            self._ensure_column(conn, 'scan_history', 'signal_count', 'INTEGER NOT NULL DEFAULT 0')
            self._ensure_column(conn, 'scan_history', 'signal_details', "TEXT NOT NULL DEFAULT ''")
            self._ensure_column(conn, 'scan_history', 'ml_top_terms', "TEXT NOT NULL DEFAULT ''")
            self._ensure_column(conn, 'scan_history', 'recommendation', "TEXT NOT NULL DEFAULT ''")
            self._ensure_column(conn, 'scan_history', 'source_type', "TEXT NOT NULL DEFAULT 'manual'")
            self._ensure_column(conn, 'scan_history', 'source_name', "TEXT NOT NULL DEFAULT ''")
            self._ensure_column(conn, 'scan_history', 'user_feedback', "TEXT NOT NULL DEFAULT ''")
        self.ensure_default_admin()

    def _ensure_column(self, conn: sqlite3.Connection, table: str, column: str, definition: str) -> None:
        existing = {row['name'] for row in conn.execute(f'PRAGMA table_info({table})').fetchall()}
        if column not in existing:
            conn.execute(f'ALTER TABLE {table} ADD COLUMN {column} {definition}')

    @staticmethod
    def hash_password(password: str, salt_hex: Optional[str] = None) -> Dict[str, str]:
        salt = bytes.fromhex(salt_hex) if salt_hex else os.urandom(16)
        hashed = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 180_000)
        return {'salt': salt.hex(), 'password_hash': hashed.hex()}

    def ensure_default_admin(self) -> None:
        with self.connect() as conn:
            admin_count = conn.execute("SELECT COUNT(*) AS count FROM users WHERE role = 'admin'").fetchone()['count']
        if admin_count:
            return
        admin_id = self.create_user(
            full_name=DEFAULT_ADMIN_NAME,
            email=DEFAULT_ADMIN_EMAIL,
            password=DEFAULT_ADMIN_PASSWORD,
            institution='PhishGuard AI',
            bio='Default administrative account created automatically for first-run access.',
            role='admin',
        )
        self.log_event(
            actor_user_id=admin_id,
            action='seed_admin',
            target_type='user',
            target_id=admin_id,
            description='Default administrator account was generated automatically during initialization.',
            severity='info',
        )

    def create_user(
        self,
        full_name: str,
        email: str,
        password: str,
        institution: str = '',
        bio: str = '',
        role: str = 'user',
    ) -> int:
        email = email.strip().lower()
        password_data = self.hash_password(password)
        role = role if role in {'user', 'admin'} else 'user'
        with self.connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO users (full_name, email, password_hash, salt, role, institution, bio, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    full_name.strip(),
                    email,
                    password_data['password_hash'],
                    password_data['salt'],
                    role,
                    institution.strip(),
                    bio.strip(),
                    utc_now_iso(),
                ),
            )
            user_id = int(cursor.lastrowid)
        return user_id

    def get_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        email = email.strip().lower()
        with self.connect() as conn:
            row = conn.execute('SELECT * FROM users WHERE email = ?', (email,)).fetchone()
        return dict(row) if row else None

    def get_user_by_id(self, user_id: int) -> Optional[Dict[str, Any]]:
        with self.connect() as conn:
            row = conn.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
        return dict(row) if row else None

    def verify_user(self, email: str, password: str) -> tuple[Optional[Dict[str, Any]], str]:
        user = self.get_user_by_email(email)
        if not user:
            return None, 'Account not found.'
        if not int(user.get('is_active', 1)):
            return None, 'This account has been deactivated by an administrator.'
        check = self.hash_password(password, salt_hex=user['salt'])
        if check['password_hash'] != user['password_hash']:
            return None, 'Invalid password.'
        with self.connect() as conn:
            conn.execute('UPDATE users SET last_login_at = ? WHERE id = ?', (utc_now_iso(), user['id']))
        return self.get_user_by_id(user['id']), 'Login successful.'

    def update_user_profile(self, user_id: int, full_name: str, institution: str, bio: str) -> None:
        with self.connect() as conn:
            conn.execute(
                'UPDATE users SET full_name = ?, institution = ?, bio = ? WHERE id = ?',
                (full_name.strip(), institution.strip(), bio.strip(), user_id),
            )

    def change_password(self, user_id: int, current_password: str, new_password: str) -> bool:
        user = self.get_user_by_id(user_id)
        if not user:
            return False
        check = self.hash_password(current_password, salt_hex=user['salt'])
        if check['password_hash'] != user['password_hash']:
            return False
        password_data = self.hash_password(new_password)
        with self.connect() as conn:
            conn.execute(
                'UPDATE users SET password_hash = ?, salt = ? WHERE id = ?',
                (password_data['password_hash'], password_data['salt'], user_id),
            )
        return True

    def admin_reset_password(self, user_id: int, new_password: str) -> bool:
        user = self.get_user_by_id(user_id)
        if not user:
            return False
        password_data = self.hash_password(new_password)
        with self.connect() as conn:
            conn.execute(
                'UPDATE users SET password_hash = ?, salt = ? WHERE id = ?',
                (password_data['password_hash'], password_data['salt'], user_id),
            )
        return True

    def set_user_active(self, user_id: int, is_active: bool) -> None:
        with self.connect() as conn:
            conn.execute('UPDATE users SET is_active = ? WHERE id = ?', (1 if is_active else 0, user_id))

    def set_user_role(self, user_id: int, role: str) -> None:
        role = role if role in {'user', 'admin'} else 'user'
        with self.connect() as conn:
            conn.execute('UPDATE users SET role = ? WHERE id = ?', (role, user_id))

    def save_scan(
        self,
        user_id: int,
        result: Dict[str, Any],
        original_text: str,
        source_type: str = 'manual',
        source_name: str = '',
    ) -> int:
        with self.connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO scan_history (
                    user_id, message_text, normalized_text, predicted_label, predicted_name,
                    confidence, phishing_probability, legitimate_probability,
                    heuristic_risk_score, risk_level, suspicious_keywords, detected_urls,
                    signal_count, signal_details, ml_top_terms, recommendation,
                    source_type, source_name, user_feedback, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    original_text,
                    result['normalized_text'],
                    result['predicted_label'],
                    result['predicted_name'],
                    float(result['confidence']),
                    float(result['phishing_probability']),
                    float(result['legitimate_probability']),
                    int(result['heuristic_risk_score']),
                    result.get('risk_level', 'Low'),
                    ', '.join(result['suspicious_keywords']),
                    '\n'.join(result['detected_urls']),
                    int(result.get('signal_count', 0)),
                    '\n'.join(result.get('signal_details', [])),
                    ', '.join(result.get('ml_top_terms', [])),
                    result.get('recommendation', ''),
                    source_type,
                    source_name,
                    result.get('user_feedback', ''),
                    utc_now_iso(),
                ),
            )
            return int(cursor.lastrowid)

    def get_history(
        self,
        user_id: int,
        limit: int = 100,
        risk_level: str = 'All',
        search: str = '',
        predicted_name: str = 'All',
    ) -> List[Dict[str, Any]]:
        query = 'SELECT * FROM scan_history WHERE user_id = ?'
        params: List[Any] = [user_id]
        if risk_level != 'All':
            query += ' AND risk_level = ?'
            params.append(risk_level)
        if predicted_name != 'All':
            query += ' AND predicted_name = ?'
            params.append(predicted_name)
        if search.strip():
            query += ' AND (message_text LIKE ? OR suspicious_keywords LIKE ? OR predicted_name LIKE ? OR recommendation LIKE ?)'
            term = f"%{search.strip()}%"
            params.extend([term, term, term, term])
        query += ' ORDER BY created_at DESC LIMIT ?'
        params.append(limit)
        with self.connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def get_scan_by_id(self, scan_id: int, user_id: int) -> Optional[Dict[str, Any]]:
        with self.connect() as conn:
            row = conn.execute('SELECT * FROM scan_history WHERE id = ? AND user_id = ?', (scan_id, user_id)).fetchone()
        return dict(row) if row else None

    def get_user_statistics(self, user_id: int) -> Dict[str, Any]:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT
                    COUNT(*) AS total_scans,
                    SUM(CASE WHEN predicted_name = 'Phishing' THEN 1 ELSE 0 END) AS phishing_count,
                    SUM(CASE WHEN predicted_name = 'Legitimate' THEN 1 ELSE 0 END) AS legitimate_count,
                    SUM(CASE WHEN risk_level = 'High' THEN 1 ELSE 0 END) AS high_risk_count,
                    AVG(confidence) AS avg_confidence,
                    AVG(heuristic_risk_score) AS avg_risk_score
                FROM scan_history
                WHERE user_id = ?
                """,
                (user_id,),
            ).fetchone()
        data = dict(row) if row else {}
        return {
            'total_scans': int(data.get('total_scans') or 0),
            'phishing_count': int(data.get('phishing_count') or 0),
            'legitimate_count': int(data.get('legitimate_count') or 0),
            'high_risk_count': int(data.get('high_risk_count') or 0),
            'avg_confidence': float(data.get('avg_confidence') or 0.0),
            'avg_risk_score': float(data.get('avg_risk_score') or 0.0),
        }

    def get_platform_statistics(self) -> Dict[str, Any]:
        with self.connect() as conn:
            user_stats = conn.execute(
                """
                SELECT
                    COUNT(*) AS total_users,
                    SUM(CASE WHEN is_active = 1 THEN 1 ELSE 0 END) AS active_users,
                    SUM(CASE WHEN role = 'admin' THEN 1 ELSE 0 END) AS admin_users
                FROM users
                """
            ).fetchone()
            scan_stats = conn.execute(
                """
                SELECT
                    COUNT(*) AS total_scans,
                    SUM(CASE WHEN predicted_name = 'Phishing' THEN 1 ELSE 0 END) AS phishing_scans,
                    SUM(CASE WHEN risk_level = 'High' THEN 1 ELSE 0 END) AS high_risk_scans,
                    AVG(confidence) AS avg_confidence
                FROM scan_history
                """
            ).fetchone()
            training_runs = conn.execute('SELECT COUNT(*) AS count FROM training_runs').fetchone()['count']
        db_size = self.db_path.stat().st_size if self.db_path.exists() else 0
        return {
            'total_users': int(user_stats['total_users'] or 0),
            'active_users': int(user_stats['active_users'] or 0),
            'admin_users': int(user_stats['admin_users'] or 0),
            'total_scans': int(scan_stats['total_scans'] or 0),
            'phishing_scans': int(scan_stats['phishing_scans'] or 0),
            'high_risk_scans': int(scan_stats['high_risk_scans'] or 0),
            'avg_confidence': float(scan_stats['avg_confidence'] or 0.0),
            'training_runs': int(training_runs or 0),
            'db_size_kb': round(db_size / 1024, 1),
        }

    def get_all_users(self, limit: int = 500, search: str = '', role: str = 'All', active: str = 'All') -> List[Dict[str, Any]]:
        query = 'SELECT * FROM users WHERE 1=1'
        params: List[Any] = []
        if role != 'All':
            query += ' AND role = ?'
            params.append(role)
        if active == 'Active':
            query += ' AND is_active = 1'
        elif active == 'Inactive':
            query += ' AND is_active = 0'
        if search.strip():
            term = f"%{search.strip()}%"
            query += ' AND (full_name LIKE ? OR email LIKE ? OR institution LIKE ?)'
            params.extend([term, term, term])
        query += ' ORDER BY created_at DESC LIMIT ?'
        params.append(limit)
        with self.connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def get_global_history(self, limit: int = 500, risk_level: str = 'All') -> List[Dict[str, Any]]:
        query = (
            'SELECT scan_history.*, users.full_name, users.email FROM scan_history '
            'JOIN users ON users.id = scan_history.user_id WHERE 1=1'
        )
        params: List[Any] = []
        if risk_level != 'All':
            query += ' AND scan_history.risk_level = ?'
            params.append(risk_level)
        query += ' ORDER BY scan_history.created_at DESC LIMIT ?'
        params.append(limit)
        with self.connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def update_feedback(self, history_id: int, user_id: int, feedback: str) -> None:
        with self.connect() as conn:
            conn.execute(
                'UPDATE scan_history SET user_feedback = ? WHERE id = ? AND user_id = ?',
                (feedback.strip(), history_id, user_id),
            )

    def clear_history(self, user_id: int) -> None:
        with self.connect() as conn:
            conn.execute('DELETE FROM scan_history WHERE user_id = ?', (user_id,))

    def delete_history_item(self, history_id: int, user_id: int) -> None:
        with self.connect() as conn:
            conn.execute('DELETE FROM scan_history WHERE id = ? AND user_id = ?', (history_id, user_id))

    def log_event(
        self,
        actor_user_id: Optional[int],
        action: str,
        description: str,
        severity: str = 'info',
        target_type: str = '',
        target_id: Optional[int] = None,
    ) -> None:
        severity = severity if severity in {'info', 'warning', 'critical'} else 'info'
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO audit_log (actor_user_id, action, target_type, target_id, severity, description, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (actor_user_id, action, target_type, target_id, severity, description, utc_now_iso()),
            )

    def get_audit_logs(self, limit: int = 200, severity: str = 'All', search: str = '') -> List[Dict[str, Any]]:
        query = (
            'SELECT audit_log.*, users.full_name AS actor_name, users.email AS actor_email '
            'FROM audit_log LEFT JOIN users ON users.id = audit_log.actor_user_id WHERE 1=1'
        )
        params: List[Any] = []
        if severity != 'All':
            query += ' AND audit_log.severity = ?'
            params.append(severity)
        if search.strip():
            term = f"%{search.strip()}%"
            query += ' AND (audit_log.action LIKE ? OR audit_log.description LIKE ? OR COALESCE(users.email, "") LIKE ?)'
            params.extend([term, term, term])
        query += ' ORDER BY audit_log.created_at DESC LIMIT ?'
        params.append(limit)
        with self.connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def record_training_run(self, summary: Any, actor_user_id: Optional[int] = None) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO training_runs (
                    actor_user_id, accuracy, precision_score, recall_score, f1_score, roc_auc, train_size, test_size, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    actor_user_id,
                    float(summary.accuracy),
                    float(summary.precision),
                    float(summary.recall),
                    float(summary.f1_score),
                    float(summary.roc_auc),
                    int(summary.train_size),
                    int(summary.test_size),
                    utc_now_iso(),
                ),
            )

    def get_training_runs(self, limit: int = 50) -> List[Dict[str, Any]]:
        query = (
            'SELECT training_runs.*, users.full_name AS actor_name, users.email AS actor_email '
            'FROM training_runs LEFT JOIN users ON users.id = training_runs.actor_user_id '
            'ORDER BY training_runs.created_at DESC LIMIT ?'
        )
        with self.connect() as conn:
            rows = conn.execute(query, (limit,)).fetchall()
        return [dict(row) for row in rows]
