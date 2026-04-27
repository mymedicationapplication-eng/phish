from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / 'data' / 'sample_phishing_dataset.csv'
ARTIFACTS_DIR = BASE_DIR / 'artifacts'
MODEL_PATH = ARTIFACTS_DIR / 'phishing_detector.joblib'
METRICS_PATH = ARTIFACTS_DIR / 'metrics.json'
STORAGE_DIR = BASE_DIR / 'storage'
DB_PATH = STORAGE_DIR / 'phishguard_ai.db'
LABEL_MAP = {0: 'Legitimate', 1: 'Phishing'}
RANDOM_STATE = 42

APP_NAME = 'PhishGuard AI Enterprise'
APP_TAGLINE = (
    'A professional phishing detection platform with authentication, SQLite persistence, '
    'role-based access control, explainable machine learning, saved history, batch analysis, '
    'analytics, and an administrative control center.'
)
MAX_HISTORY_EXPORT_ROWS = 5000
MAX_AUDIT_EXPORT_ROWS = 5000
DEFAULT_ADMIN_NAME = 'System Administrator'
DEFAULT_ADMIN_EMAIL = 'admin@phishguard.local'
DEFAULT_ADMIN_PASSWORD = 'Admin@12345'
SUPPORTED_UPLOAD_TYPES = ['txt', 'csv']
