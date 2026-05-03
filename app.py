from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List

import pandas as pd
import streamlit as st

from src.auth import assess_password_strength, login_user, register_user
from src.config import (
    APP_NAME,
    APP_TAGLINE,
    DATA_PATH,
    DEFAULT_ADMIN_EMAIL,
    DEFAULT_ADMIN_PASSWORD,
    MAX_AUDIT_EXPORT_ROWS,
    MAX_HISTORY_EXPORT_ROWS,
    METRICS_PATH,
)
from src.database import Database
from src.inference import ModelNotFoundError, clear_caches, load_metrics, predict_many, predict_text
from src.training import train_and_save_model


st.set_page_config(
    page_title=APP_NAME,
    page_icon='🛡️',
    layout='wide',
    initial_sidebar_state='collapsed',
)

css_path = Path(__file__).resolve().parent / 'app' / 'styles.css'
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


db = Database()

for key, default in {
    'user_id': None,
    'user': None,
    'last_result': None,
    'batch_results_df': None,
    'public_page': 'Landing',
    'app_page': 'Dashboard',
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


SAMPLE_MESSAGES = {
    'Benign meeting note': 'Hello team, our project review meeting has been moved to 3:00 PM tomorrow. Please bring your weekly update slides.',
    'Credential phishing email': 'Urgent verify your password now to avoid account suspension. Click http://security-check.example to continue.',
    'Invoice scam message': 'Action required. We were unable to process your invoice payment. Open the secure portal and confirm your bank details immediately.',
    'HR impersonation attempt': 'Dear employee, your payroll account is locked. Confirm your bank credentials today to avoid salary delay.',
}

APP_PAGES = [
    'Dashboard',
    'Scan Message',
    'Batch Analysis',
    'Scan History',
    'Analytics',
    'Model Metrics',
    'Platform Architecture',
    'Account',
]
ADMIN_PAGE = 'Admin Control Center'
PUBLIC_PAGES = ['Landing', 'Login', 'Register']


def set_user(user: dict | None) -> None:
    st.session_state.user = user
    st.session_state.user_id = user['id'] if user else None


@st.cache_data(show_spinner=False)
def load_dataset_preview() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


@st.cache_data(show_spinner=False)
def file_to_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


@st.cache_data(show_spinner=False)
def get_metrics() -> dict:
    return load_metrics()


def refresh_caches() -> None:
    clear_caches()
    get_metrics.clear()
    load_dataset_preview.clear()


def goto_public(page: str) -> None:
    st.session_state.public_page = page


def goto_app(page: str) -> None:
    st.session_state.app_page = page


def logout() -> None:
    user = st.session_state.user
    if user:
        db.log_event(
            actor_user_id=user['id'],
            action='logout',
            target_type='user',
            target_id=user['id'],
            description=f"User {user['email']} signed out.",
            severity='info',
        )
    set_user(None)
    goto_public('Landing')
    goto_app('Dashboard')
    st.success('You have been signed out.')
    st.rerun()


def require_admin(user: dict) -> bool:
    if user.get('role') != 'admin':
        st.error('This page is available to administrators only.')
        return False
    return True


def badge_class_for_risk(risk_level: str) -> str:
    return {
        'High': 'badge-high',
        'Medium': 'badge-medium',
    }.get(risk_level, 'badge-low')


def render_signal_chips(items: Iterable[str]) -> None:
    values = [item for item in items if item]
    if not values:
        st.write('None')
        return
    chips = ''.join([f"<span class='signal-chip'>{item}</span>" for item in values])
    st.markdown(chips, unsafe_allow_html=True)


def dataframe_download_button(label: str, dataframe: pd.DataFrame, filename: str) -> None:
    st.download_button(
        label,
        dataframe.to_csv(index=False).encode('utf-8'),
        file_name=filename,
        mime='text/csv',
        use_container_width=True,
    )


def render_shell_header() -> None:
    st.markdown(
        f"""
        <div class='shell-header'>
            <div>
                <div class='shell-brand'>{APP_NAME}</div>
                <div class='shell-subtitle'>{APP_TAGLINE}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_page_header(title: str, subtitle: str, kicker: str | None = None) -> None:
    kicker_html = f"<div class='page-kicker'>{kicker}</div>" if kicker else ''
    st.markdown(
        f"""
        <div class='page-hero'>
            {kicker_html}
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def nav_button(label: str, target: str, active: bool, key: str, public: bool = False) -> None:
    if st.button(label, key=key, use_container_width=True, type='primary' if active else 'secondary'):
        if public:
            goto_public(target)
        else:
            goto_app(target)
        st.rerun()


def render_public_topbar() -> None:
    st.markdown("<div class='topbar-shell'>", unsafe_allow_html=True)
    left, right = st.columns([1.8, 1])
    with left:
        st.markdown(
            f"""
            <div class='topbar-brand'>
                <div class='topbar-title'>{APP_NAME}</div>
                <div class='topbar-text'>Enterprise phishing detection platform</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        nav_cols = st.columns(3)
        with st.container():
            for idx, page in enumerate(PUBLIC_PAGES):
                with nav_cols[idx]:
                    nav_button(page, page, st.session_state.public_page == page, f'public_{page}', public=True)
    st.markdown("</div>", unsafe_allow_html=True)


def render_app_topbar(user: dict) -> None:
    st.markdown("""
    <div style="background: #ffffff; border-bottom: 1px solid #e2e8f0; padding: 1rem 0; margin-bottom: 2rem;">
        <div style="max-width: 1200px; margin: 0 auto; padding: 0 2rem; display: flex; justify-content: space-between; align-items: center;">
            <div style="font-size: 1.5rem; font-weight: 700; color: #2563eb;">🛡️ PhishGuard</div>
            <div style="display: flex; align-items: center; gap: 1rem;">
                <span style="color: #64748b;">Welcome, <strong>{user['full_name']}</strong></span>
                <span style="background: {'#10b981' if user.get('role') == 'admin' else '#64748b'}; color: white; padding: 0.25rem 0.75rem; border-radius: 999px; font-size: 0.75rem; font-weight: 600;">{user.get('role', 'user').title()}</span>
                <button onclick="document.querySelector('[data-testid=stButton]').click()" style="background: #ef4444; color: white; border: none; padding: 0.5rem 1rem; border-radius: 6px; cursor: pointer;">Sign Out</button>
            </div>
        </div>
    </div>
    """.format(user=user), unsafe_allow_html=True)

    # Navigation
    all_pages = APP_PAGES.copy()
    if user.get('role') == 'admin':
        all_pages.insert(-1, ADMIN_PAGE)

    st.markdown("<div style=\"display: flex; gap: 0.5rem; flex-wrap: wrap; margin-bottom: 2rem; max-width: 1200px; margin-left: auto; margin-right: auto; padding: 0 2rem;\">", unsafe_allow_html=True)
    for page in all_pages:
        short_label = page.replace('Control Center', 'Admin')
        active = st.session_state.app_page == page
        button_type = 'primary' if active else 'secondary'
        if st.button(short_label, key=f'app_{page}', use_container_width=False, type=button_type):
            goto_app(page)
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)


def build_scan_report(result: dict, original_text: str) -> str:
    return (
        f"Prediction: {result['predicted_name']}\n"
        f"Confidence: {result['confidence']:.2%}\n"
        f"Phishing Probability: {result['phishing_probability']:.2%}\n"
        f"Legitimate Probability: {result['legitimate_probability']:.2%}\n"
        f"Risk Level: {result['risk_level']}\n"
        f"Risk Score: {result['heuristic_risk_score']}/100\n"
        f"Signal Count: {result['signal_count']}\n\n"
        f"Explanation:\n{result['explanation']}\n\n"
        f"Recommendation:\n{result['recommendation']}\n\n"
        f"Suspicious Keywords:\n{', '.join(result['suspicious_keywords']) or 'None'}\n\n"
        f"Detected URLs:\n{chr(10).join(result['detected_urls']) or 'None'}\n\n"
        f"Influential Model Terms:\n{', '.join(result['ml_top_terms']) or 'None'}\n\n"
        f"Original Text:\n{original_text}\n"
    )


def render_scan_result(result: dict, original_text: str) -> None:
    st.markdown(
        f"<div class='{badge_class_for_risk(result['risk_level'])}'>{result['predicted_name']} | {result['risk_level']} Risk</div>",
        unsafe_allow_html=True,
    )
    cols = st.columns(5)
    cols[0].metric('Confidence', f"{result['confidence']:.2%}")
    cols[1].metric('Phishing Probability', f"{result['phishing_probability']:.2%}")
    cols[2].metric('Legitimate Probability', f"{result['legitimate_probability']:.2%}")
    cols[3].metric('Risk Score', f"{result['heuristic_risk_score']}/100")
    cols[4].metric('Signals', result['signal_count'])

    st.info(result['explanation'])

    prob_df = pd.DataFrame(
        {
            'Class': ['Legitimate', 'Phishing'],
            'Probability': [result['legitimate_probability'], result['phishing_probability']],
        }
    )
    left, right = st.columns([1.2, 1])
    with left:
        st.subheader('Probability Breakdown')
        st.bar_chart(prob_df.set_index('Class'))
        st.subheader('Action Recommendation')
        st.write(result['recommendation'])
    with right:
        st.subheader('Rule-Based Signals')
        for detail in result['signal_details']:
            st.write(f'- {detail}')
        st.subheader('Suspicious Keywords')
        render_signal_chips(result['suspicious_keywords'])
        st.subheader('Influential Model Terms')
        render_signal_chips(result['ml_top_terms'])
        st.subheader('Detected URLs')
        if result['detected_urls']:
            for url in result['detected_urls']:
                st.code(url)
        else:
            st.write('None')

    report_text = build_scan_report(result, original_text)
    export_cols = st.columns(2)
    export_cols[0].download_button(
        'Download scan report',
        report_text.encode('utf-8'),
        file_name='scan_report.txt',
        mime='text/plain',
        use_container_width=True,
    )
    export_cols[1].download_button(
        'Download raw result JSON',
        json.dumps(result, indent=2).encode('utf-8'),
        file_name='scan_result.json',
        mime='application/json',
        use_container_width=True,
    )


def render_landing() -> None:
    metrics = get_metrics()
    platform_stats = db.get_platform_statistics()

    # Header
    st.markdown("""
    <div class="header">
        <div class="header-content">
            <div class="logo">🛡️ PhishGuard</div>
            <nav class="nav-links">
                <a href="#" class="nav-link" onclick="document.querySelector('[data-testid=stButton]').click()">Home</a>
                <a href="#" class="nav-link">Features</a>
                <a href="#" class="nav-link">About</a>
                <a href="#" class="nav-link">Contact</a>
            </nav>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Hero Section
    st.markdown("""
    <div class="hero">
        <h1 class="hero-title">Advanced Phishing Detection Platform</h1>
        <p class="hero-subtitle">Professional enterprise-grade solution for detecting and preventing phishing attacks with AI-powered analysis and comprehensive security monitoring.</p>
        <div class="cta-buttons">
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button('🚀 Get Started', use_container_width=True, type='primary'):
            goto_public('Register')
            st.rerun()
    with col2:
        if st.button('🔐 Login', use_container_width=True):
            goto_public('Login')
            st.rerun()
    with col3:
        if st.button('📊 Platform Demo', use_container_width=True):
            goto_public('Landing')
            st.rerun()

    st.markdown("</div></div>", unsafe_allow_html=True)

    # Stats Section
    st.markdown("<div class=\"stats-grid\">", unsafe_allow_html=True)
    stat_cols = st.columns(4)
    stat_cols[0].metric('Active Users', platform_stats['active_users'])
    stat_cols[1].metric('Threats Detected', platform_stats['high_risk_scans'])
    stat_cols[2].metric('Accuracy Rate', f"{metrics.get('accuracy', 0):.1%}" if metrics else 'N/A')
    stat_cols[3].metric('Total Scans', platform_stats['total_scans'])
    st.markdown("</div>", unsafe_allow_html=True)

    # Features Section
    st.markdown("<div class=\"feature-grid\">", unsafe_allow_html=True)

    features = [
        {
            "icon": "🤖",
            "title": "AI-Powered Detection",
            "description": "Advanced machine learning algorithms analyze messages with high accuracy, detecting sophisticated phishing attempts that traditional methods miss."
        },
        {
            "icon": "📊",
            "title": "Real-time Analytics",
            "description": "Comprehensive dashboards and reporting tools provide insights into security trends, user behavior, and threat patterns."
        },
        {
            "icon": "🔒",
            "title": "Enterprise Security",
            "description": "Role-based access control, audit logging, and secure authentication ensure your organization's data remains protected."
        },
        {
            "icon": "⚡",
            "title": "Batch Processing",
            "description": "Process thousands of messages simultaneously with our efficient batch analysis tools, perfect for large-scale security operations."
        }
    ]

    for feature in features:
        st.markdown(f"""
        <div class="feature-card">
            <h3>{feature['icon']} {feature['title']}</h3>
            <p>{feature['description']}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


def render_login_page() -> None:
    st.markdown("""
    <div class="header">
        <div class="header-content">
            <div class="logo">🛡️ PhishGuard</div>
            <nav class="nav-links">
                <a href="#" class="nav-link">Home</a>
                <a href="#" class="nav-link">Features</a>
                <a href="#" class="nav-link">About</a>
            </nav>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class=\"page-container\">", unsafe_allow_html=True)
    st.markdown("<div class=\"page-header\"><h1 class=\"page-title\">Welcome Back</h1><p class=\"page-subtitle\">Sign in to access your phishing detection dashboard</p></div>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("<div class=\"card\">", unsafe_allow_html=True)
        st.markdown("<h2 class=\"card-title\">Sign In</h2>", unsafe_allow_html=True)

        with st.form('login_form', clear_on_submit=False):
            st.markdown("<div class=\"form-group\"><label class=\"form-label\">Email Address</label>", unsafe_allow_html=True)
            email = st.text_input('', placeholder='your.email@example.com', label_visibility='collapsed')
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div class=\"form-group\"><label class=\"form-label\">Password</label>", unsafe_allow_html=True)
            password = st.text_input('', type='password', placeholder='Enter your password', label_visibility='collapsed')
            st.markdown("</div>", unsafe_allow_html=True)

            submitted = st.form_submit_button('Sign In', type='primary', use_container_width=True)
            if submitted:
                ok, message, user = login_user(db, email, password)
                if ok and user:
                    set_user(user)
                    goto_app('Dashboard')
                    st.success(message)
                    st.rerun()
                st.error(message)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<p style=\"text-align: center; margin-top: 1rem;\">Don't have an account? <a href=\"#\" onclick=\"document.querySelector('[data-testid=stButton]').click()\">Create one here</a></p>", unsafe_allow_html=True)
        if st.button('Create Account', key='go_register_from_login'):
            goto_public('Register')
            st.rerun()

    with col2:
        st.markdown("""
        <div class="card">
            <h3 class="card-title">Platform Benefits</h3>
            <ul style="list-style: none; padding: 0;">
                <li style="margin-bottom: 1rem; display: flex; align-items: center;">
                    <span style="color: #10b981; margin-right: 0.5rem;">✓</span>
                    Advanced AI-powered threat detection
                </li>
                <li style="margin-bottom: 1rem; display: flex; align-items: center;">
                    <span style="color: #10b981; margin-right: 0.5rem;">✓</span>
                    Real-time security analytics
                </li>
                <li style="margin-bottom: 1rem; display: flex; align-items: center;">
                    <span style="color: #10b981; margin-right: 0.5rem;">✓</span>
                    Comprehensive audit trails
                </li>
                <li style="margin-bottom: 1rem; display: flex; align-items: center;">
                    <span style="color: #10b981; margin-right: 0.5rem;">✓</span>
                    Enterprise-grade security
                </li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


def render_register_page() -> None:
    st.markdown("""
    <div class="header">
        <div class="header-content">
            <div class="logo">🛡️ PhishGuard</div>
            <nav class="nav-links">
                <a href="#" class="nav-link">Home</a>
                <a href="#" class="nav-link">Features</a>
                <a href="#" class="nav-link">About</a>
            </nav>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class=\"page-container\">", unsafe_allow_html=True)
    st.markdown("<div class=\"page-header\"><h1 class=\"page-title\">Create Your Account</h1><p class=\"page-subtitle\">Join our platform to start detecting phishing threats</p></div>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("<div class=\"card\">", unsafe_allow_html=True)
        st.markdown("<h2 class=\"card-title\">Sign Up</h2>", unsafe_allow_html=True)

        with st.form('register_form', clear_on_submit=True):
            st.markdown("<div class=\"form-group\"><label class=\"form-label\">Full Name</label>", unsafe_allow_html=True)
            full_name = st.text_input('', placeholder='Your full name', label_visibility='collapsed')
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div class=\"form-group\"><label class=\"form-label\">Institution</label>", unsafe_allow_html=True)
            institution = st.text_input('', placeholder='University or organization', label_visibility='collapsed')
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div class=\"form-group\"><label class=\"form-label\">Email Address</label>", unsafe_allow_html=True)
            email = st.text_input('', placeholder='your.email@example.com', label_visibility='collapsed')
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div class=\"form-group\"><label class=\"form-label\">Short Bio (Optional)</label>", unsafe_allow_html=True)
            bio = st.text_area('', placeholder='Brief description of your role', height=80, label_visibility='collapsed')
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div class=\"form-group\"><label class=\"form-label\">Password</label>", unsafe_allow_html=True)
            password = st.text_input('', type='password', placeholder='Create a strong password', label_visibility='collapsed')
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div class=\"form-group\"><label class=\"form-label\">Confirm Password</label>", unsafe_allow_html=True)
            confirm_password = st.text_input('', type='password', placeholder='Confirm your password', label_visibility='collapsed')
            st.markdown("</div>", unsafe_allow_html=True)

            submitted = st.form_submit_button('Create Account', type='primary', use_container_width=True)
            if submitted:
                ok, message, user = register_user(db, full_name, email, password, confirm_password, institution, bio)
                if ok and user:
                    set_user(user)
                    goto_app('Dashboard')
                    st.success(message)
                    st.rerun()
                st.error(message)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<p style=\"text-align: center; margin-top: 1rem;\">Already have an account? <a href=\"#\" onclick=\"document.querySelector('[data-testid=stButton]').click()\">Sign in here</a></p>", unsafe_allow_html=True)
        if st.button('Sign In', key='go_login_from_register'):
            goto_public('Login')
            st.rerun()

    with col2:
        st.markdown("""
        <div class="card">
            <h3 class="card-title">Password Requirements</h3>
            <ul style="list-style: none; padding: 0;">
                <li style="margin-bottom: 0.5rem; display: flex; align-items: center;">
                    <span style="color: #10b981; margin-right: 0.5rem;">✓</span>
                    At least 8 characters long
                </li>
                <li style="margin-bottom: 0.5rem; display: flex; align-items: center;">
                    <span style="color: #10b981; margin-right: 0.5rem;">✓</span>
                    Include uppercase letters
                </li>
                <li style="margin-bottom: 0.5rem; display: flex; align-items: center;">
                    <span style="color: #10b981; margin-right: 0.5rem;">✓</span>
                    Include lowercase letters
                </li>
                <li style="margin-bottom: 0.5rem; display: flex; align-items: center;">
                    <span style="color: #10b981; margin-right: 0.5rem;">✓</span>
                    Include numbers
                </li>
                <li style="margin-bottom: 0.5rem; display: flex; align-items: center;">
                    <span style="color: #10b981; margin-right: 0.5rem;">✓</span>
                    Include special characters
                </li>
            </ul>
            <hr style="margin: 1rem 0; border: none; border-top: 1px solid #e2e8f0;">
            <p style="color: #64748b; font-size: 0.875rem;">Your account will be created with user privileges. Administrators can upgrade your access level if needed.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


def render_dashboard(user: dict) -> None:
    metrics = get_metrics()
    history = db.get_history(user['id'], limit=400)
    stats_data = db.get_user_statistics(user['id'])

    st.markdown("<div class=\"page-container\">", unsafe_allow_html=True)
    st.markdown("<div class=\"page-header\"><h1 class=\"page-title\">Dashboard</h1><p class=\"page-subtitle\">Welcome back! Here's your security overview and recent activity.</p></div>", unsafe_allow_html=True)

    # Stats Cards
    st.markdown("<div class=\"stats-grid\">", unsafe_allow_html=True)
    stat_cols = st.columns(6)
    stat_cols[0].metric('Total Scans', stats_data['total_scans'])
    stat_cols[1].metric('Phishing Detected', stats_data['phishing_count'])
    stat_cols[2].metric('Safe Messages', stats_data['legitimate_count'])
    stat_cols[3].metric('High Risk Alerts', stats_data['high_risk_count'])
    stat_cols[4].metric('Avg Confidence', f"{stats_data['avg_confidence']:.1%}" if stats_data['total_scans'] else 'N/A')
    stat_cols[5].metric('Model Accuracy', f"{metrics.get('accuracy', 0):.1%}" if metrics else 'N/A')
    st.markdown("</div>", unsafe_allow_html=True)

    col1, col2 = st.columns([1.5, 1])
    with col1:
        st.markdown("<div class=\"card\"><h3 class=\"card-title\">Activity Timeline</h3>", unsafe_allow_html=True)
        if history:
            hist_df = pd.DataFrame(history)
            chart_df = (
                hist_df.assign(created_date=pd.to_datetime(hist_df['created_at']).dt.date)
                .groupby(['created_date', 'predicted_name'])
                .size()
                .unstack(fill_value=0)
            )
            st.line_chart(chart_df)
            st.markdown("<h4>Recent Scans</h4>", unsafe_allow_html=True)
            display_df = hist_df[['created_at', 'predicted_name', 'risk_level', 'confidence', 'source_type']].head(12)
            st.dataframe(display_df, use_container_width=True)
        else:
            st.info('No scans yet. Start by analyzing a message!')
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="card">
            <h3 class="card-title">Quick Actions</h3>
            <div style="display: flex; flex-direction: column; gap: 1rem;">
        """, unsafe_allow_html=True)

        if st.button('🔍 Scan New Message', use_container_width=True):
            goto_app('Scan Message')
            st.rerun()

        if st.button('📊 Batch Analysis', use_container_width=True):
            goto_app('Batch Analysis')
            st.rerun()

        if st.button('📈 View Analytics', use_container_width=True):
            goto_app('Analytics')
            st.rerun()

        st.markdown("</div></div>", unsafe_allow_html=True)

        st.markdown("""
        <div class="card">
            <h3 class="card-title">System Status</h3>
            <p style="color: #64748b; margin-bottom: 1rem;">All systems operational</p>
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span>AI Model</span>
                <span style="color: #10b981;">● Active</span>
            </div>
            <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 0.5rem;">
                <span>Database</span>
                <span style="color: #10b981;">● Connected</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button('🔄 Retrain Model', use_container_width=True):
            with st.spinner('Retraining AI model...'):
                _, summary = train_and_save_model()
                db.record_training_run(summary, actor_user_id=user['id'])
                db.log_event(
                    actor_user_id=user['id'],
                    action='train_model',
                    target_type='model',
                    description=(
                        f'Model retrained with accuracy {summary.accuracy:.2%}, F1 score {summary.f1_score:.2%}, '
                        f'and ROC AUC {summary.roc_auc:.2%}.'
                    ),
                    severity='warning',
                )
                refresh_caches()
            st.success(
                f"Model updated! Accuracy: {summary.accuracy:.1%} | F1-score: {summary.f1_score:.1%} | ROC-AUC: {summary.roc_auc:.1%}"
            )
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


def render_scan_page(user: dict) -> None:
    st.markdown("<div class=\"page-container\">", unsafe_allow_html=True)
    st.markdown("<div class=\"page-header\"><h1 class=\"page-title\">Message Scanner</h1><p class=\"page-subtitle\">Analyze individual messages for phishing threats with detailed AI-powered insights.</p></div>", unsafe_allow_html=True)

    st.markdown("<div class=\"card\">", unsafe_allow_html=True)
    st.markdown("<h3 class=\"card-title\">Message Analysis</h3>", unsafe_allow_html=True)

    chosen_sample = st.selectbox('Choose a sample message', ['Custom Message'] + list(SAMPLE_MESSAGES.keys()))
    default_text = SAMPLE_MESSAGES.get(chosen_sample, '') if chosen_sample != 'Custom Message' else ''

    message_input = st.text_area(
        'Enter message content',
        height=200,
        placeholder='Paste your message or email content here for analysis...',
        value=default_text,
        label_visibility='collapsed'
    )

    uploaded = st.file_uploader('Or upload a text file', type=['txt'], key='single_upload')

    if uploaded is not None:
        try:
            message_input = uploaded.read().decode('utf-8', errors='ignore')
            st.text_area('Uploaded content preview', value=message_input, height=150, disabled=True)
        except Exception:
            st.error('Could not read the uploaded file as UTF-8 text.')

    col1, col2, col3 = st.columns([1, 1, 2])
    analyze = col1.button('🔍 Analyze Message', type='primary', use_container_width=True)
    save_toggle = col2.checkbox('Save to history', value=True)
    col3.caption('Analysis reports can be downloaded after scanning.')

    if analyze and message_input.strip():
        try:
            result = predict_text(message_input)
            st.session_state.last_result = {'result': result, 'original_text': message_input}
            if save_toggle:
                scan_id = db.save_scan(user['id'], result, message_input, source_type='manual', source_name='message scanner')
                db.log_event(
                    actor_user_id=user['id'],
                    action='scan_message',
                    target_type='scan',
                    target_id=scan_id,
                    description=f"User analyzed a message and received {result['predicted_name']} with {result['risk_level']} risk.",
                    severity='warning' if result['risk_level'] in {'High', 'Medium'} else 'info',
                )
                st.success('✅ Analysis complete and saved to your history.')
            else:
                st.success('✅ Analysis complete.')

            # Display results in a professional card layout
            st.markdown("<div class=\"card\" style=\"margin-top: 2rem;\">", unsafe_allow_html=True)
            st.markdown(f"<div style=\"display: flex; align-items: center; gap: 1rem; margin-bottom: 1rem;\"><span style=\"font-size: 2rem;\">{ '🚨' if result['risk_level'] == 'High' else '⚠️' if result['risk_level'] == 'Medium' else '✅' }</span><h3 style=\"margin: 0; color: {'#ef4444' if result['risk_level'] == 'High' else '#f59e0b' if result['risk_level'] == 'Medium' else '#10b981'};\">{result['predicted_name']} - {result['risk_level']} Risk</h3></div>", unsafe_allow_html=True)

            # Metrics grid
            st.markdown("<div class=\"stats-grid\">", unsafe_allow_html=True)
            cols = st.columns(5)
            cols[0].metric('Confidence', f"{result['confidence']:.1%}")
            cols[1].metric('Phishing Prob', f"{result['phishing_probability']:.1%}")
            cols[2].metric('Safe Prob', f"{result['legitimate_probability']:.1%}")
            cols[3].metric('Risk Score', f"{result['heuristic_risk_score']}/100")
            cols[4].metric('Signals Found', result['signal_count'])
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown(f"<p style=\"color: #64748b; margin: 1rem 0;\"><strong>Recommendation:</strong> {result['recommendation']}</p>", unsafe_allow_html=True)

            col_left, col_right = st.columns([1.2, 1])
            with col_left:
                st.markdown("<h4>📋 Detection Signals</h4>", unsafe_allow_html=True)
                for detail in result['signal_details']:
                    st.write(f"• {detail}")
                st.markdown("<h4>🔍 Suspicious Keywords</h4>", unsafe_allow_html=True)
                if result['suspicious_keywords']:
                    for keyword in result['suspicious_keywords']:
                        st.code(keyword, language=None)
                else:
                    st.write("None detected")

            with col_right:
                st.markdown("<h4>📊 Probability Breakdown</h4>", unsafe_allow_html=True)
                prob_df = pd.DataFrame({
                    'Category': ['Safe', 'Phishing'],
                    'Probability': [result['legitimate_probability'], result['phishing_probability']]
                })
                st.bar_chart(prob_df.set_index('Category'))

                st.markdown("<h4>🎯 Model Insights</h4>", unsafe_allow_html=True)
                if result['ml_top_terms']:
                    for term in result['ml_top_terms']:
                        st.code(term, language=None)
                else:
                    st.write("No significant terms")

                st.markdown("<h4>🔗 Detected URLs</h4>", unsafe_allow_html=True)
                if result['detected_urls']:
                    for url in result['detected_urls']:
                        st.code(url, language=None)
                else:
                    st.write("None found")

            # Download options
            st.markdown("<hr style=\"margin: 2rem 0;\">", unsafe_allow_html=True)
            download_cols = st.columns(2)
            report_text = build_scan_report(result, message_input)
            download_cols[0].download_button(
                '📄 Download Report',
                report_text.encode('utf-8'),
                file_name='phishguard_scan_report.txt',
                mime='text/plain',
                use_container_width=True,
            )
            download_cols[1].download_button(
                '📊 Download JSON',
                json.dumps(result, indent=2).encode('utf-8'),
                file_name='phishguard_scan_result.json',
                mime='application/json',
                use_container_width=True,
            )

            st.markdown("</div>", unsafe_allow_html=True)

        except ModelNotFoundError as exc:
            st.error(f"❌ {str(exc)}")
        except ValueError as exc:
            st.error(f"❌ {str(exc)}")
    elif analyze:
        st.warning("⚠️ Please enter a message to analyze.")

    elif st.session_state.last_result:
        st.markdown("### 📋 Last Analysis Result")
        result = st.session_state.last_result['result']
        st.markdown(f"<div style=\"padding: 1rem; border-radius: 8px; background: {'#fef2f2' if result['risk_level'] == 'High' else '#fefce8' if result['risk_level'] == 'Medium' else '#f0fdf4'}; border: 1px solid {'#fecaca' if result['risk_level'] == 'High' else '#fde047' if result['risk_level'] == 'Medium' else '#bbf7d0'};\">", unsafe_allow_html=True)
        st.markdown(f"<strong>{result['predicted_name']} - {result['risk_level']} Risk</strong> | Confidence: {result['confidence']:.1%}", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div></div>", unsafe_allow_html=True)


def render_batch_page(user: dict) -> None:
    st.markdown("<div class=\"page-container\">", unsafe_allow_html=True)
    st.markdown("<div class=\"page-header\"><h1 class=\"page-title\">Batch Analysis</h1><p class=\"page-subtitle\">Process multiple messages simultaneously for efficient threat detection at scale.</p></div>", unsafe_allow_html=True)

    st.markdown("<div class=\"card\">", unsafe_allow_html=True)
    st.markdown("<h3 class=\"card-title\">📁 File Upload & Processing</h3>", unsafe_allow_html=True)

    uploaded = st.file_uploader('Upload CSV or TXT file for batch analysis', type=['csv', 'txt'], key='batch_upload')
    save_results = st.checkbox('💾 Save all results to history', value=False)
    rows_to_process = st.slider('📊 Maximum rows to process', 5, 250, 50, step=5, help="Limit processing for large files to avoid timeouts")

    texts: List[str] = []
    source_name = ''

    if uploaded is not None:
        source_name = uploaded.name
        if uploaded.name.lower().endswith('.csv'):
            try:
                df = pd.read_csv(uploaded)
                st.success(f"✅ CSV loaded with {len(df)} rows")
                st.markdown("<h4>📋 Data Preview</h4>", unsafe_allow_html=True)
                st.dataframe(df.head(10), use_container_width=True)
                column_choice = st.selectbox('🎯 Select the text column to analyze', list(df.columns))
                if column_choice:
                    texts = [str(value) for value in df[column_choice].dropna().astype(str).tolist()[:rows_to_process]]
                    st.info(f"📝 Ready to process {len(texts)} messages from '{column_choice}' column")
            except Exception as exc:
                st.error(f'❌ Could not read CSV file: {exc}')
        else:
            try:
                text_content = uploaded.read().decode('utf-8', errors='ignore')
                texts = file_to_lines(text_content)[:rows_to_process]
                st.success(f"✅ TXT file loaded with {len(texts)} messages")
                st.markdown("<h4>📋 Content Preview</h4>", unsafe_allow_html=True)
                st.text_area('First 20 lines preview', value='\n'.join(texts[:20]), height=150, disabled=True)
            except Exception as exc:
                st.error(f'❌ Could not read TXT file: {exc}')

    if st.button('🚀 Start Batch Analysis', type='primary', disabled=not texts, use_container_width=True):
        try:
            with st.spinner('🔄 Analyzing messages... Please wait.'):
                results = predict_many(texts)
            if not results:
                st.warning('⚠️ No valid text rows were found to analyze.')
                return

            records = []
            for original_text, result in zip(texts, results):
                records.append({
                    'text': original_text,
                    'prediction': result['predicted_name'],
                    'confidence': round(result['confidence'], 4),
                    'phishing_probability': round(result['phishing_probability'], 4),
                    'risk_level': result['risk_level'],
                    'risk_score': result['heuristic_risk_score'],
                    'signals': '; '.join(result['signal_details']),
                    'keywords': ', '.join(result['suspicious_keywords']),
                })
                if save_results:
                    scan_id = db.save_scan(user['id'], result, original_text, source_type='batch', source_name=source_name)
                    db.log_event(
                        actor_user_id=user['id'],
                        action='batch_scan_item',
                        target_type='scan',
                        target_id=scan_id,
                        description=f"Batch scan saved from {source_name or 'uploaded file'} with result {result['predicted_name']}.",
                        severity='warning' if result['risk_level'] in {'High', 'Medium'} else 'info',
                    )

            results_df = pd.DataFrame(records)
            st.session_state.batch_results_df = results_df

            # Summary stats
            total_scans = len(results_df)
            phishing_count = (results_df['prediction'] == 'Phishing').sum()
            high_risk_count = (results_df['risk_level'] == 'High').sum()

            st.success(f'✅ Batch analysis completed! Processed {total_scans} messages.')

            # Results summary
            st.markdown("<div class=\"stats-grid\">", unsafe_allow_html=True)
            cols = st.columns(4)
            cols[0].metric('Total Messages', total_scans)
            cols[1].metric('Phishing Detected', phishing_count)
            cols[2].metric('Safe Messages', total_scans - phishing_count)
            cols[3].metric('High Risk Alerts', high_risk_count)
            st.markdown("</div>", unsafe_allow_html=True)

            # Results table
            st.markdown("<h4>📊 Detailed Results</h4>", unsafe_allow_html=True)
            display_cols = ['prediction', 'confidence', 'phishing_probability', 'risk_level', 'risk_score', 'signals']
            st.dataframe(results_df[display_cols], use_container_width=True)

            # Download options
            download_cols = st.columns(2)
            download_cols[0].download_button(
                '📄 Download CSV Report',
                results_df.to_csv(index=False).encode('utf-8'),
                file_name='batch_analysis_results.csv',
                mime='text/csv',
                use_container_width=True,
            )
            download_cols[1].download_button(
                '📊 Download JSON Data',
                json.dumps(records, indent=2).encode('utf-8'),
                file_name='batch_analysis_results.json',
                mime='application/json',
                use_container_width=True,
            )

        except ModelNotFoundError as exc:
            st.error(f"❌ {str(exc)}")
        except Exception as exc:
            st.error(f"❌ Analysis failed: {str(exc)}")

    if isinstance(st.session_state.batch_results_df, pd.DataFrame):
        st.markdown("### 📋 Last Batch Run Results")
        st.dataframe(st.session_state.batch_results_df, use_container_width=True)

    st.markdown("</div></div>", unsafe_allow_html=True)


def render_history_page(user: dict) -> None:
    st.markdown("<div class=\"page-container\">", unsafe_allow_html=True)
    st.markdown("<div class=\"page-header\"><h1 class=\"page-title\">Scan History</h1><p class=\"page-subtitle\">Browse, search, and manage your previously analyzed messages and results.</p></div>", unsafe_allow_html=True)

    st.markdown("<div class=\"card\">", unsafe_allow_html=True)
    st.markdown("<h3 class=\"card-title\">🔍 Search & Filter</h3>", unsafe_allow_html=True)

    filters = st.columns([1, 1, 1, 1, 2])
    risk_filter = filters[0].selectbox('Risk Level', ['All', 'High', 'Medium', 'Low'])
    class_filter = filters[1].selectbox('Prediction', ['All', 'Phishing', 'Legitimate'])
    max_rows = filters[2].selectbox('Rows to Show', [25, 50, 100, 250, 500], index=2)
    export_limit = filters[3].slider('Export Limit', 25, MAX_HISTORY_EXPORT_ROWS, min(500, MAX_HISTORY_EXPORT_ROWS), step=25)
    search_term = filters[4].text_input('Search', placeholder='keyword, recommendation, or class...')

    history = db.get_history(user['id'], limit=max_rows, risk_level=risk_filter, search=search_term, predicted_name=class_filter)
    if not history:
        st.info('⚠️ No scans match your current filters. Try adjusting your search criteria.')
        st.markdown("</div></div>", unsafe_allow_html=True)
        return

    history_df = pd.DataFrame(history)
    display_columns = [
        'created_at', 'predicted_name', 'risk_level', 'confidence', 'heuristic_risk_score',
        'source_type', 'source_name', 'suspicious_keywords', 'recommendation', 'user_feedback'
    ]

    st.markdown("<h4>📋 Scan Results</h4>", unsafe_allow_html=True)
    st.dataframe(history_df[display_columns], use_container_width=True)

    export_df = pd.DataFrame(
        db.get_history(user['id'], limit=export_limit, risk_level=risk_filter, search=search_term, predicted_name=class_filter)
    )
    download_cols = st.columns(2)
    download_cols[0].download_button(
        '📄 Export as CSV',
        export_df.to_csv(index=False).encode('utf-8'),
        file_name='phishguard_history.csv',
        mime='text/csv',
        use_container_width=True,
    )
    download_cols[1].download_button(
        '📊 Export as JSON',
        export_df.to_json(orient='records', indent=2).encode('utf-8'),
        file_name='phishguard_history.json',
        mime='application/json',
        use_container_width=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)

    # Individual Management
    st.markdown("<div class=\"card\">", unsafe_allow_html=True)
    st.markdown("<h3 class=\"card-title\">⚙️ Manage Individual Scans</h3>", unsafe_allow_html=True)

    options = {f"#{row['id']} | {row['created_at']} | {row['predicted_name']} | {row['risk_level']}": row['id'] for row in history}
    selected_label = st.selectbox('Select a scan to manage', list(options.keys()))
    selected_id = options[selected_label]
    selected_row = next(row for row in history if row['id'] == selected_id)

    st.markdown("<h4>📝 Original Message</h4>", unsafe_allow_html=True)
    st.text_area('Message content', value=selected_row['message_text'], height=150, disabled=True)

    feedback = st.text_area('Add or update feedback', value=selected_row.get('user_feedback', ''), height=80)

    action_cols = st.columns(3)
    if action_cols[0].button('💾 Save Feedback', key=f'save_{selected_id}', use_container_width=True):
        db.update_feedback(selected_id, user['id'], feedback)
        db.log_event(
            actor_user_id=user['id'],
            action='update_feedback',
            target_type='scan',
            target_id=selected_id,
            description=f'Feedback updated for saved scan #{selected_id}.',
            severity='info',
        )
        st.success('✅ Feedback updated.')
        st.rerun()

    if action_cols[1].button('🗑️ Delete Scan', key=f'delete_{selected_id}', use_container_width=True):
        db.delete_history_item(selected_id, user['id'])
        db.log_event(
            actor_user_id=user['id'],
            action='delete_scan',
            target_type='scan',
            target_id=selected_id,
            description=f'Saved scan #{selected_id} was deleted.',
            severity='warning',
        )
        st.success('✅ Selected scan deleted.')
        st.rerun()

    if action_cols[2].button('🗑️ Clear All History', key='clear_all_history', use_container_width=True):
        db.clear_history(user['id'])
        db.log_event(
            actor_user_id=user['id'],
            action='clear_history',
            target_type='scan',
            description='User cleared all personal scan history.',
            severity='warning',
        )
        st.success('✅ All history entries were removed.')
        st.rerun()

    st.markdown("</div></div>", unsafe_allow_html=True)


def render_analytics_page(user: dict) -> None:
    st.markdown("<div class=\"page-container\">", unsafe_allow_html=True)
    st.markdown("<div class=\"page-header\"><h1 class=\"page-title\">Analytics Dashboard</h1><p class=\"page-subtitle\">Comprehensive insights into your scanning patterns, trends, and security metrics.</p></div>", unsafe_allow_html=True)

    history = db.get_history(user['id'], limit=1000)
    if not history:
        st.markdown("<div class=\"card\"><p style=\"text-align: center; color: #64748b; padding: 2rem;\">📊 Analytics will appear after you save some scan results. Start by analyzing messages!</p></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        return

    df = pd.DataFrame(history)

    # Overview Stats
    st.markdown("<div class=\"stats-grid\">", unsafe_allow_html=True)
    top = st.columns(4)
    top[0].metric('Messages Analyzed', len(df))
    top[1].metric('High Risk Rate', f"{(df['risk_level'].eq('High').mean() if len(df) else 0):.1%}")
    top[2].metric('Phishing Detection Rate', f"{(df['predicted_name'].eq('Phishing').mean() if len(df) else 0):.1%}")
    top[3].metric('Average Confidence', f"{df['confidence'].mean():.1%}")
    st.markdown("</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class=\"card\"><h3 class=\"card-title\">📈 Risk Level Distribution</h3>", unsafe_allow_html=True)
        risk_chart = df['risk_level'].value_counts()
        st.bar_chart(risk_chart)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class=\"card\"><h3 class=\"card-title\">📊 Average Confidence by Class</h3>", unsafe_allow_html=True)
        confidence_by_class = df.groupby('predicted_name')['confidence'].mean()
        st.bar_chart(confidence_by_class)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class=\"card\"><h3 class=\"card-title\">📉 Average Risk Score by Class</h3>", unsafe_allow_html=True)
        risk_score_by_class = df.groupby('predicted_name')['heuristic_risk_score'].mean()
        st.bar_chart(risk_score_by_class)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class=\"card\"><h3 class=\"card-title\">📅 Daily Activity Timeline</h3>", unsafe_allow_html=True)
        timeline = (
            df.assign(created_date=pd.to_datetime(df['created_at']).dt.date)
            .groupby(['created_date', 'risk_level'])
            .size()
            .unstack(fill_value=0)
        )
        st.line_chart(timeline)
        st.markdown("</div>", unsafe_allow_html=True)

    # Suspicious Keywords Analysis
    keyword_series = (
        df['suspicious_keywords']
        .fillna('')
        .str.split(', ')
        .explode()
        .str.strip()
    )
    keyword_series = keyword_series[keyword_series.astype(bool)]
    if not keyword_series.empty:
        st.markdown("<div class=\"card\"><h3 class=\"card-title\">🔍 Most Frequent Suspicious Keywords</h3>", unsafe_allow_html=True)
        top_keywords = keyword_series.value_counts().head(15)
        st.bar_chart(top_keywords)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


def render_metrics_page(user: dict) -> None:
    render_page_header('Model Metrics', 'Inspect dataset quality, evaluation scores, confusion matrix, and training metadata on a dedicated metrics screen.', 'Model')
    try:
        metrics = get_metrics()
        if not metrics:
            raise ModelNotFoundError('No metrics file was found. Please train the model first.')
        top = st.columns(5)
        top[0].metric('Accuracy', f"{metrics.get('accuracy', 0):.2%}")
        top[1].metric('Precision', f"{metrics.get('precision', 0):.2%}")
        top[2].metric('Recall', f"{metrics.get('recall', 0):.2%}")
        top[3].metric('F1 Score', f"{metrics.get('f1_score', 0):.2%}")
        top[4].metric('ROC-AUC', f"{metrics.get('roc_auc', 0):.2%}")

        data_left, data_right = st.columns([1, 1])
        with data_left:
            st.subheader('Confusion Matrix')
            matrix = metrics.get('confusion_matrix', [[0, 0], [0, 0]])
            st.dataframe(
                pd.DataFrame(matrix, index=['Actual Legitimate', 'Actual Phishing'], columns=['Pred Legitimate', 'Pred Phishing']),
                use_container_width=True,
            )
            st.subheader('Dataset Preview')
            dataset_df = load_dataset_preview()
            st.dataframe(dataset_df.head(10), use_container_width=True)
        with data_right:
            st.subheader('Classification Report')
            report = metrics.get('class_report', {})
            if report:
                st.dataframe(pd.DataFrame(report).T, use_container_width=True)
            st.subheader('Training Metadata')
            st.write(f"Training size: {metrics.get('train_size', 'N/A')}")
            st.write(f"Test size: {metrics.get('test_size', 'N/A')}")
            if st.button('Retrain Model Now', use_container_width=True):
                with st.spinner('Retraining model...'):
                    _, summary = train_and_save_model()
                    db.record_training_run(summary, actor_user_id=user['id'])
                    db.log_event(
                        actor_user_id=user['id'],
                        action='train_model',
                        target_type='model',
                        description=f'Model retrained from the metrics page with accuracy {summary.accuracy:.2%}.',
                        severity='warning',
                    )
                    refresh_caches()
                st.success('Model retrained successfully.')
                st.rerun()

        runs = db.get_training_runs(limit=10)
        if runs:
            st.subheader('Recent Training Runs')
            st.dataframe(pd.DataFrame(runs), use_container_width=True)

        st.subheader('Raw Metrics JSON')
        st.code(json.dumps(metrics, indent=2), language='json')
    except ModelNotFoundError as exc:
        st.error(str(exc))


def render_platform_page() -> None:
    render_page_header('Platform Architecture', 'A cleaner high-level description of how the implemented system is structured across interface, model, security, and database layers.', 'System')
    cols = st.columns(2)
    cols[0].markdown(
        """
        <div class='panel-card'>
            <h3>Application stack</h3>
            <ul>
                <li><strong>Frontend:</strong> Streamlit drives the landing page, authentication pages, dashboards, and all task-specific screens.</li>
                <li><strong>Machine learning:</strong> scikit-learn is used with TF IDF feature extraction and Logistic Regression classification.</li>
                <li><strong>Database:</strong> SQLite stores users, scan history, audit logs, and training runs in a portable relational file.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
    cols[1].markdown(
        """
        <div class='panel-card'>
            <h3>Operational capabilities</h3>
            <ul>
                <li>Secure registration, login, and role-based access control.</li>
                <li>Single-message analysis and batch processing workflows.</li>
                <li>Saved history, analytics, report export, and user feedback notes.</li>
                <li>Administrative monitoring, user management, and audit visibility.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_account_page(user: dict) -> None:
    st.markdown("<div class=\"page-container\">", unsafe_allow_html=True)
    st.markdown("<div class=\"page-header\"><h1 class=\"page-title\">Account Settings</h1><p class=\"page-subtitle\">Manage your profile, security settings, and account preferences.</p></div>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("<div class=\"card\">", unsafe_allow_html=True)
        st.markdown("<h3 class=\"card-title\">👤 Profile Information</h3>", unsafe_allow_html=True)

        with st.form('profile_form'):
            st.markdown("<div class=\"form-group\"><label class=\"form-label\">Full Name</label>", unsafe_allow_html=True)
            full_name = st.text_input('', value=user.get('full_name', ''), label_visibility='collapsed')
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div class=\"form-group\"><label class=\"form-label\">Institution</label>", unsafe_allow_html=True)
            institution = st.text_input('', value=user.get('institution', ''), label_visibility='collapsed')
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div class=\"form-group\"><label class=\"form-label\">Bio</label>", unsafe_allow_html=True)
            bio = st.text_area('', value=user.get('bio', ''), height=100, label_visibility='collapsed')
            st.markdown("</div>", unsafe_allow_html=True)

            submitted = st.form_submit_button('💾 Save Profile', use_container_width=True)
            if submitted:
                if len(full_name.strip()) < 3:
                    st.error('❌ Full name must be at least 3 characters long.')
                else:
                    db.update_user_profile(user['id'], full_name, institution, bio)
                    db.log_event(
                        actor_user_id=user['id'],
                        action='update_profile',
                        target_type='user',
                        target_id=user['id'],
                        description='Profile information was updated.',
                        severity='info',
                    )
                    st.success('✅ Profile updated successfully.')
                    st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class=\"card\">", unsafe_allow_html=True)
        st.markdown("<h3 class=\"card-title\">🔐 Change Password</h3>", unsafe_allow_html=True)

        with st.form('password_form'):
            st.markdown("<div class=\"form-group\"><label class=\"form-label\">Current Password</label>", unsafe_allow_html=True)
            current_password = st.text_input('', type='password', label_visibility='collapsed')
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div class=\"form-group\"><label class=\"form-label\">New Password</label>", unsafe_allow_html=True)
            new_password = st.text_input('', type='password', label_visibility='collapsed')
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div class=\"form-group\"><label class=\"form-label\">Confirm New Password</label>", unsafe_allow_html=True)
            confirm_password = st.text_input('', type='password', label_visibility='collapsed')
            st.markdown("</div>", unsafe_allow_html=True)

            submitted = st.form_submit_button('🔄 Update Password', use_container_width=True)
            if submitted:
                password_ok, password_message = assess_password_strength(new_password)
                if not password_ok:
                    st.error(f'❌ {password_message}')
                elif new_password != confirm_password:
                    st.error('❌ New passwords do not match.')
                elif not db.change_password(user['id'], current_password, new_password):
                    st.error('❌ Current password is incorrect.')
                else:
                    db.log_event(
                        actor_user_id=user['id'],
                        action='change_password',
                        target_type='user',
                        target_id=user['id'],
                        description='User password was changed successfully.',
                        severity='warning',
                    )
                    st.success('✅ Password updated successfully.')
        st.markdown("</div>", unsafe_allow_html=True)

    # Account Summary
    st.markdown("<div class=\"card\">", unsafe_allow_html=True)
    st.markdown("<h3 class=\"card-title\">📊 Account Summary</h3>", unsafe_allow_html=True)

    stats_data = db.get_user_statistics(user['id'])
    summary_cols = st.columns(4)
    summary_cols[0].metric('Total Scans', stats_data['total_scans'])
    summary_cols[1].metric('Phishing Found', stats_data['phishing_count'])
    summary_cols[2].metric('High Risk Scans', stats_data['high_risk_count'])
    summary_cols[3].metric('Account Age', f"{(pd.Timestamp.now() - pd.Timestamp(user.get('created_at', pd.Timestamp.now()))).days} days")

    st.markdown(f"""
    <div style="background: #f8fafc; padding: 1.5rem; border-radius: 8px; margin-top: 1rem;">
        <h4 style="margin: 0 0 1rem 0; color: #0f172a;">Account Details</h4>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
            <div><strong>Email:</strong> {user['email']}</div>
            <div><strong>Role:</strong> {user.get('role', 'user').title()}</div>
            <div><strong>Full Name:</strong> {user.get('full_name', 'Not set')}</div>
            <div><strong>Institution:</strong> {user.get('institution', 'Not set')}</div>
            <div><strong>Created:</strong> {user.get('created_at', 'Unknown')}</div>
            <div><strong>Last Login:</strong> {user.get('last_login_at', 'Not available')}</div>
        </div>
        {f"<div style=\"margin-top: 1rem;\"><strong>Bio:</strong> {user.get('bio', 'Not set')}</div>" if user.get('bio') else ""}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("</div></div>", unsafe_allow_html=True)


def render_admin_center(user: dict) -> None:
    if not require_admin(user):
        return

    render_page_header('Admin Control Center', 'Manage users, review audit activity, and monitor platform-wide operations from a dedicated administration area.', 'Administration')

    tab_overview, tab_users, tab_activity = st.tabs(['Overview', 'User Management', 'Audit and Operations'])

    with tab_overview:
        platform = db.get_platform_statistics()
        metrics = get_metrics()
        cards = st.columns(6)
        cards[0].metric('Total Users', platform['total_users'])
        cards[1].metric('Active Users', platform['active_users'])
        cards[2].metric('Administrators', platform['admin_users'])
        cards[3].metric('Saved Scans', platform['total_scans'])
        cards[4].metric('High Risk Scans', platform['high_risk_scans'])
        cards[5].metric('DB Size', f"{platform['db_size_kb']} KB")

        left, right = st.columns([1.15, 1])
        with left:
            global_history = db.get_global_history(limit=500)
            if global_history:
                gdf = pd.DataFrame(global_history)
                trend = (
                    gdf.assign(created_date=pd.to_datetime(gdf['created_at']).dt.date)
                    .groupby(['created_date', 'risk_level'])
                    .size()
                    .unstack(fill_value=0)
                )
                st.subheader('Global Risk Timeline')
                st.line_chart(trend)
                st.subheader('Recent High-Risk Queue')
                high_df = gdf[gdf['risk_level'] == 'High'][['created_at', 'full_name', 'email', 'predicted_name', 'confidence']].head(15)
                if not high_df.empty:
                    st.dataframe(high_df, use_container_width=True)
                else:
                    st.info('No high-risk records are currently saved.')
            else:
                st.info('Global scan analytics will appear after users save results.')
        with right:
            st.subheader('Model Snapshot')
            if metrics:
                st.metric('Model Accuracy', f"{metrics.get('accuracy', 0):.2%}")
                st.metric('F1 Score', f"{metrics.get('f1_score', 0):.2%}")
                st.metric('ROC AUC', f"{metrics.get('roc_auc', 0):.2%}")
            runs_df = pd.DataFrame(db.get_training_runs(limit=10))
            if not runs_df.empty:
                st.subheader('Training Run History')
                st.dataframe(runs_df, use_container_width=True)

    with tab_users:
        controls = st.columns([2, 1, 1, 1])
        search = controls[0].text_input('Search users', placeholder='name email institution')
        role_filter = controls[1].selectbox('Role', ['All', 'admin', 'user'])
        active_filter = controls[2].selectbox('Status', ['All', 'Active', 'Inactive'])
        row_limit = controls[3].selectbox('Rows', [25, 50, 100, 250], index=1)

        users = db.get_all_users(limit=row_limit, search=search, role=role_filter, active=active_filter)
        users_df = pd.DataFrame(users)
        if not users_df.empty:
            st.dataframe(users_df[['id', 'full_name', 'email', 'role', 'institution', 'is_active', 'created_at', 'last_login_at']], use_container_width=True)
            dataframe_download_button('Download user registry as CSV', users_df, 'user_registry.csv')
        else:
            st.info('No users matched the current filters.')

        st.markdown('### Create User Account')
        with st.form('admin_create_user_form'):
            new_cols = st.columns(3)
            full_name = new_cols[0].text_input('Full Name')
            email = new_cols[1].text_input('Email')
            role = new_cols[2].selectbox('Role to assign', ['user', 'admin'])
            institution = st.text_input('Institution')
            bio = st.text_area('Bio', height=90)
            password = st.text_input('Temporary Password', type='password')
            confirm = st.text_input('Confirm Temporary Password', type='password')
            submit = st.form_submit_button('Create Account', use_container_width=True)
            if submit:
                ok, message, created_user = register_user(
                    db,
                    full_name,
                    email,
                    password,
                    confirm,
                    institution,
                    bio,
                    role=role,
                    actor_user_id=user['id'],
                )
                if ok and created_user:
                    st.success('User account created successfully.')
                    db.log_event(
                        actor_user_id=user['id'],
                        action='admin_create_user',
                        target_type='user',
                        target_id=created_user['id'],
                        description=f"Administrator created {role} account for {created_user['email']}.",
                        severity='warning',
                    )
                    st.rerun()
                st.error(message)

        if users:
            st.markdown('### Manage Existing User')
            selected_label = st.selectbox(
                'Select user',
                [f"#{u['id']} | {u['full_name']} | {u['email']} | {u['role']}" for u in users],
                key='admin_user_select',
            )
            selected_id = int(selected_label.split('|')[0].replace('#', '').strip())
            target_user = next(u for u in users if u['id'] == selected_id)
            manage_cols = st.columns(3)
            if manage_cols[0].button('Toggle Active Status', use_container_width=True):
                platform = db.get_platform_statistics()
                if target_user['role'] == 'admin' and int(target_user['is_active']) == 1 and platform['admin_users'] <= 1:
                    st.error('The last active administrator cannot be deactivated.')
                else:
                    new_status = not bool(target_user['is_active'])
                    db.set_user_active(target_user['id'], new_status)
                    db.log_event(
                        actor_user_id=user['id'],
                        action='set_user_active',
                        target_type='user',
                        target_id=target_user['id'],
                        description=f"Administrator changed active status for {target_user['email']} to {new_status}.",
                        severity='warning',
                    )
                    st.success('User status updated.')
                    st.rerun()
            if manage_cols[1].button('Toggle Admin Role', use_container_width=True):
                platform = db.get_platform_statistics()
                new_role = 'user' if target_user['role'] == 'admin' else 'admin'
                if target_user['role'] == 'admin' and platform['admin_users'] <= 1:
                    st.error('The last administrator cannot be demoted.')
                else:
                    db.set_user_role(target_user['id'], new_role)
                    db.log_event(
                        actor_user_id=user['id'],
                        action='set_user_role',
                        target_type='user',
                        target_id=target_user['id'],
                        description=f"Administrator changed role for {target_user['email']} to {new_role}.",
                        severity='warning',
                    )
                    st.success('User role updated.')
                    st.rerun()
            with manage_cols[2]:
                new_password = st.text_input('New password for selected user', type='password', key='admin_reset_password_input')
                if st.button('Reset Password', use_container_width=True):
                    password_ok, password_message = assess_password_strength(new_password)
                    if not password_ok:
                        st.error(password_message)
                    elif db.admin_reset_password(target_user['id'], new_password):
                        db.log_event(
                            actor_user_id=user['id'],
                            action='reset_password',
                            target_type='user',
                            target_id=target_user['id'],
                            description=f"Administrator reset the password for {target_user['email']}.",
                            severity='critical',
                        )
                        st.success('User password reset successfully.')
                        st.rerun()

    with tab_activity:
        controls = st.columns([1, 2, 1])
        severity = controls[0].selectbox('Severity', ['All', 'info', 'warning', 'critical'])
        search_term = controls[1].text_input('Search audit logs', placeholder='action description or actor email')
        limit = controls[2].slider('Rows', 25, MAX_AUDIT_EXPORT_ROWS, 100, step=25)
        logs = db.get_audit_logs(limit=limit, severity=severity, search=search_term)
        if logs:
            logs_df = pd.DataFrame(logs)
            st.dataframe(logs_df, use_container_width=True)
            dataframe_download_button('Download audit logs as CSV', logs_df, 'audit_logs.csv')
        else:
            st.info('No audit log entries matched the current filters.')

        global_history = db.get_global_history(limit=limit, risk_level='All')
        if global_history:
            st.markdown('### Global Scan Export')
            history_df = pd.DataFrame(global_history)
            st.dataframe(history_df[['created_at', 'full_name', 'email', 'predicted_name', 'risk_level', 'confidence', 'source_type']], use_container_width=True)
            dataframe_download_button('Download global scan history as CSV', history_df, 'global_scan_history.csv')



if not st.session_state.user_id:
    current_public = st.session_state.public_page
    if current_public == 'Login':
        render_login_page()
    elif current_public == 'Register':
        render_register_page()
    else:
        render_landing()
else:
    user = db.get_user_by_id(st.session_state.user_id)
    if not user:
        set_user(None)
        goto_public('Landing')
        st.rerun()
    else:
        render_app_topbar(user)
        selected_page = st.session_state.app_page
        if selected_page == 'Dashboard':
            render_dashboard(user)
        elif selected_page == 'Scan Message':
            render_scan_page(user)
        elif selected_page == 'Batch Analysis':
            render_batch_page(user)
        elif selected_page == 'Scan History':
            render_history_page(user)
        elif selected_page == 'Analytics':
            render_analytics_page(user)
        elif selected_page == 'Model Metrics':
            render_metrics_page(user)
        elif selected_page == 'Platform Architecture':
            render_platform_page()
        elif selected_page == 'Admin Control Center':
            render_admin_center(user)
        else:
            render_account_page(user)
