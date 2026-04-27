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
    left, right = st.columns([1.7, 1])
    with left:
        st.markdown(
            f"""
            <div class='topbar-brand'>
                <div class='topbar-title'>{APP_NAME}</div>
                <div class='topbar-text'>Enterprise styled phishing detection platform</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        nav_cols = st.columns(3)
        for idx, page in enumerate(PUBLIC_PAGES):
            with nav_cols[idx]:
                nav_button(page, page, st.session_state.public_page == page, f'public_{page}', public=True)


def render_app_topbar(user: dict) -> None:
    render_shell_header()
    all_pages = APP_PAGES.copy()
    if user.get('role') == 'admin':
        all_pages.insert(-1, ADMIN_PAGE)

    st.markdown("<div class='nav-wrap'>", unsafe_allow_html=True)
    nav_cols = st.columns(len(all_pages))
    for idx, page in enumerate(all_pages):
        short_label = page.replace('Control Center', 'Admin')
        with nav_cols[idx]:
            nav_button(short_label, page, st.session_state.app_page == page, f'app_{page}')

    info_left, info_right = st.columns([1.6, 1])
    with info_left:
        st.markdown(
            f"""
            <div class='user-strip'>
                <span class='role-pill'>{user.get('role', 'user').title()}</span>
                <span><strong>{user['full_name']}</strong></span>
                <span>{user['email']}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with info_right:
        action_cols = st.columns(2)
        with action_cols[0]:
            if st.button('Open Account', use_container_width=True, key='open_account_btn'):
                goto_app('Account')
                st.rerun()
        with action_cols[1]:
            if st.button('Sign Out', use_container_width=True, key='sign_out_top_btn'):
                logout()


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

    render_public_topbar()

    hero_left, hero_right = st.columns([1.25, 1])
    with hero_left:
        st.markdown("<div class='hero-chip'>Final professional academic version</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='landing-title'>{APP_NAME}</div>", unsafe_allow_html=True)
        st.markdown(f"<p class='landing-copy'>{APP_TAGLINE}</p>", unsafe_allow_html=True)
        st.write(
            'This system is a polished phishing detection platform built with Python, machine learning, authentication, saved history, analytics, and administrative oversight. '
            'It is designed to present the project as a complete high-level tool rather than a basic demo.'
        )
        ctas = st.columns(3)
        with ctas[0]:
            if st.button('Login to Platform', use_container_width=True, type='primary'):
                goto_public('Login')
                st.rerun()
        with ctas[1]:
            if st.button('Create Account', use_container_width=True):
                goto_public('Register')
                st.rerun()
        with ctas[2]:
            if st.button('View Architecture', use_container_width=True):
                st.session_state.public_page = 'Landing'
        stats = st.columns(4)
        stats[0].metric('Database', 'SQLite')
        stats[1].metric('Authentication', 'Enabled')
        stats[2].metric('Admin Console', 'Included')
        stats[3].metric('Model Accuracy', f"{metrics.get('accuracy', 0):.2%}" if metrics else 'N/A')
    with hero_right:
        st.markdown(
            """
            <div class='panel-card hero-card'>
                <h3>System highlights</h3>
                <ul>
                    <li>Dedicated landing login and registration screens</li>
                    <li>Individual scan page and batch analysis page</li>
                    <li>Personal history export and model explainability</li>
                    <li>Admin control center and audit trail</li>
                    <li>Cleaner light UI with product-style navigation</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    cards = st.columns(4)
    cards[0].metric('Users', platform_stats['total_users'])
    cards[1].metric('Saved Scans', platform_stats['total_scans'])
    cards[2].metric('High Risk Scans', platform_stats['high_risk_scans'])
    cards[3].metric('Training Runs', platform_stats['training_runs'])

    features = st.columns(3)
    features[0].markdown(
        "<div class='panel-card feature-card'><h4>Detection Engine</h4><p>The application uses TF IDF and Logistic Regression to classify messages as phishing or legitimate with probability outputs and explanation signals.</p></div>",
        unsafe_allow_html=True,
    )
    features[1].markdown(
        "<div class='panel-card feature-card'><h4>Persistence Layer</h4><p>SQLite stores users, scan history, audit activity, and training records inside a portable relational database created automatically by the system.</p></div>",
        unsafe_allow_html=True,
    )
    features[2].markdown(
        "<div class='panel-card feature-card'><h4>Administrative Oversight</h4><p>Administrators can review activity, manage users, retrain the model, and monitor operational events from a dedicated control center.</p></div>",
        unsafe_allow_html=True,
    )


def render_login_page() -> None:
    render_public_topbar()
    render_page_header('Login', 'Access the platform through a dedicated sign-in screen with a cleaner professional layout.', 'Public Access')
    left, right = st.columns([1.1, 0.9])
    with left:
        with st.form('login_form', clear_on_submit=False):
            st.subheader('Sign in to your account')
            email = st.text_input('Email', placeholder='student@example.com')
            password = st.text_input('Password', type='password')
            submitted = st.form_submit_button('Sign In', type='primary', use_container_width=True)
            if submitted:
                ok, message, user = login_user(db, email, password)
                if ok and user:
                    set_user(user)
                    goto_app('Dashboard')
                    st.success(message)
                    st.rerun()
                st.error(message)
    with right:
        st.markdown(
            f"""
            <div class='panel-card side-note'>
                <h3>Default administrator</h3>
                <p>Use the initial administrator account for first-time access, then change the password from the account page.</p>
                <p><strong>Email:</strong> {DEFAULT_ADMIN_EMAIL}</p>
                <p><strong>Password:</strong> {DEFAULT_ADMIN_PASSWORD}</p>
                <hr />
                <p>This screen is now independent from registration to provide a clearer user flow and a cleaner presentation during your demo.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button('Need a new account?', use_container_width=True, key='go_register_from_login'):
            goto_public('Register')
            st.rerun()


def render_register_page() -> None:
    render_public_topbar()
    render_page_header('Create Account', 'Register through a separate dedicated page with profile details and password validation.', 'Public Access')
    left, right = st.columns([1.1, 0.9])
    with left:
        with st.form('register_form', clear_on_submit=True):
            st.subheader('Create a new user account')
            full_name = st.text_input('Full Name', placeholder='Your full name')
            institution = st.text_input('Institution', placeholder='University or department')
            email = st.text_input('Email Address', placeholder='student@example.com')
            bio = st.text_area('Short Bio', placeholder='Optional short role or project description', height=90)
            password = st.text_input('Create Password', type='password')
            confirm_password = st.text_input('Confirm Password', type='password')
            submitted = st.form_submit_button('Create Account', type='primary', use_container_width=True)
            if submitted:
                ok, message, user = register_user(db, full_name, email, password, confirm_password, institution, bio)
                if ok and user:
                    set_user(user)
                    goto_app('Dashboard')
                    st.success(message)
                    st.rerun()
                st.error(message)
    with right:
        st.markdown(
            """
            <div class='panel-card side-note'>
                <h3>Password guidance</h3>
                <p>Use a strong password that includes uppercase letters, lowercase letters, numbers, and a symbol. The system stores credentials using salted password hashing.</p>
                <hr />
                <h3>What you get after registration</h3>
                <p>Each account has access to message analysis, saved history, personal analytics, report exports, and profile management.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button('Already have an account?', use_container_width=True, key='go_login_from_register'):
            goto_public('Login')
            st.rerun()


def render_dashboard(user: dict) -> None:
    metrics = get_metrics()
    history = db.get_history(user['id'], limit=400)
    stats_data = db.get_user_statistics(user['id'])

    render_page_header('Dashboard', 'Review your recent activity, model quality, and current system state from a cleaner overview page.', 'Workspace')

    stats = st.columns(6)
    stats[0].metric('Total Scans', stats_data['total_scans'])
    stats[1].metric('Phishing Results', stats_data['phishing_count'])
    stats[2].metric('Legitimate Results', stats_data['legitimate_count'])
    stats[3].metric('High Risk Results', stats_data['high_risk_count'])
    stats[4].metric('Average Confidence', f"{stats_data['avg_confidence']:.2%}" if stats_data['total_scans'] else 'N/A')
    stats[5].metric('Model Accuracy', f"{metrics.get('accuracy', 0):.2%}" if metrics else 'N/A')

    left, right = st.columns([1.25, 0.95])
    with left:
        st.subheader('Recent Activity Trend')
        if history:
            hist_df = pd.DataFrame(history)
            chart_df = (
                hist_df.assign(created_date=pd.to_datetime(hist_df['created_at']).dt.date)
                .groupby(['created_date', 'predicted_name'])
                .size()
                .unstack(fill_value=0)
            )
            st.line_chart(chart_df)
            st.subheader('Latest Saved Results')
            st.dataframe(
                hist_df[['created_at', 'predicted_name', 'risk_level', 'confidence', 'source_type']].head(12),
                use_container_width=True,
            )
        else:
            st.info('No saved scans yet. Use the Scan Message page or Batch Analysis page to create your first results.')
    with right:
        st.markdown(
            """
            <div class='panel-card info-card'>
                <h4>Current stack</h4>
                <p>Streamlit interface plus TF IDF and Logistic Regression for classification with SQLite as the persistent relational store.</p>
            </div>
            <div class='panel-card info-card'>
                <h4>Capabilities</h4>
                <p>Single message analysis, batch processing, explainable outputs, personal history export, analytics, and administrative monitoring are all included in this build.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button('Train or Refresh Model', use_container_width=True):
            with st.spinner('Training model and refreshing metrics...'):
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
                f"Training complete. Accuracy: {summary.accuracy:.2%} | F1-score: {summary.f1_score:.2%} | ROC-AUC: {summary.roc_auc:.2%}"
            )
            st.rerun()


def render_scan_page(user: dict) -> None:
    render_page_header('Scan Message', 'Analyze one message at a time with a cleaner single-purpose page and clearer result presentation.', 'Detection')

    chosen_sample = st.selectbox('Load a sample message', ['Custom'] + list(SAMPLE_MESSAGES.keys()))
    default_text = SAMPLE_MESSAGES.get(chosen_sample, '') if chosen_sample != 'Custom' else ''
    user_input = st.text_area(
        'Message or email text',
        height=220,
        placeholder='Paste message content here...',
        value=default_text,
    )
    uploaded = st.file_uploader('Optional text file upload', type=['txt'], key='single_upload')
    if uploaded is not None:
        try:
            user_input = uploaded.read().decode('utf-8', errors='ignore')
            st.text_area('Uploaded content preview', value=user_input, height=160)
        except Exception:
            st.error('Could not read the uploaded file as UTF-8 text.')

    action_cols = st.columns([1, 1, 3])
    analyze = action_cols[0].button('Analyze', type='primary', use_container_width=True)
    save_toggle = action_cols[1].checkbox('Save to history', value=True)
    action_cols[2].caption('Reports can be downloaded after a successful scan.')

    if analyze:
        try:
            result = predict_text(user_input)
            st.session_state.last_result = {'result': result, 'original_text': user_input}
            if save_toggle:
                scan_id = db.save_scan(user['id'], result, user_input, source_type='manual', source_name='message box')
                db.log_event(
                    actor_user_id=user['id'],
                    action='scan_message',
                    target_type='scan',
                    target_id=scan_id,
                    description=f"User analyzed a message and received {result['predicted_name']} with {result['risk_level']} risk.",
                    severity='warning' if result['risk_level'] in {'High', 'Medium'} else 'info',
                )
                st.success('Result saved to your history.')
            render_scan_result(result, user_input)
        except ModelNotFoundError as exc:
            st.error(str(exc))
        except ValueError as exc:
            st.error(str(exc))

    elif st.session_state.last_result:
        st.markdown('### Last Scan Result')
        render_scan_result(st.session_state.last_result['result'], st.session_state.last_result['original_text'])


def render_batch_page(user: dict) -> None:
    render_page_header('Batch Analysis', 'Upload CSV or TXT content and process multiple messages in one run from a dedicated batch workflow.', 'Detection')

    uploaded = st.file_uploader('Upload batch file', type=['csv', 'txt'], key='batch_upload')
    save_results = st.checkbox('Save all batch results to history', value=False)
    rows_to_process = st.slider('Maximum rows to process', 5, 250, 50, step=5)

    texts: List[str] = []
    source_name = ''

    if uploaded is not None:
        source_name = uploaded.name
        if uploaded.name.lower().endswith('.csv'):
            try:
                df = pd.read_csv(uploaded)
                st.dataframe(df.head(10), use_container_width=True)
                column_choice = st.selectbox('Select the text column', list(df.columns))
                if column_choice:
                    texts = [str(value) for value in df[column_choice].dropna().astype(str).tolist()[:rows_to_process]]
            except Exception as exc:
                st.error(f'Could not read the CSV file: {exc}')
        else:
            try:
                text_content = uploaded.read().decode('utf-8', errors='ignore')
                texts = file_to_lines(text_content)[:rows_to_process]
                st.text_area('TXT preview', value='\n'.join(texts[:20]), height=180)
            except Exception as exc:
                st.error(f'Could not read the TXT file: {exc}')

    if st.button('Run Batch Analysis', type='primary', disabled=not texts, use_container_width=True):
        try:
            results = predict_many(texts)
            if not results:
                st.warning('No valid text rows were found to analyze.')
                return
            records = []
            for original_text, result in zip(texts, results):
                records.append(
                    {
                        'text': original_text,
                        'prediction': result['predicted_name'],
                        'confidence': round(result['confidence'], 4),
                        'phishing_probability': round(result['phishing_probability'], 4),
                        'risk_level': result['risk_level'],
                        'risk_score': result['heuristic_risk_score'],
                        'signals': '; '.join(result['signal_details']),
                        'keywords': ', '.join(result['suspicious_keywords']),
                    }
                )
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
            st.success(f'Batch analysis completed for {len(results_df)} messages.')
            st.dataframe(results_df, use_container_width=True)
            dataframe_download_button('Download batch results as CSV', results_df, 'batch_analysis_results.csv')
        except ModelNotFoundError as exc:
            st.error(str(exc))

    if isinstance(st.session_state.batch_results_df, pd.DataFrame):
        st.markdown('### Last Batch Run')
        st.dataframe(st.session_state.batch_results_df, use_container_width=True)


def render_history_page(user: dict) -> None:
    render_page_header('Scan History', 'Search, filter, export, and manage previously saved records from a dedicated history workspace.', 'Workspace')

    filters = st.columns([1, 1, 1, 1, 2])
    risk_filter = filters[0].selectbox('Risk Level', ['All', 'High', 'Medium', 'Low'])
    class_filter = filters[1].selectbox('Class', ['All', 'Phishing', 'Legitimate'])
    max_rows = filters[2].selectbox('Rows', [25, 50, 100, 250, 500], index=2)
    export_limit = filters[3].slider('CSV Rows', 25, MAX_HISTORY_EXPORT_ROWS, min(500, MAX_HISTORY_EXPORT_ROWS), step=25)
    search_term = filters[4].text_input('Search', placeholder='keyword recommendation or class')

    history = db.get_history(user['id'], limit=max_rows, risk_level=risk_filter, search=search_term, predicted_name=class_filter)
    if not history:
        st.info('No saved scans matched the selected filters.')
        return

    history_df = pd.DataFrame(history)
    display_columns = [
        'created_at', 'predicted_name', 'risk_level', 'confidence', 'heuristic_risk_score',
        'source_type', 'source_name', 'suspicious_keywords', 'recommendation', 'user_feedback'
    ]
    st.dataframe(history_df[display_columns], use_container_width=True)

    export_df = pd.DataFrame(
        db.get_history(user['id'], limit=export_limit, risk_level=risk_filter, search=search_term, predicted_name=class_filter)
    )
    dataframe_download_button('Download filtered history as CSV', export_df, 'phishguard_history.csv')

    st.subheader('Manage individual history items')
    options = {f"#{row['id']} | {row['created_at']} | {row['predicted_name']} | {row['risk_level']}": row['id'] for row in history}
    selected_label = st.selectbox('Select a saved scan', list(options.keys()))
    selected_id = options[selected_label]
    selected_row = next(row for row in history if row['id'] == selected_id)
    st.text_area('Saved message', value=selected_row['message_text'], height=180)
    feedback = st.text_area('Feedback note', value=selected_row.get('user_feedback', ''), key=f'feedback_{selected_id}')
    action_cols = st.columns(3)
    if action_cols[0].button('Save Feedback', key=f'save_{selected_id}', use_container_width=True):
        db.update_feedback(selected_id, user['id'], feedback)
        db.log_event(
            actor_user_id=user['id'],
            action='update_feedback',
            target_type='scan',
            target_id=selected_id,
            description=f'Feedback updated for saved scan #{selected_id}.',
            severity='info',
        )
        st.success('Feedback updated.')
        st.rerun()
    if action_cols[1].button('Delete Selected', key=f'delete_{selected_id}', use_container_width=True):
        db.delete_history_item(selected_id, user['id'])
        db.log_event(
            actor_user_id=user['id'],
            action='delete_scan',
            target_type='scan',
            target_id=selected_id,
            description=f'Saved scan #{selected_id} was deleted.',
            severity='warning',
        )
        st.success('Selected history item deleted.')
        st.rerun()
    if action_cols[2].button('Clear All History', key='clear_all_history', use_container_width=True):
        db.clear_history(user['id'])
        db.log_event(
            actor_user_id=user['id'],
            action='clear_history',
            target_type='scan',
            description='User cleared all personal scan history.',
            severity='warning',
        )
        st.success('All history entries were removed.')
        st.rerun()


def render_analytics_page(user: dict) -> None:
    render_page_header('Analytics', 'Review your saved scanning trends, confidence levels, and keyword patterns from a cleaner analytics page.', 'Workspace')
    history = db.get_history(user['id'], limit=1000)
    if not history:
        st.info('Analytics will appear after you save some scan results.')
        return

    df = pd.DataFrame(history)
    top = st.columns(4)
    top[0].metric('Rows Analysed', len(df))
    top[1].metric('High Risk Rate', f"{(df['risk_level'].eq('High').mean() if len(df) else 0):.2%}")
    top[2].metric('Phishing Rate', f"{(df['predicted_name'].eq('Phishing').mean() if len(df) else 0):.2%}")
    top[3].metric('Average Confidence', f"{df['confidence'].mean():.2%}")

    left, right = st.columns(2)
    with left:
        st.subheader('Risk Level Distribution')
        st.bar_chart(df['risk_level'].value_counts())
        st.subheader('Average Confidence by Class')
        st.bar_chart(df.groupby('predicted_name')['confidence'].mean())
    with right:
        st.subheader('Average Risk Score by Class')
        st.bar_chart(df.groupby('predicted_name')['heuristic_risk_score'].mean())
        st.subheader('Daily Timeline')
        timeline = (
            df.assign(created_date=pd.to_datetime(df['created_at']).dt.date)
            .groupby(['created_date', 'risk_level'])
            .size()
            .unstack(fill_value=0)
        )
        st.line_chart(timeline)

    keyword_series = (
        df['suspicious_keywords']
        .fillna('')
        .str.split(', ')
        .explode()
        .str.strip()
    )
    keyword_series = keyword_series[keyword_series.astype(bool)]
    if not keyword_series.empty:
        st.subheader('Most Frequent Suspicious Keywords')
        st.bar_chart(keyword_series.value_counts().head(10))


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
    render_page_header('Account', 'Manage your profile, password, and account snapshot from a dedicated account page.', 'Workspace')
    left, right = st.columns([1, 1])

    with left:
        st.subheader('Profile')
        with st.form('profile_form'):
            full_name = st.text_input('Full Name', value=user.get('full_name', ''))
            institution = st.text_input('Institution', value=user.get('institution', ''))
            bio = st.text_area('Bio', value=user.get('bio', ''), height=120)
            submitted = st.form_submit_button('Save Profile', use_container_width=True)
            if submitted:
                if len(full_name.strip()) < 3:
                    st.error('Full name must be at least 3 characters long.')
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
                    st.success('Profile updated successfully.')
                    st.rerun()

    with right:
        st.subheader('Change Password')
        with st.form('password_form'):
            current_password = st.text_input('Current Password', type='password')
            new_password = st.text_input('New Password', type='password')
            confirm_password = st.text_input('Confirm New Password', type='password')
            submitted = st.form_submit_button('Update Password', use_container_width=True)
            if submitted:
                password_ok, password_message = assess_password_strength(new_password)
                if not password_ok:
                    st.error(password_message)
                elif new_password != confirm_password:
                    st.error('New passwords do not match.')
                elif not db.change_password(user['id'], current_password, new_password):
                    st.error('Current password is incorrect.')
                else:
                    db.log_event(
                        actor_user_id=user['id'],
                        action='change_password',
                        target_type='user',
                        target_id=user['id'],
                        description='User password was changed successfully.',
                        severity='warning',
                    )
                    st.success('Password updated successfully.')

    st.markdown(
        f"""
        <div class='panel-card account-snapshot'>
            <h3>{user['full_name']}</h3>
            <p><strong>Email:</strong> {user['email']}</p>
            <p><strong>Institution:</strong> {user.get('institution', '') or 'Not set'}</p>
            <p><strong>Role:</strong> {user.get('role', 'user')}</p>
            <p><strong>Created:</strong> {user.get('created_at', 'Unknown')}</p>
            <p><strong>Last login:</strong> {user.get('last_login_at', 'Not available')}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


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
