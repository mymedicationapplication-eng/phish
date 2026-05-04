# PhishGuard AI Enterprise

PhishGuard AI Enterprise is a polished academic phishing detection platform built with Python, Streamlit, scikit-learn, and SQLite. It supports registration, login, role-based access control, manual and batch message scanning, explainable machine learning outputs, saved personal history, audit logging, and a full administrative control center.

## Why SQLite was chosen

SQLite is the most suitable database for this version of the project because it is lightweight, file-based, portable, and easy to deploy on a single machine. It does not require a separate database server, which makes it ideal for classroom demonstrations, academic submissions, and local prototyping while still supporting structured relational data.

## Main Features

- Professional landing page and dashboard
- User registration and secure login
- Role-based access control with administrator capabilities
- SQLite database with automatic schema initialization
- Manual single-message phishing analysis
- Batch analysis for CSV and TXT uploads
- Personal phishing scan history with search, filtering, export, and notes
- Explainable AI output with confidence, risk level, rule signals, URLs, and influential model terms
- Model metrics page with dataset preview, confusion matrix, and classification report
- Administrative control center with global analytics, user management, password reset, role changes, audit logs, and training run history
- Local training and retraining workflows

## Tech Stack

- **Frontend:** Streamlit
- **Machine Learning:** scikit-learn
- **Feature Extraction:** TF-IDF
- **Classifier:** Logistic Regression
- **Database:** SQLite
- **Language:** Python

## Default Administrator Account

The app automatically creates a default administrator account the first time the database is initialized.

- **Email:** `admin@phishguard.local`
- **Password:** `Admin@12345`

After the first login, change the password from the **Account** page.

## Run Locally

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

If the browser does not open automatically, go to:

```text
http://localhost:8501
```

## Deploying

### Streamlit Community Cloud
1. Push this repository to GitHub.
2. Open https://share.streamlit.io and connect your GitHub account.
3. Create a new app using this repository with the `main` branch and `app.py` as the entrypoint.
4. Streamlit Cloud will install dependencies from `requirements.txt` and run your app automatically.

### Heroku
1. Install the Heroku CLI.
2. Log in with `heroku login`.
3. Create a new app with `heroku create your-app-name`.
4. Push to Heroku with `git push heroku main`.
5. Heroku will use the included `Procfile`, `runtime.txt`, and `requirements.txt` to launch the app.

### Notes for Deployment
- The app is configured to bind to `0.0.0.0` and use the `$PORT` environment variable.
- SQLite data persists to `storage/phishguard_ai.db`, so the deployed instance is best suited for demo and small-team use.
- For production, consider a managed database and secure secret handling.

## Project Structure

```text
phishguard_ai_enterprise/
├── app.py
├── train_model.py
├── predict.py
├── requirements.txt
├── README.md
├── .streamlit/
│   └── config.toml
├── app/
│   └── styles.css
├── artifacts/
│   ├── metrics.json
│   └── phishing_detector.joblib
├── data/
│   └── sample_phishing_dataset.csv
├── storage/
│   └── .gitkeep
├── src/
│   ├── auth.py
│   ├── config.py
│   ├── database.py
│   ├── inference.py
│   ├── rules.py
│   ├── text_utils.py
│   └── training.py
└── tests/
    └── test_rules.py
```

## Database Entities

- **users**: user accounts, password hashes, role, status, institution, and profile fields
- **scan_history**: saved predictions, probabilities, risk metadata, feedback, and source tracking
- **audit_log**: administrative and user activity trail for operations and security events
- **training_runs**: model retraining history with core evaluation metrics

## Notes

- The database file is created automatically at `storage/phishguard_ai.db`.
- The application is optimized for local academic deployment and demonstration.
- For free cloud hosting, Streamlit Community Cloud is the easiest target after pushing the project to GitHub.
