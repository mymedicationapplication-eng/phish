from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Tuple

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .config import ARTIFACTS_DIR, DATA_PATH, METRICS_PATH, MODEL_PATH, RANDOM_STATE
from .text_utils import normalize_text


@dataclass
class TrainingSummary:
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    train_size: int
    test_size: int
    confusion_matrix: list
    class_report: Dict[str, Dict[str, float]]



def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {'text', 'label'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'Dataset is missing required columns: {sorted(missing)}')
    df = df.dropna(subset=['text', 'label']).copy()
    df['text'] = df['text'].astype(str).map(normalize_text)
    df['label'] = df['label'].astype(int)
    df = df[df['text'].str.len() > 0].copy()
    return df



def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            (
                'vectorizer',
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    stop_words='english',
                    min_df=1,
                    max_df=0.95,
                    sublinear_tf=True,
                ),
            ),
            (
                'classifier',
                LogisticRegression(
                    max_iter=2000,
                    class_weight='balanced',
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )



def train_and_save_model(data_path: Path = DATA_PATH) -> Tuple[Pipeline, TrainingSummary]:
    df = load_dataset(data_path)
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'],
        df['label'],
        test_size=0.25,
        stratify=df['label'],
        random_state=RANDOM_STATE,
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, predictions, average='binary', zero_division=0
    )
    roc_auc = roc_auc_score(y_test, probabilities)
    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
    matrix = confusion_matrix(y_test, predictions).tolist()

    summary = TrainingSummary(
        accuracy=float(accuracy),
        precision=float(precision),
        recall=float(recall),
        f1_score=float(f1),
        roc_auc=float(roc_auc),
        train_size=int(len(X_train)),
        test_size=int(len(X_test)),
        confusion_matrix=matrix,
        class_report=report,
    )

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    with open(METRICS_PATH, 'w', encoding='utf-8') as f:
        json.dump(asdict(summary), f, indent=2)

    return pipeline, summary
