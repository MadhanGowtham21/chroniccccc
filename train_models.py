import pandas as pd
import os
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


os.makedirs("../models", exist_ok=True)

datasets = {
    "diabetes": "datasets/Diabetes.csv",
    "hypertension": "datasets/Hypertension.csv",
    "cardiovascular": "datasets/Cardiovascular_Disease.csv",
    "ckd": "datasets/Chronic_Kidney_Disease.csv",
    "copd": "datasets/Chronic_Obstructive_Pulmonary_Disease.csv",
    "asthma": "datasets/Asthma.csv",
    "liver": "datasets/Liver_Disease.csv"
}

for name, path in datasets.items():
    print(f"\n📁 Processing {name.title()} Dataset")
    df = pd.read_csv(path)
    df.dropna(inplace=True)

    le_dict = {}
    for col in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le

    X = df.drop("Disease", axis=1)
    y = df["Disease"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    selector = SelectKBest(score_func=f_classif, k='all')  # or k=10
    X_selected = selector.fit_transform(X_scaled, y)

    sm = SMOTE(random_state=42)
    X_resampled, y_resampled = sm.fit_resample(X_selected, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.2, random_state=42
    )

    rf = RandomForestClassifier()
    rf_params = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }
    rf_search = RandomizedSearchCV(rf, rf_params, n_iter=5, cv=3, scoring='accuracy', random_state=42)
    rf_search.fit(X_train, y_train)
    best_rf = rf_search.best_estimator_

    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    xgb_params = {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 5, 7]
    }
    xgb_search = RandomizedSearchCV(xgb, xgb_params, n_iter=5, cv=3, scoring='accuracy', random_state=42)
    xgb_search.fit(X_train, y_train)
    best_xgb = xgb_search.best_estimator_

    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)

    voting = VotingClassifier(estimators=[
        ('lr', lr),
        ('rf', best_rf),
        ('xgb', best_xgb)
    ], voting='soft')

    voting.fit(X_train, y_train)
    y_pred = voting.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"🎯 Voting Classifier Accuracy: {acc:.4f}")

    # Save the entire pipeline pieces for inference
    save_data = {
        "model": voting,
        "scaler": scaler,
        "label_encoders": le_dict,
        "feature_selector": selector
    }
    joblib.dump(save_data, f"models/model_{name}.pkl")
    print(f"✅ Saved VotingClassifier for {name.title()} with accuracy {acc:.4f}")
