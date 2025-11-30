# -*- coding: utf-8 -*-
"""
Script complet pour prédiction du risque élevé avec plusieurs modèles
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import ADASYN
from sklearn.metrics import classification_report, accuracy_score, f1_score

# -----------------------------
# 1. Charger le dataset
# -----------------------------
df = pd.read_csv('medical_insurance.csv')  # Modifier avec le chemin correct

# -----------------------------
# 2. Préparer les données
# -----------------------------
# Séparer colonnes numériques et catégorielles
num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

# Encoder les colonnes catégorielles
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

# Remplir les NaN
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

df['is_high_risk'] = df['is_high_risk'].fillna(df['is_high_risk'].mode()[0])

# Séparer X et y
X = df.drop('is_high_risk', axis=1)
y = df['is_high_risk']

# -----------------------------
# 3. Rééchantillonnage ADASYN
# -----------------------------
adasyn = ADASYN(random_state=42)
X_resampled, y_resampled = adasyn.fit_resample(X, y)

# -----------------------------
# 4. Split train/test
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.25, random_state=0
)

# -----------------------------
# 5. Entraîner les modèles
# -----------------------------
# Logistic Regression
lr_model = LogisticRegression(random_state=0, max_iter=1000)
lr_model.fit(X_train, y_train)
y_lr_pred = lr_model.predict(X_test)

# Decision Tree
dt_model = DecisionTreeClassifier(criterion='entropy', random_state=0)
dt_model.fit(X_train, y_train)
y_dt_pred = dt_model.predict(X_test)

# K-Nearest Neighbors
knn_model = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
knn_model.fit(X_train, y_train)
y_knn_pred = knn_model.predict(X_test)

# XGBoost
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train, y_train)
y_xgb_pred = xgb_model.predict(X_test)

# -----------------------------
# 6. Évaluer les modèles
# -----------------------------
models = {
    "Logistic Regression": y_lr_pred,
    "Decision Tree": y_dt_pred,
    "KNN": y_knn_pred,
    "XGBoost": y_xgb_pred
}

for name, pred in models.items():
    print(f"\n==== {name} ====")
    print("Accuracy:", accuracy_score(y_test, pred))
    print("F1-score:", f1_score(y_test, pred))
    print(classification_report(y_test, pred))

# -----------------------------
# 7. Optionnel : sauvegarder les prédictions
# -----------------------------
pd.DataFrame(y_lr_pred, columns=['LR_Predicted']).to_csv('LR_predictions.csv', index=False)
pd.DataFrame(y_dt_pred, columns=['DT_Predicted']).to_csv('DT_predictions.csv', index=False)
pd.DataFrame(y_knn_pred, columns=['KNN_Predicted']).to_csv('KNN_predictions.csv', index=False)
pd.DataFrame(y_xgb_pred, columns=['XGB_Predicted']).to_csv('XGB_predictions.csv', index=False)

print("\nPrédictions sauvegardées dans des fichiers CSV.")
