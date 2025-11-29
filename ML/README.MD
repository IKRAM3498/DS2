# IKRAM EL-WALI
# CAC2

Wine Quality Prediction - Machine Learning Report
Dataset: Vinho Verde (Red & White Wines)
Author: 
Date:


# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load datasets
red = pd.read_csv("winequality-red.csv", sep=';')
white = pd.read_csv("winequality-white.csv", sep=';')

# Merge datasets (optional)
red['type'] = 'red'
white['type'] = 'white'
wine = pd.concat([red, white], ignore_index=True)

# Feature selection
X = wine.drop(['quality'], axis=1)
y = wine['quality']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train_scaled, y_train)

# Predictions
y_pred = model.predict(X_test_scaled)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MSE:", mse)
print("R2 Score:", r2)

# Feature importance
importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print(importances)

# Save results
importances.to_csv("feature_importance.csv", index=False)

# Plot feature importance
plt.figure(figsize=(10,6))
plt.bar(importances['Feature'], importances['Importance'])
plt.xticks(rotation=45)
plt.title("Feature Importance - Random Forest")
plt.tight_layout()
plt.savefig("feature_importance.png")
