# ==========================================
# WEEK 2 ASSIGNMENT: AI FOR SUSTAINABLE DEVELOPMENT
# Project: CreditPath - Financial Inclusion Prediction
# Author: Brandon Mutua Mwanza
# SDG: 1 - No Poverty
# ==========================================

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. GENERATE DUMMY DATA
print("🔄 Generating synthetic data...")
data = pd.DataFrame({
    'age': np.random.randint(18, 80, 500),
    'education_level': np.random.choice(['Primary', 'Secondary', 'Tertiary', 'None'], 500),
    'job_type': np.random.choice(['Farming', 'Self employed', 'Formal', 'Government'], 500),
    'location_type': np.random.choice(['Rural', 'Urban'], 500),
    'has_bank_account': np.random.choice([0, 1], 500, p=[0.6, 0.4])
})

# 2. PREPROCESSING
data_encoded = pd.get_dummies(data, columns=['education_level', 'job_type', 'location_type'], drop_first=True)
X = data_encoded.drop('has_bank_account', axis=1)
y = data_encoded['has_bank_account']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. TRAIN MODEL
print("🤖 Training Random Forest model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. EVALUATE
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"\n🚀 Model Accuracy: {accuracy * 100:.2f}%")
print("\n📊 Classification Report:")
print(classification_report(y_test, predictions))

# 5. SAVE THE GRAPH (Crucial for Codespaces/GitHub)
print("💾 Saving Confusion Matrix image...")
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, predictions), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix: Predicted vs Actual')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig('confusion_matrix.png') # <--- This saves the file instead of showing it
print("✅ Done! Check your file explorer for 'confusion_matrix.png'")