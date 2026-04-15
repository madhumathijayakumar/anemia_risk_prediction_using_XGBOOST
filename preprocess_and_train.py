import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import shap

# Load dataset
df = pd.read_csv('anemia_extended_5000.csv')

# Features and target
X = df.drop('risk', axis=1)
y = df['risk']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train XGBoost
model = xgb.XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)

model.fit(X_train, y_train)

# ✅ SAVE ONLY JSON (IMPORTANT)
model.save_model("model_xgb.json")
print("Model saved as model_xgb.json")

# Evaluate
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# SHAP (optional check)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test[:10])  # small sample
print("SHAP working")