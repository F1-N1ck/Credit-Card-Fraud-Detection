import kagglehub
from preprocess import load_data_kagglehub, preprocess_data
from models import train_rf, train_gb, train_xgb, evaluate_model
import matplotlib.pyplot as plt

# Download using kagglehub
path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
df = load_data_kagglehub(path)

# Preprocess
X_train, X_test, y_train, y_test = preprocess_data(df)

# Train models
rf = train_rf(X_train, y_train)
gb = train_gb(X_train, y_train)
xgb_model = train_xgb(X_train, y_train)

# Evaluate
plt.figure(figsize=(8, 6))
evaluate_model(rf, X_test, y_test, "Random Forest")
evaluate_model(gb, X_test, y_test, "Gradient Boosting")
evaluate_model(xgb_model, X_test, y_test, "XGBoost")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves")
plt.legend()
plt.show()
