# EBPL-PROJECT
DATA SCIENCE# Install dependencies if needed
# !pip install pandas numpy scikit-learn matplotlib seaborn xgboost streamlit

# 1️⃣ Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
import streamlit as st

# 2️⃣ Load Dataset
data = pd.read_csv('Telco-Customer-Churn.csv')  # replace with your dataset path
print(data.head())

# 3️⃣ Data Preprocessing
# Handle missing values
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
data.dropna(inplace=True)

# Remove unnecessary columns
data.drop(['customerID'], axis=1, inplace=True)

# Encode categorical variables
label_enc = LabelEncoder()
for col in data.select_dtypes(include='object').columns:
    data[col] = label_enc.fit_transform(data[col])

# 4️⃣ Feature Scaling
scaler = StandardScaler()
scaled_features = scaler.fit_transform(data.drop('Churn', axis=1))
X = pd.DataFrame(scaled_features, columns=data.columns[:-1])
y = data['Churn']

# 5️⃣ Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6️⃣ Model Building

# Logistic Regression
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
log_pred = log_reg.predict(X_test)

# Random Forest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

# XGBoost
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb.fit(X_train, y_train)
xgb_pred = xgb.predict(X_test)

# 7️⃣ Model Evaluation Function
def evaluate_model(model_name, y_test, y_pred):
    print(f"Model: {model_name}")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title(f'{model_name} Confusion Matrix')
    plt.show()

# Evaluate Models
evaluate_model("Logistic Regression", y_test, log_pred)
evaluate_model("Random Forest", y_test, rf_pred)
evaluate_model("XGBoost", y_test, xgb_pred)

# 8️⃣ ROC Curve Example for Random Forest
rf_probs = rf.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, rf_probs)
plt.plot(fpr, tpr, label='Random Forest (area = %0.2f)' % roc_auc_score(y_test, rf_probs))
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# 🔥 Optional: Streamlit App
# Save this block as app.py to run `streamlit run app.py`
def run_app():
    st.title("Customer Churn Prediction")

    # Input fields
    tenure = st.slider("Tenure (months)", 0, 80, 24)
    monthly_charges = st.slider("Monthly Charges", 20, 120, 70)
    total_charges = st.slider("Total Charges", 20, 8000, 1000)

    if st.button("Predict Churn"):
        user_data = np.array([[tenure, monthly_charges, total_charges]])
        user_data_scaled = scaler.transform(user_data)
        prediction = rf.predict(user_data_scaled)

        if prediction[0] == 1:
            st.error("Churn Likely 😢")
        else:
            st.success("Customer will stay 👍")

# Uncomment below to run streamlit app directly
# if __name__ == '__main__':
#     run_app()

