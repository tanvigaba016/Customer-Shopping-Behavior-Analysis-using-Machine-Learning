import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

# ================= LOAD DATA =================
@st.cache_data
def load_data():
    df = pd.read_csv("shopping_behavior_updated.csv")
    return df

df = load_data()

st.title("🛍 Customer Analytics System")

# ================= TARGETS =================
threshold = df['Purchase Amount (USD)'].median()
df['Spender'] = df['Purchase Amount (USD)'].apply(lambda x: 1 if x >= threshold else 0)

df['Subscription Status'] = df['Subscription Status'].map({'Yes':1,'No':0})

# ================= FEATURE ENGINEERING =================
df['Avg Purchase Value'] = df['Purchase Amount (USD)'] / (df['Previous Purchases'] + 1)
df['Engagement Score'] = df['Review Rating'] + df['Previous Purchases']

# ================= EDA CLEAN =================
eda_df = df.drop(columns=['Subscription Status'])

# ================= TABS =================
tab1, tab2, tab3 = st.tabs(["📊 EDA", "💰 Spender Prediction", "📩 Subscription Prediction"])

# =================================================
# 📊 TAB 1 — EDA (FROM PREVIOUS VERSION)
# =================================================
with tab1:
    st.subheader("Dataset Overview")
    st.write(eda_df.head())

    st.subheader("Summary Statistics")
    st.write(eda_df.describe())

    st.subheader("Gender Distribution")
    st.bar_chart(eda_df['Gender'].value_counts())

    st.subheader("Category Distribution")
    st.bar_chart(eda_df['Category'].value_counts())

    st.subheader("Payment Method Distribution")
    st.bar_chart(eda_df['Payment Method'].value_counts())

    st.subheader("Purchase Amount Distribution")
    st.line_chart(eda_df['Purchase Amount (USD)'])

    st.subheader("Review Rating Distribution")
    st.bar_chart(eda_df['Review Rating'].value_counts())

# =================================================
# 💰 TAB 2 — SPENDER (SMOTE FIXED)
# =================================================
with tab2:

    st.subheader("Predict High / Low Spender")

    features = [
        'Age',
        'Previous Purchases',
        'Review Rating',
        'Avg Purchase Value',
        'Engagement Score'
    ]

    X = df[features]
    y = df['Spender']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ⭐ SMOTE
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_scaled, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        random_state=42
    )

    model.fit(X_train, y_train)

    acc = model.score(X_test, y_test)
    st.write(f"Model Accuracy: *{round(acc,3)}*")

    # -------- INPUT --------
    age = st.slider("Age", 18, 80, 25)
    prev = st.slider("Previous Purchases", 0, 50, 5)
    rating = st.slider("Review Rating", 1, 5, 3)
    avg = st.slider("Avg Purchase Value", 10, 500, 100)
    eng = st.slider("Engagement Score", 1, 60, 10)

    if st.button("Predict Spender"):
        pred = model.predict(scaler.transform([[age, prev, rating, avg, eng]]))[0]

        if pred == 1:
            st.success("High Spender 💰")
        else:
            st.warning("Low Spender")

# =================================================
# 📩 TAB 3 — SUBSCRIPTION (SMOTE FIXED)
# =================================================
with tab3:

    st.subheader("Predict Subscription Status")

    features = [
        'Age',
        'Previous Purchases',
        'Review Rating',
        'Avg Purchase Value',
        'Engagement Score'
    ]

    X2 = df[features]
    y2 = df['Subscription Status']

    scaler2 = StandardScaler()
    X2_scaled = scaler2.fit_transform(X2)

    # ⭐ SMOTE
    sm2 = SMOTE(random_state=42)
    X2_res, y2_res = sm2.fit_resample(X2_scaled, y2)

    X2_train, X2_test, y2_train, y2_test = train_test_split(
        X2_res, y2_res, test_size=0.2, random_state=42
    )

    model2 = RandomForestClassifier(
        n_estimators=120,
        max_depth=7,
        random_state=42
    )

    model2.fit(X2_train, y2_train)

    acc2 = model2.score(X2_test, y2_test)
    st.write(f"Model Accuracy: *{round(acc2,3)}*")

    # -------- INPUT --------
    age2 = st.slider("Age ", 18, 80, 25, key="a2")
    prev2 = st.slider("Previous Purchases ", 0, 50, 5, key="p2")
    rating2 = st.slider("Review Rating ", 1, 5, 3, key="r2")
    avg2 = st.slider("Avg Purchase Value ", 10, 500, 100, key="avg2")
    eng2 = st.slider("Engagement Score ", 1, 60, 10, key="eng2")

    if st.button("Predict Subscription"):
        pred2 = model2.predict(scaler2.transform([[age2, prev2, rating2, avg2, eng2]]))[0]

        if pred2 == 1:
            st.success("Will Subscribe 📩")
        else:
            st.error("Will NOT Subscribe")