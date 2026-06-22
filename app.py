import streamlit as st
import pandas as pd
import joblib

# =========================================================
# LOAD MODEL
# =========================================================
bundle_p  = joblib.load('best_attrition_model.pkl')

model_p   = bundle_p['model']
scaler_p  = bundle_p['scaler']
le_map    = bundle_p['label_encoders']
feat_ord  = bundle_p['feature_names']
is_scaled = bundle_p['needs_scaling']

cat_cols = ['JobRole', 'Gender', 'Department', 'MaritalStatus', 'OverTime']

# =========================================================
# UI TITLE
# =========================================================
st.set_page_config(page_title="Employee Attrition Predictor")
st.title("💼 Employee Attrition Prediction")

# =========================================================
# INPUT UI
# =========================================================
st.sidebar.header("Enter Employee Details")

# Numeric Inputs
YearsSinceLastPromotion = st.sidebar.number_input("Years Since Last Promotion", 0, 20, 3)
JobSatisfaction = st.sidebar.selectbox("Job Satisfaction", [1, 2, 3, 4])
JobLevel = st.sidebar.selectbox("Job Level", [1, 2, 3, 4, 5])
YearsInCurrentRole = st.sidebar.number_input("Years In Current Role", 0, 40, 4)
WorkLifeBalance = st.sidebar.selectbox("Work Life Balance", [1, 2, 3, 4])
JobInvolvement = st.sidebar.selectbox("Job Involvement", [1, 2, 3, 4])
RelationshipSatisfaction = st.sidebar.selectbox("Relationship Satisfaction", [1, 2, 3, 4])
YearsAtCompany = st.sidebar.number_input("Years At Company", 0, 40, 5)
MonthlyIncome = st.sidebar.number_input("Monthly Income", 1000, 50000, 5000)

# ✅ FIXED HERE (use slider instead of wrong selectbox)
TrainingTimesLastYear = st.sidebar.slider(
    "Training Times Last Year",
    min_value=0,
    max_value=100,
    value=0
)

EnvironmentSatisfaction = st.sidebar.selectbox("Environment Satisfaction", [1, 2, 3, 4])
Age = st.sidebar.number_input("Age", 18, 60, 30)
HourlyRate = st.sidebar.number_input("Hourly Rate", 10, 150, 60)
PerformanceRating = st.sidebar.selectbox("Performance Rating", [1, 2, 3, 4])
DistanceFromHome = st.sidebar.number_input("Distance From Home", 0, 100, 10)
NumCompaniesWorked = st.sidebar.number_input("Num Companies Worked", 0, 10, 2)

# Categorical Inputs
JobRole = st.sidebar.selectbox("Job Role", le_map['JobRole'].classes_)
Gender = st.sidebar.selectbox("Gender", le_map['Gender'].classes_)
Department = st.sidebar.selectbox("Department", le_map['Department'].classes_)
MaritalStatus = st.sidebar.selectbox("Marital Status", le_map['MaritalStatus'].classes_)
OverTime = st.sidebar.selectbox("Over Time", le_map['OverTime'].classes_)

# =========================================================
# PREDICT BUTTON
# =========================================================
if st.button("🔍 Predict Attrition"):

    employee = {
        'YearsSinceLastPromotion': YearsSinceLastPromotion,
        'JobSatisfaction': JobSatisfaction,
        'JobLevel': JobLevel,
        'YearsInCurrentRole': YearsInCurrentRole,
        'WorkLifeBalance': WorkLifeBalance,
        'JobInvolvement': JobInvolvement,
        'RelationshipSatisfaction': RelationshipSatisfaction,
        'YearsAtCompany': YearsAtCompany,
        'MonthlyIncome': MonthlyIncome,
        'TrainingTimesLastYear': TrainingTimesLastYear,
        'EnvironmentSatisfaction': EnvironmentSatisfaction,
        'Age': Age,
        'HourlyRate': HourlyRate,
        'PerformanceRating': PerformanceRating,
        'DistanceFromHome': DistanceFromHome,
        'NumCompaniesWorked': NumCompaniesWorked,
        'JobRole': JobRole,
        'Gender': Gender,
        'Department': Department,
        'MaritalStatus': MaritalStatus,
        'OverTime': OverTime
    }

    # =========================================================
    # PREPROCESS
    # =========================================================
    input_data = {}

    for feat in feat_ord:
        if feat in cat_cols:
            input_data[feat] = le_map[feat].transform([employee[feat]])[0]
        else:
            input_data[feat] = employee[feat]

    input_df = pd.DataFrame([input_data])[feat_ord]

    if is_scaled:
        input_df = scaler_p.transform(input_df)

    # =========================================================
    # PREDICTION
    # =========================================================
    pred = model_p.predict(input_df)[0]

    # =========================================================
    # OUTPUT
    # =========================================================
    st.subheader("📊 Prediction Result")

    if pred == 1:
        st.error("❌ Employee likely to LEAVE")
    else:
        st.success("✅ Employee likely to STAY")
