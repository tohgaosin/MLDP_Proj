import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn import datasets

# Load the pre-trained model
model = joblib.load('stroke_rf.pkl')

st.set_page_config(page_title="Stroke Prediction App", page_icon="🧠", layout="wide")

st.write("""
# Stroke Prediction App 🧠
This app predicts the likelihood of **stroke** based on user input.
""")

# Add some CSS for styling
st.markdown("""
<style>
    .big-font {
        font-size: 24px !important;
        font-weight: bold;
    }
    .medium-font {
        font-size: 18px !important;
        font-weight: normal;
    }
    .section-header {
        color: #1f77b4;
        font-size: 22px;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .subheader {
        font-size: 18px;
        color: #4CAF50;
    }
</style>
""", unsafe_allow_html=True)

st.sidebar.header('User Input Parameters')

def user_input_features():
    age = st.sidebar.slider('Age', 0, 100, 50)
    hypertension = st.sidebar.selectbox('Hypertension', ('No', 'Yes'))
    heart_disease = st.sidebar.selectbox('Heart Disease', ('No', 'Yes'))
    avg_glucose_level = st.sidebar.slider('Average Glucose Level', 50.0, 300.0, 100.0)
    height = st.sidebar.slider('Height (cm)', 100, 220, 170)
    weight = st.sidebar.slider('Weight (kg)', 30, 150, 70)
    bmi = weight / ((height / 100) ** 2)  # BMI calculation
    gender = st.sidebar.selectbox('Gender', ('Male', 'Female'))
    ever_married = st.sidebar.selectbox('Ever Married', ('Yes', 'No'))
    work_type = st.sidebar.selectbox('Work Type', 
                                     ('Private', 'Self-employed', 'Govt_job'))    
    residence_type = st.sidebar.selectbox('Residence Type', ('Urban', 'Rural'))
    smoking_status = st.sidebar.selectbox('Smoking Status', 
                                          ('Unknown', 'Formerly Smoked', 'Never Smoked', 'Smokes'))
    
    # Adjust the feature names to match the ones used during model training
    data = {
        'age': age,
        'hypertension': 1 if hypertension == 'Yes' else 0,
        'heart_disease': 1 if heart_disease == 'Yes' else 0,
        'avg_glucose_level': avg_glucose_level,
        'bmi': bmi,
        'gender_Female': 1 if gender == 'Female' else 0,
        'gender_Male': 1 if gender == 'Male' else 0,
        'ever_married_No': 1 if ever_married == 'No' else 0,
        'ever_married_Yes': 1 if ever_married == 'Yes' else 0,
        'work_type_Govt_job': 1 if work_type == 'Govt_job' else 0,
        'work_type_Private': 1 if work_type == 'Private' else 0,
        'work_type_Self-employed': 1 if work_type == 'Self-employed' else 0,
        'Residence_type_Rural': 1 if residence_type == 'Rural' else 0,
        'Residence_type_Urban': 1 if residence_type == 'Urban' else 0,
        'smoking_status_Unknown': 1 if smoking_status == 'Unknown' else 0,
        'smoking_status_formerly smoked': 1 if smoking_status == 'Formerly Smoked' else 0,
        'smoking_status_never smoked': 1 if smoking_status == 'Never Smoked' else 0,
        'smoking_status_smokes': 1 if smoking_status == 'Smokes' else 0
    }
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

# Display user input parameters in a clean table
st.subheader('User Input Parameters:')
st.write(df.style.set_properties(**{'text-align': 'center'}))

# Predict with the trained model
prediction = model.predict(df)
prediction_proba = model.predict_proba(df)

# Get the class labels
class_labels = model.classes_

# Display the prediction result
st.markdown(f'<div class="section-header">Prediction</div>', unsafe_allow_html=True)
st.write(f"Predicted: **{'You have stroke' if int(prediction[0]) == 1 else 'No stroke'}**")

# Display prediction probabilities
st.markdown(f'<div class="subheader">Prediction Probability</div>', unsafe_allow_html=True)
st.write(f"Probability of Stroke: **{prediction_proba[0][1] * 100:.2f}%**")
st.write(f"Probability of No Stroke: **{prediction_proba[0][0] * 100:.2f}%**")

# BMI Classification based on updated ranges
user_bmi = df['bmi'][0]
if 18.5 <= user_bmi <= 24.9:
    bmi_classification = 'Healthy weight range for young and middle-aged adults'
elif 25.0 <= user_bmi <= 29.9:
    bmi_classification = 'Overweight'
else:
    bmi_classification = 'Obese'

st.markdown(f'<div class="section-header">BMI Classification</div>', unsafe_allow_html=True)
st.write(f'Your BMI: **{user_bmi:.2f}** – {bmi_classification}')

# Risk comparison with general population
average_bmi = 28  # Example average BMI for general population
bmi_comparison = 'Normal' if user_bmi < 24.9 else 'Overweight' if user_bmi < 30 else 'Obese'
bmi_message = f'Your BMI: **{user_bmi:.2f}** ({bmi_comparison}) vs. General Population Average BMI: **{average_bmi}** (Overweight)'

# Plotting BMI comparison with general population using Streamlit's line chart
st.markdown(f'<div class="section-header">BMI Comparison with General Population</div>', unsafe_allow_html=True)
st.write(bmi_message)



# 1. Age Distribution (Histogram)
st.markdown(f'<div class="section-header">Age Distribution</div>', unsafe_allow_html=True)
ages = np.random.randint(0, 100, size=500)  # Random age data for illustration
st.scatter_chart(pd.Series(ages).value_counts())
