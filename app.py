import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load training dataset from Excel
train_data = pd.read_csv('logistic_train_data.csv')

# Separate features and target variable
X_train = train_data.drop(columns=['depression'])  # Assuming 'target' is the column name for labels
y_train = train_data['depression']

# Train Logistic Regression model
logistic_model = LogisticRegression(random_state=42)
logistic_model.fit(X_train, y_train)

st.title("Depression Prediction App")

# Input fields for user input
age = st.number_input('Age', min_value=18, max_value=34)
study_satisfaction = st.slider('Study Satisfaction (0-5)', 0, 5)
dietary_habits = st.slider('Dietary Habits (0-3)', 0, 3)
suicidal_thoughts = st.selectbox('Suicidal Thoughts (Yes=1, No=0)', [0, 1])
work_study_hours = st.slider('Work/Study Hours', 0, 12)
overall_stress = st.slider('Overall Stress (0-9)', 0, 9)

# Predict button
if st.button('Predict Depression'):
    input_data = np.array([[age, study_satisfaction, dietary_habits, suicidal_thoughts, work_study_hours, overall_stress]])
    prediction = logistic_model.predict(input_data)

    if prediction[0] == 1:
        st.error('Prediction: Depression')
    else:
        st.success('Prediction: No Depression')
