import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder


def load_model():
    with open("student_final_model.pkl","rb") as file:
        model, scaler, le=pickle.load(file)
    return model, scaler, le

def preprocessing_input_data(data,scaler, le):
    data["Extracurricular Activities"] = le.transform([data["Extracurricular Activities"]])[0]
    df = pd.DataFrame([data])
    df_transformed = scaler.transform(df)
    return df_transformed

def predict_data(data):
    model, scaler, le = load_model()
    process_data = preprocessing_input_data(data, scaler, le)
    prediction = model.predict(process_data)
    return prediction


def main():
    st.title("Student Performance Prediction")
    st.write("Enter your data to get a prediction for your performance")
    
    hour_study = st.number_input("Hour Studied", min_value=1, max_value=10, value=5)
    prev_score = st.number_input("Previous Score", min_value=40, max_value=100, value=70)
    extra = st.selectbox("Extra Curricular Activities", ["Yes", "No"])
    sleep_hour = st.number_input("Sleeping Hour", min_value=4, max_value=10, value=7)
    paper_count = st.number_input("Number of Question Paper Solved", min_value=0, max_value=10, value=5)

    if st.button("Predict Your Performance Score"):
        user_data = {
            "Hours Studied":hour_study,
            "Previous Scores":prev_score,
            "Extracurricular Activities": extra,
            "Sleep Hours": sleep_hour,
            "Sample Question Papers Practiced":paper_count
        }
        prediction = predict_data(user_data)
        st.success(f"Your Prediction Result is {prediction}")



if __name__ == "__main__":
    main()