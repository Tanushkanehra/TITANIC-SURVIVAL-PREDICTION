import streamlit as st
import joblib
import numpy as np

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Titanic Survival Prediction")

pclass = st.selectbox("Passenger Class", [1,2,3])
sex = st.selectbox("Sex", ["Male","Female"])
age = st.slider("Age", 1, 80, 30)
sibsp = st.number_input("Siblings/Spouses", 0, 8, 0)
parch = st.number_input("Parents/Children", 0, 6, 0)
fare = st.number_input("Fare", 0.0, 600.0, 32.0)
embarked = st.selectbox("Embarked", ["S","C","Q"])

sex = 0 if sex == "Male" else 1
embarked = {"S":0,"C":1,"Q":2}[embarked]

family_size = sibsp + parch + 1
is_alone = 1 if family_size == 1 else 0

X = np.array([[pclass, sex, age, sibsp, parch, fare, embarked, 0, family_size, is_alone]])
X[:,[2,5]] = scaler.transform(X[:,[2,5]])

if st.button("Predict"):
    pred = model.predict(X)
    st.success("Survived" if pred[0]==1 else "Did Not Survive")
