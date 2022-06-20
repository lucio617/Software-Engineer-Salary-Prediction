import streamlit as st
import pickle
import numpy as np

def load_models():
    with open('saved_steps.pkl','rb') as file:
        data=pickle.load(file)
    return data

data=load_models()

regressor=data['model']
le_country=data['le_country']
le_education=data['le_education']

def show_predict_page():
    st.title("Software Developer Salary Prediction")

    st.write("""### We need some information to predict the salary""")

    countries = (
        "United States of America",
        "India",
        "United Kingdom",
        "Germany",
        "Canada",
        "Brazil",
        "France",
        "Spain",
        "Australia",
        "Netherlands",
        "Poland",
        "Italy",
        "Russian Federation",
        "Sweden",
    )

    education = (
        "Less than a Bachelors",
        "Bachelor’s degree",
        "Master’s degree",
        "Post grad",
    )

    country=st.selectbox('Country',countries)#drop down select box
    education=st.selectbox('Education Level',education)

    experience=st.slider('Years of experience',0,50,3)

    ok=st.button('Calculate Salary')## ok holds a boolean if button is clicked
    if ok:
        X = np.array([[country, education, experience ]])
        X[:, 0] = le_country.transform(X[:,0])##first column
        X[:, 1] = le_education.transform(X[:,1])## second column
        X = X.astype(float)

        salary=regressor.predict(X)

        st.subheader(f"The estimated salary is ${salary[0]:.2f}")
