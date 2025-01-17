import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
import pickle


## Load the trained model

model = load_model('model.h5')


## Load the encoder and scaler 

with open('onehotencoder.pkl', 'rb') as file:
    encoder = pickle.load(file)

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)



### streamlit app

st.title('Customer Churn Prediction')

geography = st.selectbox('Geography', encoder.categories_[0])
gender = st.selectbox('Gender',label_encoder_gender.classes_)
age = st.slider('Age',18,100)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure',0,10)
num_of_products = st.slider('Number of Products',1,4)
has_credit_card = st.selectbox('Has Credit Card',[0,1])
is_active_member = st.selectbox('Is Active Member',[0,1]) 


input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender':[label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_credit_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

geo_encoded = encoder.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns = encoder.get_feature_names_out(['Geography']))

input_data = pd.concat([input_data.reset_index(drop=True),geo_encoded_df],axis=1)

input_data_scaled = scaler.transform(input_data)

prediction = model.predict(input_data_scaled)
prediction_probablity = prediction[0][0]

st.write('Prediction:', prediction_probablity)

if prediction_probablity > 0.5:
    st.write('Customer is likely to churn')
else:
    st.write('Customer is not likely to churn')

