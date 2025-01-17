import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
import pickle


### streamlit app
st.set_page_config(
    page_title="Customer Churn Prediction", page_icon="🕵️‍♂️", layout="wide"
)

@st.cache_resource
def load_data():
    model = load_model('model.h5')
    with open('onehotencoder.pkl', 'rb') as file:
        encoder = pickle.load(file)

    with open('label_encoder_gender.pkl', 'rb') as file:
        label_encoder_gender = pickle.load(file)

    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    
    return model,encoder,label_encoder_gender,scaler

model,encoder,label_encoder_gender,scaler = load_data()




st.title('🕵️‍♂️ Customer Churn Prediction')

tab1, tab2 = st.tabs(["Introduction", "Predict Churn"])


with tab1:
    st.subheader('| Introduction')
    intro_text = '''
    Customer churn prediction predicts the likelihood of customers canceling a company’s 
    products or services. In most cases, businesses with repeat clients or clients under 
    subscriptions strive to maintain the customer base. Therefore, it is important to keep 
    track of the customers who cancel their subscription plan and those who continue with 
    the service. This approach requires the organization to know and understand their client’s 
    behavior and the attributes that lead to the risk of the client leaving.
    '''
    st.write(f'<p style="text-align: justify;">{intro_text}<p/>',unsafe_allow_html=True)

    st.subheader('| Why Predict Customer Churn?')
    why_text = '''
    It is important for any organization dealing with repeat clients to find ways to retain existing 
    ones. The approach is crucial since customer churn is expensive, and acquiring new clients is more 
    expensive than retaining existing ones. Consider an internet service (ISP) provider who has acquired 
    a new user. They will need technicians and hardware to connect the latest client to their service. 
    The client will only be required to pay the subscription fee to continue using their plan. If the 
    user fails to renew their service, the company will most likely be at a loss, especially if the trend 
    continues for several customers. The monthly recurring revenue (MRR) for such an institution will likely 
    be low; hence, it will be unable to sustain the business. Thus, a reliable churn prediction model should 
    help companies stay afloat as they scale up and attract more customers.
    '''
    st.write(f'<p style="text-align: justify;">{why_text}<p/>',unsafe_allow_html=True)

    st.subheader("| Github")
    st.write(
        '<p>To check out the code, Visit <a href="https://github.com/Koushik-j/Artificial_Neural_Network-Classification-Customer-Churn-Prediction">GitHub</a>.</p><br>',
        unsafe_allow_html=True,
    )

with tab2:
    st.subheader('| Predict Customer Churn')
    geography = st.selectbox('🗺️Geography', encoder.categories_[0])
    gender = st.selectbox('🧑🏼‍🦱Gender',label_encoder_gender.classes_)
    age = st.slider('🔞Age',18,100)
    balance = st.number_input('⚖️Balance')
    credit_score = st.number_input('💯Credit Score')
    estimated_salary = st.number_input('💰Estimated Salary')
    tenure = st.slider('⏳Tenure',0,10)
    num_of_products = st.slider('👜Number of Products',1,4)
    has_credit_card = st.checkbox('💳Has Credit Card',[0,1])
    is_active_member = st.checkbox('😎Is Active Member',[0,1]) 

    st.markdown("---")

    col1, col2, col3 , col4, col5 = st.columns(5)

    with col1:
        pass
    with col2:
        pass
    with col4:
        pass
    with col5:
        pass
    with col3 :
        if st.button('Predict'):


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
                st.error('Customer is likely to churn')
            else:
                st.success('Customer is not likely to churn')

