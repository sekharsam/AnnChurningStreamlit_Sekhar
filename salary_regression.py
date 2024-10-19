import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle as pk
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

##Load the trained model
model = tf.keras.models.load_model('regression_model.h5')

##load all the pickle files
with open('salary_label_encoder_gender.pkl','rb') as file:
   label_encoder_gender = pk.load(file)

with open('salary_onehot_encode_geo.pkl','rb') as file:
    one_hot_encoder_geo = pk.load(file)

with open('salary_scale.pkl','rb') as file:
    scaler = pk.load(file)

## streamlit app
st.title('Salary prediction')

##User input
geography = st.selectbox('Geography', one_hot_encoder_geo.categories_[0])
gender =    st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18,92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
#estimated_salary = st.number_input('Estimated salary')
tenure = st.slider('Tenure', 0,10)
num_of_products=st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0,4])
is_active_member = st.selectbox('Is Active Member', [0,1])
has_exited= st.selectbox('Has Exited', [0,1])

##prepare input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'Exited':[has_exited]
})


#One-hot encode Geography
geo_encoded= one_hot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=one_hot_encoder_geo.get_feature_names_out(['Geography']))
geo_encoded_df


##concat geo encoded with input data 
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# scale the input
input_data_scaled=scaler.transform(input_data)

#predict the salary
prediction = model.predict(input_data_scaled)
predicted_salary = prediction[0][0]

st.write(f'Predicted Estimated Salary : {predicted_salary: .2f}')