import sys
print("USING PYTHON:", sys.executable)
print("TF EXISTS:", end=" ")

try:
    import tensorflow as tf
    print(tf.__version__)
except:
    print("NO TENSORFLOW FOUND!")


import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler ,LabelEncoder,OneHotEncoder
import pickle
import streamlit as st

#loading the trained model
model=tf.keras.models.load_model("model.h5")
#load the encoders and scalar
with open ("label_encoder_gender.pkl",'rb') as file:
    label_encoder_gender=pickle.load(file)

with open("ohe.pkl",'rb') as file:
    onehot_encoder_geo=pickle.load(file)
with open("scaler.pkl",'rb') as file:
    scaler=pickle.load(file)

st.title("customer churn prediction")
geography=st.selectbox("Geography",onehot_encoder_geo.categories_[0])
gender=st.selectbox("Gender",label_encoder_gender.classes_)
age=st.slider("Age",18,100)
balance=st.number_input("Balance")
credit_score=st.number_input("credit score")
estimated_salary=st.number_input("estimated salary")
tenure=st.slider("tenure",0,10)
numof_products=st.slider("number of profucts",1,4)
cr_card=st.selectbox("has the creditcard",[0,1])
is_active_member=st.selectbox("is active member now ",[0,1])

#prepare the data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [numof_products],
    'HasCrCard': [cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# one-hot encode geography
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out())

# merge numerical + encoded features
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# ⭐ REORDER COLUMNS FOR SCALER
input_data = input_data[scaler.feature_names_in_]

# scale and predict
input_data_scaled = scaler.transform(input_data)
prediction_prob = model.predict(input_data_scaled)[0][0]


if(prediction_prob)>0.5:
    st.write("yes likely churn")
else:
    st.write("no likely not churn")
