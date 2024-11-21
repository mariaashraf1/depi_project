import streamlit as st
import joblib
import pandas as pd
import pickle
import json

model = joblib.load('customer_default_model.pkl')

with open("features.json", "r") as f:
    feature_names = json.load(f)
    
with open("feature_names1.json", "r") as f:
    feature_names1 = json.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('column_transformer.pkl', 'rb') as f:
    column_transformer = pickle.load(f)

useless_features = ['FLAG_MOBIL', 'FLAG_EMP_PHONE', 'FLAG_WORK_PHONE', 'FLAG_CONT_MOBILE',
                   'FLAG_PHONE', 'FLAG_EMAIL', 'FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_5',
                   'FLAG_DOCUMENT_7', 'FLAG_DOCUMENT_8', 'FLAG_DOCUMENT_9', 'FLAG_DOCUMENT_10',
                   'FLAG_DOCUMENT_13', 'FLAG_DOCUMENT_15', 'FLAG_DOCUMENT_16', 'FLAG_DOCUMENT_17',
                   'FLAG_DOCUMENT_18', 'FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_21', 'EXT_SOURCE_1',
                   'EXT_SOURCE_2', 'EXT_SOURCE_3', 'FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_11',
                   'FLAG_DOCUMENT_12', 'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_21',]  

# Streamlit application
st.title("Customer Default Prediction App")

# Provide example input for users (Example 1 and Example 2)
example_1 = ['Cash loans', 'M', 'N', 'Y', 0.0, 202500.0, 406597.5, 24700.5, 351000.0, 'Unaccompanied', 'Working', 'Secondary / secondary special', 
             'Single / not married', 'House / apartment', 0.018801, -9461.0, -637.0, -3648.0, -2120.0, 0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 'Laborers', 1.0, 2.0, 2.0,
             'WEDNESDAY', 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 'Business Entity Type 3', 0.0830369673913225, 0.2629485927471776, 0.1393757800997895, 0.0247, 0.0369,
             0.9722, 0.6192, 0.0143, 0.0, 0.069, 0.0833, 0.125, 0.0369, 0.0202, 0.019, 0.0, 0.0, 0.0252, 0.0383, 0.9722, 0.6341, 0.0144, 0.0, 0.069, 0.0833, 0.125, 
             0.0377, 0.022, 0.0198, 0.0, 0.0, 0.025, 0.0369, 0.9722, 0.6243, 0.0144, 0.0, 0.069, 0.0833, 0.125, 0.0375, 0.0205, 0.0193, 0.0, 0.0, 'reg oper account', 
             'block of flats', 0.0149, 'Stone, brick', 'No', 2.0, 2.0, 2.0, 2.0, -1134.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]

example_2 = ['Cash loans', 'F', 'N', 'N', 0.0, 270000.0, 1293502.5, 35698.5, 1129500.0, 'Family', 'State servant', 'Higher education', 'Married', 'House / apartment', 0.0035409999999999, 
             -16765.0, -1188.0, -1186.0, -291.0,0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 'Core staff', 2.0, 1.0, 1.0, 'MONDAY', 11.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
             'School', 0.3112673113812225, 0.6222457752555098, 0, 0.0959, 0.0529, 0.9851, 0.7959999999999999, 0.0605, 0.08, 0.0345, 0.2917, 0.3333, 0.013, 0.0773,
             0.0549, 0.0039, 0.0098, 0.0924, 0.0538, 0.9851, 0.804, 0.0497, 0.0806, 0.0345, 0.2917, 0.3333, 0.0128, 0.079, 0.0554, 0.0, 0.0, 0.0968, 0.0529,
             0.9851, 0.7987, 0.0608, 0.08, 0.0345, 0.2917, 0.3333, 0.0132, 0.0787, 0.0558, 0.0039, 0.01, 'reg oper account', 'block of flats', 0.0714, 'Block',
             'No', 1.0, 0.0, 1.0, 0.0, -828.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
             0.0, 0.0, 0.0]

st.header("Enter Features or Use an Example:")
input_choice = st.selectbox("Choose input type:", ["Manual Entry", "User 1", "User 2"])

if input_choice == "Manual Entry":
    input_string = st.text_area("Enter features (comma-separated)", "")
    input_list = [x.strip() for x in input_string.split(",")] if input_string else []
elif input_choice == "User 1":
    input_list = example_1
elif input_choice == "User 2":
    input_list = example_2

# Check if the correct number of features is provided
if len(input_list) != 120:
    st.error(f"Please provide exactly 120 values. You entered {len(input_list)} values.")
else:
    # Convert input to DataFrame with the correct feature names
    input_data = pd.DataFrame([input_list], columns=feature_names)

    st.subheader("Input Features and Corresponding Values (Before Transformation):")
    feature_dict = input_data.iloc[0].to_dict()  
    st.write(feature_dict)

    # Step 1: Drop the useless features
    input_data = input_data.drop(columns=useless_features)  

    # Step 2: Apply scaling on numerical columns
    numerical_columns = input_data.select_dtypes(include=['float64', 'int64']).columns
    input_data[numerical_columns] = scaler.transform(input_data[numerical_columns])

    # Step 3: Apply encoding using the ColumnTransformer
    input_data_encoded = column_transformer.transform(input_data)
    
    input_data_final = pd.DataFrame(input_data_encoded, columns=feature_names1)

    # Step 4: Make prediction when the button is clicked
    if st.button("Predict"):
        prediction = model.predict(input_data_final)
        result = "Defaulter" if prediction[0] == 1 else "Non-Defaulter"
        st.success(f"Prediction: {result}")
