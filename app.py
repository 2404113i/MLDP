import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import streamlit as st  
import joblib

# Load the dataset
model = joblib.load('model_lr_hdb.pkl')

#streamlit run app.py
st.title('House Price Prediction')

#Define the input fields
towns = ['Bedok', 'Punggol','Tampines']
flat_types = ['2-room', '3-room', '4-room', '5-room']
storey_ranges = ['01 to 03', '04 to 06', '07 to 09']

## user inputs
town_selected = st.selectbox('Select Town', towns)
flat_type_selected = st.selectbox('Select Flat Type', flat_types)
storey_range_selected = st.selectbox('Select Storey Range', storey_ranges)
floor_area = st.slider('Select Floor Area (sqm)', min_value=30, max_value=30, value=70)

##PRredict the price
if st.button("Predict Price"):

    # Prepare the input data
    input_data = pd.DataFrame({
        'town': [town_selected],
        'flat_type': [flat_type_selected],
        'storey_range': [storey_range_selected],
        'floor_area_sqm': [floor_area]
    })

    ##Convert user input to a DataFrame
    df_input = pd.DataFrame({
        'town': [town_selected],
        'flat_type': [flat_type_selected],
        'storey_range': [storey_range_selected],
        'floor_area_sqm': [floor_area]
    })

    ## One-hot encode 
    df_input = pd.get_dummies(df_input, columns=['town', 'flat_type', 'storey_range'])

    ##df_input =df_input.to_numpy()
    df_input = df_input.reindex(columns=model.feature_names_in_, fill_value=0)

    
    # Make the prediction
    y_unseen_pred = model.predict(df_input)[0]  
    st.success(f"The predicted price for the selected house is: ${y_unseen_pred:,.2f}")  

st.markdown(
    f"""
    <style>
        .stApp {{
            background-color: black;
            background-image: url('https://images.unsplash.com/photo-1506748686214-e9df14d4d9d0?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=1350&q=80')
            background-size: cover;
        }}
        </style>
        """,
    unsafe_allow_html=True
)