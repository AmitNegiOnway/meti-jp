import streamlit as st
import requests





API_URL = "http://127.0.0.1:8000/predicts"
st.title('Lung cap regression Prediction')
st.markdown('Enter your detail below')


Age=st.number_input("Age",min_value=1,max_value=119,value=20)
Height=st.number_input("height (cm)",min_value=1.0,value=65.5)
Smoke=st.selectbox('are you smoker ?.',options=['yes','no'])
Gender=st.selectbox('whats your gender.',options=['male','female'])

Caesarean=st.selectbox('you born to Caesarean?.',options=['yes','no'])

if st.button('prediction lung cap'):
    input_data={
        'Age':Age,
        'Height':Height,
        'Smoke':Smoke,
        'Gender':Gender,
        'Caesarean':Caesarean
    }
    try:
        response=requests.post(API_URL,json=input_data)
        if response.status_code==200:
            result=response.json()
            st.success(f"Predicted Lung Cap: **{result['Lung Cap Prediction']}**")
            
        else:
            st.error(f"API Error:{response.status_code} - {response.text})")

    except requests.exceptions.ConnectionError:
        st.error("Could not connect to the FastAPI server. Make sure it's running on port 8000.")










