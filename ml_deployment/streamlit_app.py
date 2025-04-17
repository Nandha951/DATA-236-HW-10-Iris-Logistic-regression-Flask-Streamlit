import streamlit as st
import pickle

# Load the model
try:
    model = pickle.load(open('model.pkl', 'rb'))
except Exception as e:
    st.error(f"Error loading the model: {e}")
    model = None

st.title("Iris Prediction App")

st.write("This app predicts the species of Iris flower based on the following features:")
st.write("- Sepal Length (cm)")
st.write("- Sepal Width (cm)")
st.write("- Petal Length (cm)")
st.write("- Petal Width (cm)")

# Input fields
sepal_length = st.number_input("Sepal Length", value=5.1, help="Example: 5.1")
sepal_width = st.number_input("Sepal Width", value=3.5, help="Example: 3.5")
petal_length = st.number_input("Petal Length", value=1.4, help="Example: 1.4")
petal_width = st.number_input("Petal Width", value=0.2, help="Example: 0.2")

# Prediction button
if st.button("Predict"):
    if model is not None:
        # Make prediction
        input_array = [[sepal_length, sepal_width, petal_length, petal_width]]
        prediction = model.predict(input_array)[0]

        # Map the prediction to the species name
        species_mapping = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
        species_name = species_mapping.get(prediction, 'unknown')

        # Display the prediction
        st.success(f"The prediction is: {species_name}")
    else:
        st.error("Model not loaded. Please check the model file.")

# Add styling
st.markdown("""
    <style>
    .stApp {
        background-color: #f0f2f6;
    }
    </style>
    """, unsafe_allow_html=True)

# Add image
st.image("https://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg", width=200)