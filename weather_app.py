import streamlit as st
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Load the model
model = load_model('model.h5')

# Define the Streamlit app
def main():
    st.title("Indian Rainfall Prediction")

    # Input fields for user inputs
    year = st.number_input("Year", min_value=1901, max_value=2015)
    month = st.selectbox("Month", ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"])

    # Mapping months to integers
    month_map = {'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6, 'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12}
    month_int = month_map[month]

    # Make predictions
    if st.button("Predict"):
        # Preprocess user inputs
        data = np.array([[year, month_int]])  # Adjust as needed for your model
        data = np.reshape(data, (1, 1, data.shape[1]))

        # Load the scaler used in training
        scaler = MinMaxScaler(feature_range=(0, 1))
        # Note: You should ideally save and load the scaler as well if possible.
        data = scaler.fit_transform(data.reshape(-1, 1))

        # Make prediction using the loaded model
        predicted_rainfall = scaler.inverse_transform(model.predict(data))[0][0]

        # Display the predicted rainfall
        st.write("Predicted Rainfall for", year, month, ":", predicted_rainfall)

if __name__ == '__main__':
    main()