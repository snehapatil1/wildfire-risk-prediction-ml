import streamlit as st
import joblib
import pandas as pd
from geopy.geocoders import Nominatim
import folium
from streamlit_folium import folium_static
import config

# ----------------------
# Load Models & Encoders
# ----------------------
@st.cache_resource
def load_artifacts():
    """
        Load all pre-trained ML artifacts including model, scaler, and encoders.

        Returns:
            dict: A dictionary containing model, pipeline, scaler, and encoders.
    """
    return {
        "model": joblib.load('artifacts/wildfire_model.pkl'),
        "scaler": joblib.load('artifacts/scaler.pkl'),
        "cause_encoder": joblib.load('artifacts/cause_encoder.pkl'),
        "state_encoder": joblib.load('artifacts/state_encoder.pkl'),
    }

artifacts = load_artifacts()

# ---------------------
# Utility Functions
# ---------------------
def get_location_coordinates(state: str, city: str) -> tuple:
    """
        Get latitude and longitude for a given city and state using geopy.

        Args:
            state (str): US state name.
            city (str): City name within the state.

        Returns:
            tuple: (latitude, longitude) if found, else (None, None).
    """
    geolocator = Nominatim(user_agent="wildfire_predictor")
    query = f"{city}, {state}" if city else state
    location = geolocator.geocode(query)
    return (location.latitude, location.longitude) if location else (None, None)

def preprocess_input(state: str, lat: float, lon: float, cause: str) -> pd.DataFrame:
    """
        Preprocess user input into a DataFrame suitable for model prediction.

        Args:
            state (str): Full name of the state.
            lat (float): Latitude of the location.
            lon (float): Longitude of the location.
            cause (str): Selected cause of fire.

        Returns:
            pd.DataFrame: Processed input ready for model prediction.
    """
    state_abbr = config.us_states_abbr[state]
    input_df = pd.DataFrame([{
        "STATE": state_abbr,
        "LATITUDE": lat,
        "LONGITUDE": lon,
        "STAT_CAUSE_DESCR": cause
    }])
    input_df['STATE'] = artifacts["state_encoder"].transform(input_df['STATE'])
    input_df['STAT_CAUSE_DESCR'] = artifacts["cause_encoder"].transform(input_df['STAT_CAUSE_DESCR'])
    input_df[['LATITUDE', 'LONGITUDE']] = artifacts["scaler"].transform(input_df[['LATITUDE', 'LONGITUDE']])
    return input_df

def predict_risk(input_df: pd.DataFrame) -> str:
    """
        Predict wildfire risk level using the trained model.

        Args:
            input_df (pd.DataFrame): Preprocessed input data.

        Returns:
            str: 'High' or 'Low' wildfire risk.
    """
    model = artifacts["model"]
    low_risk_index = list(model.classes_).index('Low')
    proba = model.predict_proba(input_df)[0][low_risk_index]
    return 'High' if proba > 0.97 else 'Low'

def display_map(lat: float, lon: float):
    """
        Display a Folium map with a marker for the predicted wildfire location.

        Args:
            lat (float): Latitude of the location.
            lon (float): Longitude of the location.
    """
    us_map = folium.Map(location=[lat, lon], zoom_start=6)
    folium.Marker([lat, lon], popup="Wildfire Risk", icon=folium.Icon(color='red')).add_to(us_map)
    st.subheader("Location Prediction on US Map")
    folium_static(us_map)

# ---------------------
# Streamlit UI Layout
# ---------------------
def main():
    """
        Main function to control Streamlit UI and trigger prediction logic.
    """
    st.title("Wildfire Risk Predictor")
    st.markdown(
        "Select a US State and a City. Choose a likely cause of fire "
        "(default: Debris Burning, the most frequent cause)."
    )

    state = st.selectbox("Select State", sorted(config.us_states.keys()))
    city = st.selectbox("Select City", config.us_states[state])
    default_cause = 'Debris Burning'
    cause = st.selectbox("Select a Cause", config.fire_causes, index=config.fire_causes.index(default_cause))

    if st.button("Predict Wildfire Risk"):
        try:
            lat, lon = get_location_coordinates(state, city)

            if lat is not None and lon is not None:
                input_df = preprocess_input(state, lat, lon, cause)
                prediction = predict_risk(input_df)
                result = config.fire_risk[prediction]
                st.success(f"Prediction for {city}, {state}: {result}")
                display_map(lat, lon)
            else:
                st.error("Location not found. Try a more specific city.")

        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")

# ---------------------
# Entry Point
# ---------------------
if __name__ == "__main__":
    main()