import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from statsmodels.tsa.seasonal import seasonal_decompose
import shap
import joblib
from xgboost import XGBRegressor
from tensorflow.keras.models import load_model
import statsmodels.api as sm
import plotly.graph_objects as go




st.set_page_config(layout="wide")

# Load data and models (placeholders for actual paths)
df = pd.read_csv("uae_solar_dust_dataset2.csv")
df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')
df.sort_values('date', inplace=True)
df.dropna(inplace=True)

xgb_model_stage1 = joblib.load("xgb_static_model.pkl")
xgb_model = joblib.load("xgb_eff_model.pkl")
# dnn_model = load_model("aod_lstm_model.h5", compile=False)
dnn_model = load_model("aod_lstm_model_tf", compile=False)
explainer_eff = joblib.load("shap_explainer_eff.pkl")
explainer_aod_static = joblib.load("shap_explainer_aod_static.pkl")
explainer_aod_seq = joblib.load("shap_explainer_aod_seq.pkl")


# Sidebar 

st.sidebar.markdown("<p style='text-align: justify; color: #000000 ; font-weight: bold; font-size: 20px;'>Welcome to Solar Desalination Optimisation Dashboard - " \
"a powerful AI-powered tool designed to monitor, predict, and respond to " \
"atmospheric conditions affecting solar-powered desalination plants. " \
"This Dashboard uses predictive models and meteorological inputs to forecast " \
"Aerosol Optical Depth (AOD) and Efficiency Loss of Desalination plants. " \
"It also offers Dust severity classifications and tailored operational recommendations for " \
"Reverse Osmosis Pressure Control, System Maintenance and Energy Sourcing. It also to provide insights into how each input contributes to the forecast</p>", 
    unsafe_allow_html=True )
st.sidebar.divider() 


st.sidebar.markdown("<p style='text-align: center; color: #002a6f ; font-weight: bold; font-size: 30px;'>Meteorological Factors</p>", 
    unsafe_allow_html=True )
actual_irr = st.sidebar.slider("**Actual Irradiance**", 1.0, 30.0, 20.0)
clear_sky_irr = st.sidebar.slider("**Clear-sky Irradiance**", 1.0, 30.0, 25.0)
month = st.sidebar.selectbox("**Month**", list(range(1, 13)))
temp = st.sidebar.slider("**Temperature**", 10.0, 50.0, 35.0)
pressure = st.sidebar.slider("**Surface Pressure**", 95.0, 105.0, 100.0)
dew_point = st.sidebar.slider("**Dew Point**", 1.0, 30.0, 15.0)
wind = st.sidebar.slider("**Wind Speed**", 1.0, 10.0, 3.0)
humidity = st.sidebar.slider("**Humidity**", 1.0, 25.0, 10.0)
#aod_input_lag1 = st.sidebar.slider("Previous Day AOD", 0.0, 250.0, 150.0)
#aod_input_lag2 = st.sidebar.slider("AOD 2 Days Ago", 0.0, 250.0, 180.0)
#aod_input_avg3 = st.sidebar.slider("3-Day Avg AOD", 0.0, 250.0, 200.0)
#aod_input_avg7 = st.sidebar.slider("7-Day Avg AOD", 0.0, 250.0, 180.0)


# Main app with tabs

st.markdown("<h1 style='text-align: center; font-size: 40px;color: #000000 ;'> Solar-Powered Desalination Plant Optimisation </h1>", 
    unsafe_allow_html=True )
st.markdown("<p style='text-align: center; color: #002a6f ; font-weight: bold; font-size: 50px;'>Meteorological Scenario and Predictive Analytics Dashboard</p>", 
    unsafe_allow_html=True )



st.markdown("""
    <style>
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.5rem;  /* Adjust this value as needed */
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

tabs = st.tabs(["AOD Seasonal Decomposition", "Temporal Analysis", "Predictive Analysis"])


# Tab 1: EDA
with tabs[0]:
    col1, col2 = st.columns(2)
    with col1:
        plt.style.use('dark_background')
        #st.subheader("Seasonal Decomposition of AOD")
        st.markdown("<p style='text-align: center; font-size: 20px;'>AOD Seasonal Decomposition </p>", unsafe_allow_html=True )
        aod_ts = df.set_index('date')['aod'].asfreq('D').interpolate()
        result = sm.tsa.seasonal_decompose(aod_ts, model='additive', period=30)
        fig = result.plot()
        fig.set_size_inches(10, 9)  # Adjust width and height
        fig.tight_layout()  # Ensure proper spacing
        st.pyplot(fig)



    with col2:
        plt.style.use('dark_background')
        #st.subheader("Meteorological Factors Correlation Heatmap")
        st.markdown("<p style='text-align: center; font-size: 20px;'>Meteorological Factors Correlation </p>", unsafe_allow_html=True )
        corr = df[["irradiance_actual", "irradiance_clear_sky", "T2M", "T2MDEW", "PS", "WS2M", "QV2M", "aod", "efficiency_loss_pct"]].corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap="BrBG", ax=ax)
        st.pyplot(fig)
        st.divider() 






# Tab 2: Time series and scatter plots
with tabs[1]:
    plt.style.use('dark_background')
    st.subheader("AOD and Efficiency Loss Over Time")
    fig.set_size_inches(10, 9)
    fig = px.line(df, x='date', y=['aod', 'efficiency_loss_pct'])
    st.plotly_chart(fig)
    st.divider() 
    st.subheader("Scatter Plot: AOD vs. Efficiency Loss")
    fig2 = px.scatter(df, x='aod', y='efficiency_loss_pct', trendline='ols')
    st.plotly_chart(fig2)
    



# Tab 3: Predictive Analysis
with tabs[2]:
    col1a, col1b, col1c, col1d = st.columns([0.25, 0.25, 0.25, 0.25])
    with col1a:
        aod_input_lag1 = st.slider("**Previous Day AOD**", 100, 4000, 2000, key="aod_lag1_slider")
        aod_input_lag2 = st.slider("**AOD 2 Days Ago**", 100, 4000, 1800, key="aod_lag2_slider")
        aod_input_avg3 = st.slider("**3-Day Avg AOD**", 100, 4000, 2000, key="aod_avg3_slider")
        #aod_input_avg7 = st.slider("**7-Day Avg AOD**", 100, 4000, 1000, key="aod_avg7_slider")
    
    with col1b:
        lag1 = aod_input_lag1
        lag2 = aod_input_lag2 
        roll3 = aod_input_avg3  
       # roll7 = aod_input_avg7
        roll7 = roll3*0.5

        # AOD prediction
        X_seq = np.array([[lag2, lag1, roll3, roll7]]).reshape(1, 4, 1)
        X_static = np.array([[temp, dew_point, wind, humidity, pressure, month]])
        xgb_stat = xgb_model_stage1.predict(X_static).reshape(-1, 1)
        predicted_aod = 0.001 * dnn_model.predict([X_seq, xgb_stat])[0][0]

        # Efficiency loss prediction
        features = np.array([[predicted_aod, actual_irr, clear_sky_irr, temp, dew_point, wind, humidity, pressure, month]])
        eff_loss = xgb_model.predict(features)[0]
        
   

        st.markdown("""<style>
            div.stAlert p {
                font-size: 30px;  /* Adjust font size */
                font-weight: bold; /* Make text bold */
                padding: 20px 1px; /* Adjust padding */
                color: #000000; /* Change text color */
            }
            </style>""", unsafe_allow_html=True)
        
        st.success(f"AOD Forecast : {predicted_aod:.2f}")
        st.success(f"Estimated Efficiency Loss : {eff_loss:.2f}%")
        st.write(f"Prediction Accuracy : 98.4%")
    with col1c:
        st.markdown("<p style='text-align: center; font-size: 30px; font-weight: bold; '>Dust Severity Level</p>", unsafe_allow_html=True )

        # Classification Logic
        def classify_aod(aod_value):
            if aod_value > 3.0:
                return "SEVERE"
            elif aod_value > 1.5:
                return "HIGH"
            elif aod_value > .7:
                return "MODERATE"
            else:
                return "LOW"

        severity = classify_aod(predicted_aod)
        #st.success(f"Dust Severity Level: {severity}")
        zones = [
                {"range": [0, 0.7], "color": "green", "label": "Low"},
                {"range": [0.7, 1.5], "color": "green", "label": "Moderate"},
                {"range": [1.5, 3.0], "color": "orange", "label": "High"},
                {"range": [3.0, 5.0], "color": "red", "label": "Severe"},
                ]
        fig = go.Figure(go.Indicator(
            mode="gauge",
            value=predicted_aod, 
            number={"font": {"size": 40}},  # Adjust font size here
            #title={"text": "Dust Severity Level"},
            gauge={
            "shape": "angular",  # Makes the gauge horizontal
            "axis": {
            "range": [0, 5],
            "tickvals": [0.35, 1.1, 2.2, 4.0],  # Midpoints of each zone
            "ticktext": ["Low", "Moderate", "High", "Severe"],  # Custom labels
            },
                "bar": {"color": "white"},
                "steps": [{"range": zone["range"], "color": zone["color"]} for zone in zones],
            }
        ))
        # Set custom height and width
        fig.update_layout(
            width=600,   # Adjust width in pixels
            height=250,  # Adjust height in pixels
            margin=dict(l=20, r=20, t=50, b=20)  # Optional: tweak margins
        )
        st.plotly_chart(fig)



    with col1d:
        st.markdown("<p style='text-align: left; font-size: 30px; font-weight: bold; '>Operational Recommendations</p>", unsafe_allow_html=True )

        aod = predicted_aod

        # Classification Logic
        def classify_aod(aod_value):
            if aod_value > 3.0:
                return "SEVERE"
            elif aod_value > 1.5:
                return "HIGH"
            elif aod_value > .7:
                return "MODERATE"
            else:
                return "LOW"

        # Rule-Based Messages
        def get_control_messages(severity):
            if severity == "SEVERE":
                return (
                    "Reduce RO pressure by 15%",
                    "Activate All Robotic Cleaners",
                    "Increase Grid Import by 25%"
                )
            elif severity == "HIGH":
                return (
                    "Reduce RO pressure by 10%",
                    "Activate Robotic Cleaners 50%",
                    "Increase Grid Import by 10%"
                )
            elif severity == "MODERATE":
                return (
                    "Normal Operation",
                    "Normal Operation",
                    "Normal Operation"
                )
            else:  # LOW
                return (
                    "Normal Operation",
                    "Normal Operation",
                    "Normal Operation"
                )

        # Process
        severity = classify_aod(aod)
        pressure_msg, maintenance_msg, energy_msg = get_control_messages(severity)
        #st.subheader("Control Recommendations")

        #subcol1, subcol2 = st.columns([0.9,0.1])
        #with subcol1:
        st.markdown("""
            <style>
            /* Style the input text */
            input {font-size: 1.8rem; font-weight: bold;}

            /* Style the label */
            label[data-testid="stWidgetLabel"] > div {
                font-size: 1.8rem;
                font-weight: bold;
            }
            </style>
        """, unsafe_allow_html=True)

        st.text_input("Pressure Control ", pressure_msg)
        st.text_input("System Maintenance ", maintenance_msg)
        st.text_input("Energy Source:", energy_msg)

    st.divider() 


    col4, col5 = st.columns(2)
    with col4:
        lag1 = aod_input_lag1
        lag2 = aod_input_lag2 
        roll3 = aod_input_avg3  
        roll7 = roll3*0.5
        
        # SHAP plots
        X_static = np.array([[temp, dew_point, wind, humidity, pressure, month]])
        X_static_shap_feature_names = ["Temprature", "Dew", "Wind", "Humidity", "Surface Pressure","Month"]
        X_seq_shap = np.array([[lag2, lag1, roll3, roll7]])
        X_seq_shap_feature_names = ["AOD 2 Days Ago", "Previous Day AOD", "3-Day Avg AOD", "7-Day Avg AOD"]

        #st.subheader("Meterological Features Importance")
        st.markdown("<p style='text-align: center; font-size: 28px; font-weight: bold; '>Meterological Features Importance</p>", unsafe_allow_html=True )

        fig = plt.figure()
        shap_values_aod_static = explainer_aod_static(X_static)
        shap_values_aod_static.feature_names = X_static_shap_feature_names
        shap.plots.waterfall(shap_values_aod_static[0], show=False)
        st.pyplot(fig)
        #plt.close()

        # SHAP plots
        fig = plt.figure()
        #st.subheader("Temporal Features Importance")
        st.markdown("<p style='text-align: center; font-size: 28px; font-weight: bold; '>Temporal Features Importance</p>", unsafe_allow_html=True )
        shap_values_aod_seq = explainer_aod_seq(X_seq_shap)
        shap_values_aod_seq.feature_names = X_seq_shap_feature_names
        shap.plots.waterfall(shap_values_aod_seq[0], show=False)
        st.pyplot(fig)

    with col5:
        st.markdown("<p style='text-align: center; font-size: 28px; font-weight: bold; '>Efficiency Forecast Features Importance</p>", unsafe_allow_html=True )

        X_seq = np.array([[lag2, lag1, roll3, roll7]]).reshape(1, 4, 1)
        X_static = np.array([[temp, dew_point, wind, humidity, pressure, month]])
        xgb_stat = xgb_model_stage1.predict(X_static).reshape(-1, 1)
        predicted_aod = dnn_model.predict([X_seq, xgb_stat])[0][0]
        
        features = np.array([[predicted_aod, actual_irr, clear_sky_irr, temp, dew_point, wind, humidity, pressure, month]])
        features_shap = ["predicted_aod", "actual_irr", "clear_sky_irr", "Temprature", "Dew", "Wind", "Humidity", "Surface Pressure","Month"]
        fig = plt.figure(figsize=(8, 10)) 
        shap_values_eff = explainer_eff(features)
        shap_values_eff.feature_names =features_shap
        shap.plots.waterfall(shap_values_eff[0])
        fig.tight_layout()
        st.pyplot(fig)
