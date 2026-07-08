import streamlit as st
import joblib
import plotly.express as px
import pandas as pd
from reportlab.pdfgen import canvas
import io

# ======================
# LOAD MODELS
# ======================


upv_model = joblib.load("models/upv_model.pkl")

sonreb_model = joblib.load("models/sonreb_model.pkl")

advanced_model = joblib.load("models/advanced_model.pkl")

# ======================
# PAGE CONFIG
# ======================

st.set_page_config(
    page_title="Statistical Based Son-Reb Strenght Predictable Model",
    layout="wide"
)

# ======================
# SIDEBAR
# ======================

page = st.sidebar.selectbox(
    "Navigation",
    [
        "🏠Home",
        "📈UPV Only",
        "🔨SonReB",
        "🧱Material Aware",
        "Random Forest Prediction",
        "📊 Analytics",
        "ℹ️ About"
    ]
)
# ======================
# HOME
# ======================

if page == "🏠Home":
    col1, col2 = st.columns([1,2])
    with col1:
        st.image(
            "assets/logo.png",
            width=280
        )
    with col2:
        st.title(
        "Statistical-Based Son-ReB Strength Prediction Model"
        )
    st.markdown("""
### AI Based Strength Prediction of Concrete
This system predicts concrete compressive strength
structural health using NDT and AI.
    """)
    st.title(
        "🏗 AI Structural Health Assessment System"
    )

    st.markdown("""
    ### AI Based Strength Prediction of Concrete

This system predicts concrete compressive strength
    using Ultrasonic Pulse Velocity (UPV),
    Rebound Hammer Test,
    Recycled Coarse Aggregate (RCA),
    and Machine Learning algorithms,
    Random Forest AI
    """)
    st.markdown("---")

    # ======================
    # PROJECT METRICS
    # ======================

    col1,col2,col3,col4 = st.columns(4)

    with col1:
        st.metric(
            "R² Score",
            "0.951"
        )

    with col2:
        st.metric(
            "RMSE",
            "2.08 MPa"
        )

    with col3:
        st.metric(
            "Models",
            "3"
        )
    with col4:
        st.metric(
            "Features",
            "3"
        )


    st.markdown("---")

    # ======================
    # PROJECT FEATURES
    # ======================

    st.subheader("Project Highlights")

    st.success(
        "✓ Random Forest Machine Learning Model"
    )

    st.success(
        "✓ SonReB Strength Prediction"
    )

    st.success(
        "✓ Structural Health Index (SHI)"
    )

    st.success(
        "✓ RCA Concrete Assessment"
    )

    st.markdown("---")

    # ======================
    # WORKFLOW
    # ======================

    st.subheader("System Workflow")

    st.info("""
    UPV + Rebound + RCA

                ↓

       AI Prediction Model

                ↓
      Strength Prediction

                ↓

      Grade Classification

                ↓

    Structural Health Index
    """)
# ======================
# UPV ONLY
# ======================

elif page == "📈UPV Only":

    st.title("UPV Based Strength Prediction")

    upv = st.number_input(
        "Enter UPV (km/s)",
        min_value=0.0,
        max_value=10.0,
        value=4.0
    )

    if st.button("Predict Strength"):

        strength = upv_model.predict([[upv]])[0]

        st.success(
            f"Predicted Strength = {strength:.2f} MPa"
        )

        if strength >= 40:
            grade = "M40"

        elif strength >= 30:
            grade = "M30"

        else:
            grade = "M20"

        st.info(
            f"Estimated Grade = {grade}"
        )

    st.subheader("UPV-Strength Equation")

    st.latex(
        r"f_c=-48.913+20.467V"
    )
# ======================
# SONREB
# ======================

elif page == "🔨SonReB":

    st.title("SonReB Strength Prediction")

    upv = st.number_input(
        "Enter UPV (km/s)",
        min_value=0.0,
        max_value=10.0,
        value=4.0,
        key="sonreb_upv"
    )

    rebound = st.number_input(
        "Enter Rebound Number",
        min_value=0.0,
        max_value=100.0,
        value=30.0,
        key="sonreb_rebound"
    )

    if st.button("Predict Strength", key="sonreb_button"):

        strength = sonreb_model.predict(
            [[upv,rebound]]
        )[0]

        st.success(
            f"Predicted Strength = {strength:.2f} MPa"
        )

        if strength >= 40:
            grade = "M40"

        elif strength >= 30:
            grade = "M30"

        else:
            grade = "M20"

        st.info(
            f"Estimated Grade = {grade}"
        )

    st.subheader("SonReB Equation")

    st.latex(
        r"f_c=-13.827+5.561V+0.772R"
    )
# ======================
# MATERIAL AWARE
# ======================

elif page == "🧱Material Aware":

    st.title(
        "Material Aware Strength Prediction"
    )

    upv = st.number_input(
        "UPV (km/s)",
        value=4.0,
        key="ma_upv"
    )

    rca = st.number_input(
        "RCA (%)",
        value=10.0,
        key="ma_rca"
    )

    wa = st.number_input(
        "Water Absorption (%)",
        value=1.2,
        key="ma_wa"
    )

    aiv = st.number_input(
        "Aggregate Impact Value (%)",
        value=20.0,
        key="ma_aiv"
    )

    acv = st.number_input(
        "Aggregate Crushing Value (%)",
        value=24.0,
        key="ma_acv"
    )

    sg = st.number_input(
        "Specific Gravity",
        value=2.8,
        key="ma_sg"
    )

    if st.button(
        "Predict Strength",
        key="material_button"
    ):

        strength = (
            -84.482
            + 30.895*upv
            + 0.834*rca
            + 0.033*wa
            - 0.694*aiv
            - 0.694*acv
            + 5.521*sg
        )

        st.success(
            f"Predicted Strength = {strength:.2f} MPa"
        )

        if strength >= 40:
            grade = "M40"

        elif strength >= 30:
            grade = "M30"

        else:
            grade = "M20"

        st.info(
            f"Estimated Grade = {grade}"
        )

    st.subheader(
        "UPV–RCA–Material Characteristics Equation"
    )

    st.latex(
r"f_c=-84.482+30.895V+0.834A+0.033W-0.694I-0.694C+5.521G"
    )
# ======================
# ADVANCED AI
# ======================

elif page == "Random Forest Prediction":

    st.title(
        "Advanced  Strength Prediction"
    )

    upv = st.number_input(
        "UPV (km/s)",
        value=4.0,
        key="adv_upv"
    )

    rebound = st.number_input(
        "Rebound Number",
        value=30.0,
        key="adv_rebound"
    )

    rca = st.number_input(
        "RCA (%)",
        value=10.0,
        key="adv_rca"
    )

    if st.button(
        "Predict Strength",
        key="advanced_button"
    ):

        strength = advanced_model.predict(
            [[upv,rebound,rca]]
        )[0]

        st.success(
            f"Predicted Strength = {strength:.2f} MPa"
        )

        # ==================
        # GRADE
        # ==================

        if strength >= 40:
            grade = "M40"

        elif strength >= 30:
            grade = "M30"

        else:
            grade = "M20"

        st.info(
            f"Estimated Grade = {grade}"
        )

        # ==================
        # SHI
        # ==================

        shi = min(
            100,
            (strength/40)*100
        )

        st.metric(
            "Structural Health Index",
            f"{shi:.1f}%"
        )

        # ==================
        # CONDITION
        # ==================

        if shi >= 80:
            condition = "Healthy"

        elif shi >= 60:
            condition = "Moderate"

        else:
            condition = "Poor"

        st.warning(
            f"Condition = {condition}"
        )
# ======================
# MODEL ANALYSIS
# ======================

elif page == "📊 Analytics":

    st.title("📊 Analytics Dashboard")

    # Performance Metrics

    col1,col2,col3 = st.columns(3)

    with col1:
        st.metric("R² Score","0.951")

    with col2:
        st.metric("MAE","1.738 MPa")

    with col3:
        st.metric("RMSE","2.077 MPa")

    st.markdown("---")

    # Graphs Side by Side

    col1,col2 = st.columns(2)

    with col1:
        st.subheader("Correlation Matrix")
        st.image(
            "assets/heatmap.png",
            use_container_width=True
        )

    with col2:
        st.subheader("Actual vs Predicted")
        st.image(
            "assets/actual_vs_predicted.png",
            use_container_width=True
        )

    st.markdown("---")

    # Correlation Values

    st.subheader("Correlation Summary")

    st.info("""
    UPV vs Strength = 0.819

    Rebound vs Strength = 0.953

    RCA vs Strength = -0.231
    """)

    st.markdown("---")

    # Feature Importance

    st.subheader("Feature Importance")

    import pandas as pd
    import plotly.express as px

    feature_df = pd.DataFrame({
        "Feature":["UPV","Rebound","RCA"],
        "Importance":[13.24,84.55,2.21]
    })

    fig = px.bar(
        feature_df,
        x="Feature",
        y="Importance",
        text="Importance",
        color="Feature",
        title="Feature Contribution (%)"
    )

    fig.update_traces(
        textposition="outside"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )
elif page == "ℹ️ About":

    st.title(
        "ℹ️ About AI-SHAS"
    )

    st.image(
        "assets/logo.png",
        width=250
    )

    st.markdown("""
    ## AI Structural Health Assessment System

    AI-SHAS is an intelligent decision-support system
    developed for predicting concrete compressive strength
    using Non-Destructive Testing (NDT) techniques and
    Machine Learning.

    ### Inputs

    - Ultrasonic Pulse Velocity (UPV)
    - Rebound Hammer Number
    - RCA Percentage
    - Material Characteristics

    ### AI Models

    - Linear Regression
    - SonReB Model
    - Random Forest Regressor

    ### Outputs

    - Predicted Compressive Strength
    - Concrete Grade
    - Structural Health Index (SHI)
    - Structural Condition Assessment

    ### Dataset

    - Laboratory Tested Samples
    - M20, M30, M40 Concrete
    - RCA Replacement Levels:
      0%, 10%, 20%, 30%

    ### Developed By

    Akash Ganguly

    B.Tech Civil Engineering

    Final Year Project
    """)
st.markdown("---")

st.caption(
"""
AI Structural Health Assessment System (AI-SHAS)

Developed for Concrete Strength Prediction and Structural Health Evaluation

### Developer

Akash Ganguly

B.Tech Civil Engineering


&

Utpal Kumar Singh 

B.Tech Civil Engineering

Under the Guidance of 

Dr. Sanjay Sengupta

HOD Of Civil Engineering Department 

BCREC

© 2026
"""
)
