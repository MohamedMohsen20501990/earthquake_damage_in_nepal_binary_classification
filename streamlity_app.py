import streamlit as st
import requests


# FastAPI endpoint
API_URL = "http://0.0.0.0:8000/predict"


st.set_page_config(
    page_title="Earthquake Damage Prediction",
    page_icon="🏠",
    layout="centered"
)

st.title("🏠 Earthquake in Nepal Building Damage Prediction")
st.write("Enter the building information to predict whether it has severe damage.")


# -----------------------------
# Building information
# -----------------------------

building_id = st.number_input(
    "Building ID",
    min_value=1,
    value=12312,
    step=1
)

age_building = st.number_input(
    "Building Age",
    min_value=1,
    max_value=200,
    value=20,
    step=1
)

plinth_area_sq_ft = st.number_input(
    "Plinth Area (sq ft)",
    min_value=0.1,
    value=280.0
)

height_ft_pre_eq = st.number_input(
    "Building Height (ft)",
    min_value=0.0,
    value=12.0
)


# -----------------------------
# Categorical features
# -----------------------------

land_surface_condition = st.selectbox(
    "Land Surface Condition",
    [
        "Flat",
        "Moderate slope",
        "Steep slope"
    ]
)

foundation_type = st.selectbox(
    "Foundation Type",
    [
        "Other",
        "Mud mortar-Stone/Brick",
        "Cement-Stone/Brick",
        "Bamboo/Timber",
        "RC"
    ]
)

roof_type = st.selectbox(
    "Roof Type",
    [
        "Bamboo/Timber-Light roof",
        "Bamboo/Timber-Heavy roof",
        "RCC/RB/RBC"
    ]
)

ground_floor_type = st.selectbox(
    "Ground Floor Type",
    [
        "Mud",
        "Brick/Stone",
        "RC",
        "Timber",
        "Other"
    ]
)

other_floor_type = st.selectbox(
    "Other Floor Type",
    [
        "Not applicable",
        "TImber/Bamboo-Mud",
        "Timber-Planck",
        "RCC/RB/RBC"
    ]
)

position = st.selectbox(
    "Position",
    [
        "Not attached",
        "Attached-1 side",
        "Attached-2 side",
        "Attached-3 side"
    ]
)

plan_configuration = st.selectbox(
    "Plan Configuration",
    [
        "Rectangular",
        "L-shape",
        "Square",
        "T-shape",
        "Multi-projected",
        "H-shape",
        "U-shape",
        "Others",
        "E-shape",
        "Building with Central Courtyard"
    ]
)


# -----------------------------
# Prediction
# -----------------------------

if st.button("🔮 Predict Damage", use_container_width=True):

    data = {
        "building_id": building_id,
        "age_building": age_building,
        "plinth_area_sq_ft": plinth_area_sq_ft,
        "height_ft_pre_eq": height_ft_pre_eq,
        "land_surface_condition": land_surface_condition,
        "foundation_type": foundation_type,
        "roof_type": roof_type,
        "ground_floor_type": ground_floor_type,
        "other_floor_type": other_floor_type,
        "position": position,
        "plan_configuration": plan_configuration
    }

    try:
        response = requests.post(
            API_URL,
            json=data,
            timeout=30
        )

        if response.status_code == 200:

            result = response.json()

            if result["success"]:

                prediction = result["prediction"]

                st.success("Prediction completed successfully!")

                st.write(f"### Building ID: {result['building_id']}")

                if prediction == 1:
                    st.error("⚠️ Severe Damage Predicted")
                else:
                    st.success("✅ No Severe Damage Predicted")

            else:
                st.error(result["message"])

        else:
            st.error(
                f"API Error ({response.status_code}): "
                f"{response.text}"
            )

    except requests.exceptions.ConnectionError:
        st.error(
            "Could not connect to the FastAPI server. "
            "Make sure FastAPI is running on port 8000."
        )

    except requests.exceptions.Timeout:
        st.error("The request timed out.")

    except Exception as e:
        st.error(f"Unexpected error: {e}")
