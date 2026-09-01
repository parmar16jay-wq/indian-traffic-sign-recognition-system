import streamlit as st
import tensorflow as tf
import numpy as np
import json
from pathlib import Path
from PIL import Image
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Indian Traffic Sign Recognition",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>

.main{
    background-color:#0E1117;
}

.title{
    text-align:center;
    font-size:42px;
    color:#00E5FF;
    font-weight:bold;
}

.subtitle{
    text-align:center;
    color:white;
    font-size:18px;
}

.card{
    background:#1B1F27;
    padding:20px;
    border-radius:15px;
    border:1px solid #2D3748;
}

.metric{
    background:#111827;
    padding:15px;
    border-radius:12px;
    text-align:center;
}

.footer{
    text-align:center;
    color:gray;
    padding:15px;
}

.stButton>button{
    width:100%;
    border-radius:10px;
    height:45px;
    background:#00B8D4;
    color:white;
    font-size:18px;
}

</style>
""",unsafe_allow_html=True)

# =====================================================
# PATHS
# =====================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "traffic_sign_model.keras"

CLASS_PATH = BASE_DIR/"models"/"class_names.json"

ACCURACY_GRAPH = BASE_DIR/"models"/"accuracy.png"

LOSS_GRAPH = BASE_DIR/"models"/"loss.png"

# =====================================================
# LOAD MODEL
# =====================================================

@st.cache_resource
def load_model():

    model=tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    return model

# =====================================================
# LOAD CLASS NAMES
# =====================================================

@st.cache_data
def load_classes():

    with open(CLASS_PATH,"r") as f:
        return json.load(f)

# =====================================================
# LOAD EVERYTHING
# =====================================================

model=load_model()

class_names=load_classes()

# =====================================================
# IMAGE PREPROCESSING
# =====================================================

def preprocess_image(image):

    image=image.convert("RGB")

    image=image.resize((224,224))

    img=np.array(image)

    img=np.expand_dims(img,axis=0)

    return img

# =====================================================
# PREDICTION FUNCTION
# =====================================================

def predict(image):

    img=preprocess_image(image)

    prediction=model.predict(img,verbose=0)

    predicted=np.argmax(prediction)

    confidence=float(np.max(prediction))*100

    return predicted,confidence,prediction[0]

# =====================================================
# TRAFFIC SIGN DESCRIPTIONS
# =====================================================

descriptions={

"STOP":
"Come to a complete stop before moving ahead.",

"NO_ENTRY":
"Vehicles are not allowed to enter this road.",

"SPEED_LIMIT_20":
"Maximum speed allowed is 20 km/h.",

"SPEED_LIMIT_30":
"Maximum speed allowed is 30 km/h.",

"SPEED_LIMIT_40":
"Maximum speed allowed is 40 km/h.",

"SPEED_LIMIT_50":
"Maximum speed allowed is 50 km/h.",

"SPEED_LIMIT_60":
"Maximum speed allowed is 60 km/h.",

"SPEED_LIMIT_70":
"Maximum speed allowed is 70 km/h.",

"SPEED_LIMIT_80":
"Maximum speed allowed is 80 km/h.",

"HORN_PROHIBITED":
"Blowing the horn is prohibited in this area.",

"NO_PARKING":
"Parking vehicles here is prohibited.",

"NO_STOPPING_OR_STANDING":
"Stopping or standing vehicles is prohibited.",

"LEFT_TURN_PROHIBITED":
"Vehicles cannot turn left.",

"RIGHT_TURN_PROHIBITED":
"Vehicles cannot turn right.",

"U_TURN_PROHIBITED":
"Vehicles cannot take a U-turn.",

"PEDESTRIAN_CROSSING":
"Pedestrian crossing ahead. Slow down.",

"SCHOOL_AHEAD":
"School zone ahead. Drive carefully.",

"ROUNDABOUT":
"Roundabout ahead. Follow roundabout rules.",

"GIVE_WAY":
"Give priority to other traffic.",

"TRAFFIC_SIGNAL":
"Traffic signal ahead."

}

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.image(
"https://cdn-icons-png.flaticon.com/512/3202/3202926.png",
width=90
)

st.sidebar.title("Navigation")

page=st.sidebar.radio(

"Go To",

[
"🏠 Home",
"🚦 Predict Traffic Sign",
"📈 Model Performance",
"ℹ About Project"
]

)

st.sidebar.markdown("---")

st.sidebar.success("Model Loaded Successfully")

st.sidebar.info("Classes : 85")

st.sidebar.info("Validation Accuracy : 80.61%")

st.sidebar.info("Training Accuracy : 90.12%")

# =====================================================
# HOME PAGE
# =====================================================

if page == "🏠 Home":

    st.markdown(
        "<h1 class='title'>🚦 Indian Traffic Sign Recognition System</h1>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<p class='subtitle'>Neural Network Based Traffic Sign Classification using TensorFlow & MobileNetV2</p>",
        unsafe_allow_html=True,
    )

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("🚦 Classes", "85")

    with col2:
        st.metric("🎯 Validation Accuracy", "80.61%")

    with col3:
        st.metric("🧠 Model", "MobileNetV2")

    st.markdown("---")

    left, right = st.columns([2, 1])

    with left:

        st.markdown("## 📖 Project Overview")

        st.write("""
This project recognizes Indian Traffic Signs using a Convolutional Neural Network
based on **MobileNetV2 Transfer Learning**.

The system can:

- Upload traffic sign images
- Predict the traffic sign
- Show confidence score
- Display Top-5 predictions
- Show model performance

This project has been developed using:

- Python
- TensorFlow
- Streamlit
- MobileNetV2
- Transfer Learning
        """)

    with right:

        st.success("✔ Model Loaded")

        st.info("Dataset Images : 4438")

        st.info("Training Images : 3551")

        st.info("Validation Images : 887")

        st.info("Epochs : 20")

        st.info("Optimizer : Adam")

        st.info("Loss : Sparse Categorical Crossentropy")

# =====================================================
# PREDICTION PAGE
# =====================================================

elif page == "🚦 Predict Traffic Sign":

    st.markdown(
        "<h1 class='title'>🚦 Predict Traffic Sign</h1>",
        unsafe_allow_html=True,
    )

    st.write("Upload a traffic sign image below.")

    uploaded_file = st.file_uploader(
        "Choose Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file)

        col1, col2 = st.columns(2)

        with col1:

            st.image(
                image,
                caption="Uploaded Image",
                use_container_width=True
            )

        with st.spinner("Predicting..."):

            predicted_class, confidence, probabilities = predict(image)

            sign_name = class_names[predicted_class]

        with col2:

            st.success("Prediction Completed")

            st.metric(
                "Traffic Sign",
                sign_name
            )

            st.metric(
                "Confidence",
                f"{confidence:.2f}%"
            )

            st.progress(min(int(confidence), 100))

            description = descriptions.get(
                sign_name,
                "No description available."
            )

            st.info(description)

        st.markdown("---")

        st.subheader("Top 5 Predictions")

        top5 = np.argsort(probabilities)[::-1][:5]

        labels = [class_names[i] for i in top5]

        scores = [float(probabilities[i]) * 100 for i in top5]

        df = pd.DataFrame({

            "Traffic Sign": labels,

            "Confidence": scores

        })

        fig = px.bar(

            df,

            x="Confidence",

            y="Traffic Sign",

            orientation="h",

            text="Confidence",

            title="Top 5 Predictions"

        )

        fig.update_layout(

            height=450,

            yaxis=dict(autorange="reversed")

        )

        st.plotly_chart(

            fig,

            use_container_width=True

        )

    else:

        st.info("Please upload a traffic sign image.")

        # =====================================================
# MODEL PERFORMANCE PAGE
# =====================================================

elif page == "📈 Model Performance":

    st.markdown(
        "<h1 class='title'>📈 Model Performance Dashboard</h1>",
        unsafe_allow_html=True,
    )

    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Training Accuracy", "90.12%")
    col2.metric("Validation Accuracy", "80.61%")
    col3.metric("Classes", "85")
    col4.metric("Epochs", "20")

    st.markdown("---")

    left, right = st.columns(2)

    with left:

        st.subheader("📈 Training Accuracy")

        if ACCURACY_GRAPH.exists():
            st.image(
                str(ACCURACY_GRAPH),
                use_container_width=True
            )
        else:
            st.warning("Accuracy graph not found.")

    with right:

        st.subheader("📉 Training Loss")

        if LOSS_GRAPH.exists():
            st.image(
                str(LOSS_GRAPH),
                use_container_width=True
            )
        else:
            st.warning("Loss graph not found.")

    st.markdown("---")

    st.subheader("🧠 Model Information")

    info = {
        "Model": "MobileNetV2",
        "Framework": "TensorFlow",
        "Programming Language": "Python",
        "Frontend": "Streamlit",
        "Dataset": "Indian Traffic Sign Dataset",
        "Transfer Learning": "Yes",
        "Image Size": "224 × 224",
        "Optimizer": "Adam",
        "Loss Function": "Sparse Categorical Crossentropy",
        "Activation": "Softmax",
        "Training Images": "3551",
        "Validation Images": "887",
        "Total Images": "4438",
        "Classes": "85"
    }

    df = pd.DataFrame(
        info.items(),
        columns=["Property", "Value"]
    )

    st.table(df)

    st.markdown("---")

    st.subheader("🎯 Model Summary")

    st.success("""
✔ Transfer Learning using MobileNetV2

✔ TensorFlow Deep Learning Framework

✔ 85 Indian Traffic Sign Classes

✔ 4438 Images

✔ Training Accuracy : 90.12%

✔ Validation Accuracy : 80.61%

✔ Ready for Real-Time Prediction
""")

# =====================================================
# ABOUT PROJECT PAGE
# =====================================================

elif page == "ℹ About Project":

    st.markdown(
        "<h1 class='title'>ℹ About This Project</h1>",
        unsafe_allow_html=True,
    )

    st.markdown("---")

    st.header("🚦 Indian Traffic Sign Recognition System")

    st.write("""
This project is an Artificial Intelligence application that recognizes
Indian Traffic Signs using Deep Learning.

The system has been developed using **Transfer Learning**
with **MobileNetV2**.

The application allows users to upload
a traffic sign image and predicts
the corresponding traffic sign with
its confidence score.

The model has been trained on
85 Indian Traffic Sign classes.
""")

    st.markdown("---")

    st.header("🧠 Technologies Used")

    tech = pd.DataFrame({

        "Technology":[
            "Python",
            "TensorFlow",
            "MobileNetV2",
            "Streamlit",
            "NumPy",
            "Pillow",
            "Plotly",
            "Pandas"
        ],

        "Purpose":[
            "Programming Language",
            "Deep Learning",
            "Transfer Learning Model",
            "Web Application",
            "Numerical Computation",
            "Image Processing",
            "Charts",
            "Data Handling"
        ]

    })

    st.table(tech)

    st.markdown("---")

    st.header("📌 Project Workflow")

    st.write("""
1. Collect Dataset

2. Preprocess Images

3. Train MobileNetV2

4. Save Best Model

5. Save Class Names

6. Upload Image

7. Predict Traffic Sign

8. Display Confidence

9. Show Top 5 Predictions

10. Display Result
""")

    st.markdown("---")

    st.header("🎯 Project Objective")

    st.info("""
To automatically recognize Indian traffic signs
using Artificial Intelligence and Deep Learning
so that drivers and autonomous vehicles can
understand road signs accurately and quickly.
""")
    
# =====================================================
# FOOTER
# =====================================================

st.markdown("---")

st.markdown(
    """
    <div style='text-align:center;
                padding:20px;
                color:gray;
                font-size:16px;'>

        🚦 <b>Indian Traffic Sign Recognition System</b><br><br>

        Developed using <b>Python • TensorFlow • MobileNetV2 • Streamlit</b><br><br>

        Artificial Intelligence & Neural Network Based Project<br><br>

        © 2026 All Rights Reserved

    </div>
    """,
    unsafe_allow_html=True
)

# =====================================================
# SIDEBAR FOOTER
# =====================================================

st.sidebar.markdown("---")

st.sidebar.success("✔ System Ready")

st.sidebar.markdown(
"""
### 📊 Model Information

**Model**
- MobileNetV2

**Framework**
- TensorFlow

**Classes**
- 85

**Training Accuracy**
- 90.12%

**Validation Accuracy**
- 80.61%

**Image Size**
- 224 × 224
"""
)

st.sidebar.markdown("---")

st.sidebar.caption("Version 1.0")

# =====================================================
# APP STATUS
# =====================================================

try:
    model

except NameError:

    st.error("Model not loaded.")

# =====================================================
# THANK YOU MESSAGE
# =====================================================

st.markdown(
"""
<div style="text-align:center;
padding-top:30px;
color:#00E5FF;
font-size:20px;
font-weight:bold;">

🚀 Thank you for using the Indian Traffic Sign Recognition System

</div>
""",
unsafe_allow_html=True
)