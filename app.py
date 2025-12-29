import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="CardioTrack AI",
    page_icon="❤️",
    layout="wide"
)

# --- Custom Styling (Light Red Background) ---
st.markdown("""
    <style>
    /* Light Red/Soft Rose Background */
    .stApp {
        background-color: #fff5f5;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #ffe3e3;
    }

    /* Professional Card Containers */
    .stMetric {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    h1, h2, h3 {
        color: #c92a2a;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Sidebar with Medical Descriptions ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/822/822118.png", width=80)
    st.title("About CardioTrack")
    st.write("""
    **CardioTrack AI** uses Deep Learning (ResNet18) to analyze ECG images and identify potential cardiac issues instantly.
    """)
    
    st.divider()
    
    st.subheader("Glossary")
    with st.expander("❤️ Normal"):
        st.write("The heart follows a steady, consistent rhythm (Sinus Rhythm) within 60-100 BPM.")
        
    with st.expander("⚡ Arrhythmia"):
        st.write("An irregular heartbeat. The heart may beat too fast, too slow, or with an uneven pattern.")
        
    with st.expander("⚠️ Myocardial Infarction (MI)"):
        st.write("Commonly known as a **Heart Attack**. Occurs when blood flow to the heart muscle is blocked.")

    st.divider()
    st.caption("Disclaimer: This tool is for educational purposes and is not a substitute for professional medical advice.")

# --- Load Model ---
@st.cache_resource
def load_model():
    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, 3)
    # Ensure this path matches your folder structure exactly
    model.load_state_dict(torch.load('model/ecg_model.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

# --- Main Interface ---
st.title("❤️ CardioTrack: ECG Prediction Dashboard")

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📤 Upload Patient ECG")
    uploaded_file = st.file_uploader("Upload a clear scan of the ECG strip", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Original ECG Scan', use_container_width=True)

with col2:
    st.markdown("### 📊 AI Analysis")
    
    if uploaded_file is not None:
        # Load logic
        model = load_model()
        classes = ['Arrhythmia', 'Myocardial Infarction', 'Normal']
        
        # Preprocessing
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        input_tensor = preprocess(image).unsqueeze(0)

        with st.status("Running Cardiac Neural Network...", expanded=True) as status:
            time.sleep(1.2) # For visual effect
            with torch.no_grad():
                output = model(input_tensor)
                probs = torch.nn.functional.softmax(output[0], dim=0)
                conf, pred = torch.max(probs, 0)
                result = classes[pred]
            status.update(label="Analysis Complete!", state="complete", expanded=False)

        # Result Card
        color = "#2f9e44" if result == "Normal" else "#e03131"
        st.markdown(f"""
            <div style="background-color: white; padding: 25px; border-radius: 15px; border-top: 5px solid {color}; shadow: 0 4px;">
                <h4 style="color: grey; margin-bottom: 5px;">DETECTION RESULT</h4>
                <h2 style="color: {color}; margin-top: 0;">{result}</h2>
                <hr>
                <p style="color: #444;">AI Confidence Score: <b>{conf*100:.1f}%</b></p>
            </div>
        """, unsafe_allow_html=True)
        
    else:
        st.info("Waiting for ECG upload to generate report...")