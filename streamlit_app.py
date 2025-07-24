import streamlit as st
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

# ---- Setup ----
st.set_page_config(
    page_title="Mosquito Classifier",
    page_icon="🦟",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ---- Custom CSS Styling ----
st.markdown("""
    <style>
        body {
            background-color: #f6fff8;
            font-family: 'Segoe UI', sans-serif;
        }
        .stButton>button {
            background-color: #56ab2f;
            color: white;
            font-size: 16px;
            border-radius: 8px;
            padding: 10px 24px;
        }
        .stMarkdown {
            font-size: 18px;
        }
        .stImage>img {
            border-radius: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# ---- Sidebar ----
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/f/f8/Aedes_aegypti_dorsal.jpg/320px-Aedes_aegypti_dorsal.jpg", use_column_width=True)
st.sidebar.title("🧬 Mosquito Classifier")
st.sidebar.markdown("Upload an image of a mosquito to identify its species or detect presence.")
st.sidebar.info("Supports `.jpg`, `.jpeg`, `.png`")

# ---- Load Model ----
@st.cache_resource
def load_model():
    model = torch.load("mosquito_model.pt", map_location=torch.device("cpu"))
    model.eval()
    return model

# ---- Preprocess ----
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # adjust as per training
    ])
    return transform(image).unsqueeze(0)

# ---- Prediction ----
CLASS_NAMES = ["Not Mosquito", "Aedes aegypti", "Anopheles", "Culex", "Other"]

def predict_image(image, model):
    input_tensor = preprocess_image(image)
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        predicted_idx = torch.argmax(probabilities).item()
        return CLASS_NAMES[predicted_idx], probabilities.numpy()

# ---- Main Interface ----
st.title("🦟 Smart Mosquito Identifier")
st.subheader("Upload an image to detect and classify mosquito species.")

uploaded_file = st.file_uploader("Upload image here:", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="🔍 Uploaded Image", use_column_width=True)

    if st.button("🔎 Classify"):
        with st.spinner("Detecting..."):
            model = load_model()
            label, probs = predict_image(image, model)

        st.success(f"🧬 Prediction: **{label}**")
        st.caption("Prediction Confidence Levels:")

        # ---- Bar Chart Visualization ----
        fig, ax = plt.subplots()
        ax.bar(CLASS_NAMES, probs, color="#56ab2f")
        ax.set_ylabel("Confidence")
        ax.set_ylim(0, 1)
        ax.set_title("Model Confidence for Each Class")
        st.pyplot(fig)
