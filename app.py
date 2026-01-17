import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Banana Shelf-Life AI",
    page_icon="🍌",
    layout="centered"
)

# --- 1. MODEL LOADER ---
@st.cache_resource
def load_model():
    try:
        # Re-create the exact architecture used in training
        model = models.resnet18(weights=None)
        
        # We must match the layer structure exactly:
        # Linear -> ReLU -> Dropout -> Linear
        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )
        
        # Load the weights
        model_path = 'models/banana_regression_model.pth'
        
        # Handle case where user puts model in root instead of models/
        if not os.path.exists(model_path):
            if os.path.exists('banana_regression_model.pth'):
                model_path = 'banana_regression_model.pth'
            else:
                return None

        # Load weights onto CPU (since your laptop might not have CUDA)
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# --- 2. PREDICTION ENGINE ---
def predict_days(image, model):
    # Same transforms as Validation set
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Add batch dimension (1, 3, 224, 224)
    img_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        days_left = model(img_tensor).item()
    
    return days_left

# --- 3. UI LAYOUT ---
st.title("🍌 Banana Shelf-Life Predictor")
st.markdown("### Computer Vision Regression Model")
st.info("Upload a photo of your bananas. The AI will estimate how many days until they spoil.")

# Load Model
model = load_model()

if model is None:
    st.error("⚠️ Model file not found! Please place 'banana_regression_model.pth' in the folder.")
else:
    # File Uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display Image
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Your Banana', use_container_width=True)
        
        # Progress Bar for "Thinking"
        with st.spinner('Analyzing texture patterns...'):
            days = predict_days(image, model)
            
            # Clamp negative values (Physics says time doesn't go back!)
            days = max(0.0, days)

        # --- DISPLAY RESULTS ---
        st.markdown("---")
        
        # Dynamic Color Coding
        if days < 1.0:
            color = "red"
            status = "Immediate Action Required!"
            advice = "These are perfect for banana bread. Do not wait."
        elif days < 3.0:
            color = "orange"
            status = "Ripe & Ready"
            advice = "Eat them soon. They are at peak sweetness."
        else:
            color = "green"
            status = "Fresh"
            advice = "You have plenty of time. Store in a cool place."

        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(label="Estimated Days Left", value=f"{days:.1f} Days")
        
        with col2:
            st.markdown(f":{color}[**{status}**]")
            st.write(advice)

        # Technical Details (Good for Interview Demo)
        with st.expander("Show Technical Details"):
            st.write(f"**Raw Model Output:** {days:.4f}")
            st.write("**Model Architecture:** ResNet-18 (Fine-Tuned)")
            st.write("**Error Margin (MAE):** ±0.57 Days")