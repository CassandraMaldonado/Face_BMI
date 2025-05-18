import streamlit as st
import numpy as np
from PIL import Image
import cv2

# Page configuration
st.set_page_config(
    page_title="Face to BMI Prediction Demo",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Function to display BMI category with color
def display_bmi_category(bmi, category):
    if category == 'Underweight':
        st.markdown(f"<div style='background-color:#89CFF0;padding:10px;border-radius:5px;text-align:center;'><strong>Category:</strong> {category} (BMI: {bmi:.2f})</div>", unsafe_allow_html=True)
    elif category == 'Normal weight':
        st.markdown(f"<div style='background-color:#90EE90;padding:10px;border-radius:5px;text-align:center;'><strong>Category:</strong> {category} (BMI: {bmi:.2f})</div>", unsafe_allow_html=True)
    elif category == 'Overweight':
        st.markdown(f"<div style='background-color:#FFD700;padding:10px;border-radius:5px;text-align:center;'><strong>Category:</strong> {category} (BMI: {bmi:.2f})</div>", unsafe_allow_html=True)
    elif category == 'Obesity':
        st.markdown(f"<div style='background-color:#FFA07A;padding:10px;border-radius:5px;text-align:center;'><strong>Category:</strong> {category} (BMI: {bmi:.2f})</div>", unsafe_allow_html=True)

# Simulated face detection and BMI prediction
def demo_prediction(image):
    # Convert PIL Image to numpy array
    img_array = np.array(image)
    
    # Process face (simulated)
    with st.spinner("Processing image..."):
        try:
            # Simplified face detection - just for demo
            height, width = img_array.shape[:2]
            center_x, center_y = width // 2, height // 2
            box_size = min(width, height) // 2
            x = center_x - box_size // 2
            y = center_y - box_size // 2
            w = h = box_size
            
            # Draw rectangle on the face
            img_with_rect = img_array.copy()
            cv2.rectangle(img_with_rect, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Display the image with rectangle
            st.image(img_with_rect, caption="Simulated Face Detection", use_column_width=True)
            
            # Simulate a BMI prediction
            import random
            bmi = random.uniform(18.0, 32.0)
            
            # Determine BMI category
            if bmi < 18.5:
                category = 'Underweight'
            elif 18.5 <= bmi < 25:
                category = 'Normal weight'
            elif 25 <= bmi < 30:
                category = 'Overweight'
            else:
                category = 'Obesity'
            
            # Display results
            st.subheader("Demo Prediction Results")
            st.warning("This is a simulated prediction for demo purposes only")
            display_bmi_category(bmi, category)
            
        except Exception as e:
            st.error(f"Error during processing: {str(e)}")
            return

# Main app
st.title("Face to BMI Prediction Demo")
st.write("Upload a face image to see a simulated BMI prediction")
st.info("Note: This is a demo version without the actual ML model. It shows how the UI would work but doesn't make real predictions.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "bmp"])

if uploaded_file is not None:
    try:
        # Read the image
        image = Image.open(uploaded_file)
        
        # Display the original image
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Process the image and predict
        if st.button("Generate Demo Prediction"):
            demo_prediction(image)
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")

# Footer
st.markdown("---")
st.caption("Face to BMI Prediction Demo App - UI Demonstration")