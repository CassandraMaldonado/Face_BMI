import streamlit as st
from PIL import Image
import numpy as np
import random
import time

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

# Simulate face detection without OpenCV
def draw_rectangle(image):
    # Create a copy of the image
    img_with_rect = image.copy()
    
    # Get a drawing context
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img_with_rect)
    
    # Get image dimensions
    width, height = image.size
    
    # Determine face area (simplified - just use center of image)
    center_x, center_y = width // 2, height // 2
    box_size = min(width, height) // 2
    x = center_x - box_size // 2
    y = center_y - box_size // 2
    
    # Draw rectangle
    draw.rectangle([(x, y), (x + box_size, y + box_size)], outline=(0, 255, 0), width=3)
    
    return img_with_rect

# Simulated face detection and BMI prediction
def demo_prediction(image):
    with st.spinner("Processing image..."):
        try:
            # Simulate processing delay
            time.sleep(1.5)
            
            # Draw rectangle on the face
            img_with_rect = draw_rectangle(image)
            
            # Display the image with rectangle
            st.image(img_with_rect, caption="Simulated Face Detection", use_column_width=True)
            
            # Simulate more processing
            progress_bar = st.progress(0)
            for i in range(101):
                progress_bar.progress(i/100)
                time.sleep(0.01)
            
            # Simulate a BMI prediction
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
            st.warning("This is a simulated prediction for demo purposes only. In the full app, a real AI model would make accurate predictions.")
            display_bmi_category(bmi, category)
            
        except Exception as e:
            st.error(f"Error during processing: {str(e)}")
            return

# Main app
st.title("Face to BMI Prediction Demo")
st.write("Upload a face image to see a simulated BMI prediction")
st.info("Note: This is a demo version without the actual ML model. It shows how the UI would work but makes random predictions for demonstration purposes.")

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

# About section
st.markdown("---")
st.subheader("About This Project")
st.write("""
This application demonstrates how facial features might correlate with BMI values. 
In the full version, a deep learning model based on VGG-Face architecture analyzes 
facial features to predict BMI.

The model is trained on a dataset of face images with known BMI values, leveraging 
transfer learning to repurpose facial recognition capabilities for BMI estimation.

**Technical Components:**
- Face detection using MTCNN
- Feature extraction with VGG-Face
- Custom regression layers for BMI prediction
- Streamlit for the user interface
""")

# Footer
st.markdown("---")
st.caption("Face to BMI Prediction Demo App - UI Demonstration")