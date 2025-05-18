import streamlit as st
from PIL import Image, ImageDraw, ImageEnhance
import numpy as np
import os
import time

# Import the actual BMI prediction functionality
from bmi_prediction import BMIPredictor

# Set page configuration
st.set_page_config(
    page_title="Face to BMI Prediction",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Path to the saved model
MODEL_PATH = "test_model.h5"

# Custom CSS for styling
st.markdown("""
<style>
    h1 {
        color: #2c3e50;
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 700;
    }
    h2, h3 {
        color: #34495e;
        font-family: 'Helvetica Neue', sans-serif;
    }
    .stButton button {
        background-color: #3498db;
        color: white;
        border-radius: 6px;
        padding: 0.5em 1em;
        font-weight: bold;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        background-color: #2980b9;
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .info-box {
        background-color: #f1f8ff;
        border-left: 5px solid #4b6fff;
        padding: 20px;
        border-radius: 4px;
        margin-bottom: 20px;
    }
    .img-container {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin: 20px 0;
    }
    .category-badge {
        padding: 10px;
        border-radius: 5px;
        text-align: center;
        font-weight: bold;
        margin: 15px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Function to display BMI category with styling
def display_bmi_category(bmi, category):
    if category == 'Underweight':
        color = "#3498db"
        icon = "⬇️"
        description = "BMI less than 18.5 indicates underweight. This may be associated with certain health issues."
    elif category == 'Normal weight':
        color = "#2ecc71"
        icon = "✅"
        description = "BMI between 18.5 and 24.9 indicates a healthy weight for most adults."
    elif category == 'Overweight':
        color = "#f39c12"
        icon = "⚠️"
        description = "BMI between 25 and 29.9 indicates overweight. This may increase risk for certain diseases."
    elif category == 'Obesity':
        color = "#e74c3c"
        icon = "❗"
        description = "BMI of 30 or higher indicates obesity. This increases risk for many health conditions."
    
    st.markdown(f"""
    <div class="category-badge" style="background-color:{color}20; border-left:5px solid {color}; color:{color};">
        <h2 style="margin:0;">{icon} {category} (BMI: {bmi:.1f})</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div style="margin-bottom:25px;">
        {description}
    </div>
    """, unsafe_allow_html=True)
    
    # Add a visualization gauge
    st.markdown("### BMI Scale")
    
    # Calculate position on the BMI scale (0-100%)
    if bmi < 15:
        position = 0
    elif bmi > 35:
        position = 100
    else:
        position = (bmi - 15) * 100 / 20  # Scale 15-35 to 0-100%
    
    # Create the gauge sections
    gauge_html = f"""
    <div style="display:flex; height:25px; border-radius:5px; overflow:hidden; margin-bottom:10px; width:100%;">
        <div style="background-color:#3498db; width:17.5%; height:100%;"></div>
        <div style="background-color:#2ecc71; width:32.5%; height:100%;"></div>
        <div style="background-color:#f39c12; width:25%; height:100%;"></div>
        <div style="background-color:#e74c3c; width:25%; height:100%;"></div>
    </div>
    <div style="position:relative; height:20px; margin-bottom:20px;">
        <div style="position:absolute; left:0%;">15</div>
        <div style="position:absolute; left:17.5%;">18.5</div>
        <div style="position:absolute; left:50%;">25</div>
        <div style="position:absolute; left:75%;">30</div>
        <div style="position:absolute; right:0%;">35+</div>
        <div style="position:absolute; left:{position}%; transform:translateX(-50%); top:-25px;">
            <div style="width:0; height:0; border-left:10px solid transparent; border-right:10px solid transparent; border-top:10px solid #333; margin:0 auto;"></div>
        </div>
    </div>
    """
    st.markdown(gauge_html, unsafe_allow_html=True)

# Function to highlight detected face
def draw_face_detection(image, face_box):
    # Create a copy of the image
    enhanced = image.copy()
    
    # Create a drawing surface
    draw = ImageDraw.Draw(enhanced)
    
    # Extract coordinates
    x, y, w, h = face_box
    
    # Draw a fancy border
    border_width = 3
    draw.rectangle(
        [(x, y), (x + w, y + h)], 
        outline=(41, 128, 185), 
        width=border_width
    )
    
    # Draw corner markers
    corner_size = 15
    # Top left
    draw.line([(x, y), (x + corner_size, y)], fill=(231, 76, 60), width=border_width)
    draw.line([(x, y), (x, y + corner_size)], fill=(231, 76, 60), width=border_width)
    # Top right
    draw.line([(x + w, y), (x + w - corner_size, y)], fill=(231, 76, 60), width=border_width)
    draw.line([(x + w, y), (x + w, y + corner_size)], fill=(231, 76, 60), width=border_width)
    # Bottom left
    draw.line([(x, y + h), (x + corner_size, y + h)], fill=(231, 76, 60), width=border_width)
    draw.line([(x, y + h), (x, y + h - corner_size)], fill=(231, 76, 60), width=border_width)
    # Bottom right
    draw.line([(x + w, y + h), (x + w - corner_size, y + h)], fill=(231, 76, 60), width=border_width)
    draw.line([(x + w, y + h), (x + w, y + h - corner_size)], fill=(231, 76, 60), width=border_width)
    
    return enhanced

# Function to make BMI prediction with the actual model
def predict_bmi_with_model(image):
    with st.spinner("Processing image..."):
        # Check if model exists
        if not os.path.exists(MODEL_PATH):
            st.error(f"Model file not found at {MODEL_PATH}. Please make sure the model is saved correctly.")
            return
        
        # Initialize progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step 1: Initialize the predictor
        status_text.text("Initializing BMI predictor...")
        try:
            predictor = BMIPredictor(MODEL_PATH)
            progress_bar.progress(20)
        except Exception as e:
            st.error(f"Error initializing predictor: {str(e)}")
            return
        
        # Step 2: Detect face
        status_text.text("Detecting face in image...")
        try:
            # Convert PIL Image to numpy array
            img_array = np.array(image)
            
            # Detect face using MTCNN
            faces = predictor.detector.detect_faces(img_array)
            progress_bar.progress(40)
            
            if not faces:
                st.warning("No face detected in the image. Using the entire image.")
                enhanced_image = image
                face_box = (0, 0, image.width, image.height)
            else:
                # Get the largest face
                face = max(faces, key=lambda x: x['box'][2] * x['box'][3])
                face_box = face['box']
                
                # Draw detection on image
                enhanced_image = draw_face_detection(image, face_box)
            
            # Display the processed image
            st.markdown("<div class='img-container'>", unsafe_allow_html=True)
            st.image(enhanced_image, caption="Face Detection", use_column_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
            progress_bar.progress(60)
        except Exception as e:
            st.error(f"Error during face detection: {str(e)}")
            return
        
        # Step 3: Predict BMI
        status_text.text("Analyzing facial features and predicting BMI...")
        try:
            bmi, category = predictor.predict_bmi(img=img_array)
            progress_bar.progress(80)
            
            if bmi is None:
                st.error("Could not predict BMI from the image.")
                return
        except Exception as e:
            st.error(f"Error during BMI prediction: {str(e)}")
            return
        
        # Step 4: Display results
        status_text.text("Preparing results...")
        time.sleep(0.5)  # Brief pause for visual effect
        progress_bar.progress(100)
        status_text.empty()
        
        # Display results header
        st.markdown("<h2 style='text-align:center; margin-top:30px; margin-bottom:30px;'>Prediction Results</h2>", unsafe_allow_html=True)
        
        # Display the BMI results
        display_bmi_category(bmi, category)
        
        # Add note about model
        st.info("This prediction was made using the actual trained model (test_model.h5).")

# Main app
st.title("Face to BMI Prediction")
st.markdown("""
<div class="info-box">
    This app uses a trained machine learning model to predict BMI from facial features.
    Upload a photo with a clearly visible face to test the model.
</div>
""", unsafe_allow_html=True)

# Create two columns for the interface
col1, col2 = st.columns([1, 1])

with col1:
    # Upload image
    st.markdown("### Upload Your Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "bmp"])
    
    if uploaded_file is not None:
        # Read the image
        image = Image.open(uploaded_file)
        
        # Display the original image
        st.markdown("<div class='img-container'>", unsafe_allow_html=True)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Button to trigger prediction
        predict_button = st.button("Predict BMI", key="predict_button")

with col2:
    if uploaded_file is not None and predict_button:
        # Process the image and show prediction
        predict_bmi_with_model(image)
    else:
        # Show instructions
        st.markdown("### Model Information")
        if os.path.exists(MODEL_PATH):
            file_size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
            st.success(f"Model loaded: {MODEL_PATH} ({file_size_mb:.2f} MB)")
            st.markdown("""
            This model is based on VGG-Face architecture and has been trained to predict BMI from facial features.
            
            To test the model:
            1. Upload an image on the left panel
            2. Click "Predict BMI"
            3. The results will appear here
            """)
        else:
            st.error(f"Model not found at {MODEL_PATH}")
            st.markdown("""
            Please make sure your model file exists at the specified path.
            Run the test_save.py script first to create the model.
            """)

# Add demo image option
st.markdown("---")
st.markdown("### Don't have an image? Try with a demo image:")

demo_image_path = "demo_face.jpg"  # Change this to your demo image path
if os.path.exists(demo_image_path):
    if st.button("Use Demo Image"):
        image = Image.open(demo_image_path)
        
        # Display the demo image
        st.markdown("<div class='img-container'>", unsafe_allow_html=True)
        st.image(image, caption="Demo Image", use_column_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Make prediction
        predict_bmi_with_model(image)
else:
    st.info("To use a demo image, add a file named 'demo_face.jpg' to your project directory.")

# Footer
st.markdown("---")
st.caption("Face to BMI Prediction | Using model: test_model.h5")