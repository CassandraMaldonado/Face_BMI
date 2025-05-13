import os
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import matplotlib.pyplot as plt

# Import our BMI predictor
from bmi_prediction import BMIPredictor

# Set up your specific paths
DATA_PATH = "/Users/casey/Documents/GitHub/Face_to_BMI/data.csv"
IMAGE_PATH = "/Users/casey/Documents/GitHub/Face_to_BMI/Images"
MODEL_PATH = "/Users/casey/Documents/GitHub/Face_to_BMI/bmi_predictor_model.h5"

# Page configuration
st.set_page_config(
    page_title="Face to BMI Prediction",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Function to get predictor
def get_predictor():
    return BMIPredictor(MODEL_PATH if os.path.exists(MODEL_PATH) else None)

# Initialize predictor at startup
if 'predictor' not in st.session_state:
    st.session_state.predictor = get_predictor()

# Function to display BMI category with color
def display_bmi_category(bmi, category):
    if category == 'Underweight':
        st.markdown(f"<div style='background-color:#89CFF0;padding:10px;border-radius:5px;text-align:center;'><strong>Category:</strong> {category} (BMI: {bmi:.2f})</div>", unsafe_allow_html=True)
        st.info("BMI less than 18.5 indicates underweight. This may be associated with certain health issues.")
    elif category == 'Normal weight':
        st.markdown(f"<div style='background-color:#90EE90;padding:10px;border-radius:5px;text-align:center;'><strong>Category:</strong> {category} (BMI: {bmi:.2f})</div>", unsafe_allow_html=True)
        st.info("BMI between 18.5 and 24.9 indicates a healthy weight for most adults.")
    elif category == 'Overweight':
        st.markdown(f"<div style='background-color:#FFD700;padding:10px;border-radius:5px;text-align:center;'><strong>Category:</strong> {category} (BMI: {bmi:.2f})</div>", unsafe_allow_html=True)
        st.warning("BMI between 25 and 29.9 indicates overweight. This may increase risk for certain diseases.")
    elif category == 'Obesity':
        st.markdown(f"<div style='background-color:#FFA07A;padding:10px;border-radius:5px;text-align:center;'><strong>Category:</strong> {category} (BMI: {bmi:.2f})</div>", unsafe_allow_html=True)
        st.error("BMI of 30 or higher indicates obesity. This increases risk for many health conditions.")

# Function to train the model
def train_model():
    st.info("Loading and preprocessing dataset...")
    
    # Check if data exists
    if not os.path.exists(DATA_PATH):
        st.error(f"Data file not found: {DATA_PATH}")
        return
    
    if not os.path.exists(IMAGE_PATH):
        st.error(f"Image folder not found: {IMAGE_PATH}")
        return
    
    # Load and preprocess the dataset
    progress_bar = st.progress(0)
    
    # We'll manually handle the progress updates since we can't modify the load_data function
    df = pd.read_csv(DATA_PATH)
    total_images = len(df)
    
    # Create placeholders for progress updates
    status_text = st.empty()
    
    # Create a new predictor for training
    predictor = BMIPredictor()
    
    # Load data
    X, y = predictor.load_data(DATA_PATH, IMAGE_PATH)
    
    if len(X) == 0:
        st.error("No valid images found. Please check the data paths.")
        return
    
    st.success(f"Loaded {len(X)} images for training")
    
    # Parameters
    epochs = 50  # You can make this a parameter
    batch_size = 4  # You can make this a parameter
    validation_split = 0.2  # You can make this a parameter
    
    # Create and train the model
    st.info("Creating and training the model...")
    model = predictor.create_model()
    
    # Set up progress monitoring
    progress_text = st.empty()
    
    def update_progress(epoch, epochs):
        progress = (epoch + 1) / epochs
        progress_bar.progress(progress)
        progress_text.text(f"Training Progress: {int(progress * 100)}% - Epoch {epoch+1}/{epochs}")
    
    # Train in smaller chunks to show progress
    history = {'loss': [], 'val_loss': [], 'mae': [], 'val_mae': []}
    
    # Train for each epoch
    for epoch in range(epochs):
        update_progress(epoch, epochs)
        h = model.fit(
            X, y,
            epochs=1,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=0
        )
        # Collect metrics
        history['loss'].append(h.history['loss'][0])
        history['val_loss'].append(h.history['val_loss'][0])
        history['mae'].append(h.history['mae'][0])
        history['val_mae'].append(h.history['val_mae'][0])
        
        # Show current metrics
        st.text(f"Loss: {h.history['loss'][0]:.4f}, Val Loss: {h.history['val_loss'][0]:.4f}")
        st.text(f"MAE: {h.history['mae'][0]:.4f}, Val MAE: {h.history['val_mae'][0]:.4f}")
    
    # Save the model
    predictor.save_model(MODEL_PATH)
    st.success(f"Model saved to {MODEL_PATH}")
    
    # Display training history plot
    st.subheader("Training History")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot training & validation loss
    ax1.plot(history['loss'])
    ax1.plot(history['val_loss'])
    ax1.set_title('Model Loss')
    ax1.set_ylabel('Loss')
    ax1.set_xlabel('Epoch')
    ax1.legend(['Train', 'Validation'], loc='upper right')
    
    # Plot training & validation mean absolute error
    ax2.plot(history['mae'])
    ax2.plot(history['val_mae'])
    ax2.set_title('Model Mean Absolute Error')
    ax2.set_ylabel('MAE')
    ax2.set_xlabel('Epoch')
    ax2.legend(['Train', 'Validation'], loc='upper right')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Save the plot
    history_plot_path = "/Users/casey/Documents/GitHub/Face_to_BMI/training_history.png"
    plt.savefig(history_plot_path)
    st.success(f"Training history plot saved to {history_plot_path}")
    
    # Update the predictor in session state
    st.session_state.predictor = BMIPredictor(MODEL_PATH)
    
    # Show success message
    st.success("Model has been trained successfully and is ready to use!")

# Function to preprocess and predict BMI from an image
def predict_from_image(image):
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model not found at {MODEL_PATH}. Please train the model first.")
        return
    
    # Convert PIL Image to numpy array
    img_array = np.array(image)
    
    # Process face
    with st.spinner("Processing image..."):
        # Get the predictor from session state
        predictor = st.session_state.predictor
        
        # Detect face
        faces = predictor.detector.detect_faces(img_array)
        
        if not faces:
            st.warning("No face detected in the image. Using the entire image.")
            processed_img = cv2.resize(img_array, (224, 224))
            
            # Display the processed image
            st.image(processed_img, caption="Processed Image (No face detected)", use_column_width=True)
        else:
            # Get the largest face
            face = max(faces, key=lambda x: x['box'][2] * x['box'][3])
            x, y, w, h = face['box']
            
            # Ensure coordinates are valid
            x, y = max(0, x), max(0, y)
            w = min(w, img_array.shape[1] - x)
            h = min(h, img_array.shape[0] - y)
            
            # Draw rectangle on the face
            img_with_rect = img_array.copy()
            cv2.rectangle(img_with_rect, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Display the image with rectangle
            st.image(img_with_rect, caption="Detected Face", use_column_width=True)
            
            # Crop and process the face
            face_img = img_array[y:y+h, x:x+w]
            processed_img = cv2.resize(face_img, (224, 224))
        
        # Make prediction
        bmi, category = predictor.predict_bmi(img=processed_img)
        
        if bmi is None:
            st.error("Could not predict BMI from the image.")
            return
        
        # Display results
        st.subheader("Prediction Results")
        display_bmi_category(bmi, category)

# Main app interface
def main():
    st.title("Face to BMI Prediction")
    st.write("This app predicts BMI from facial images using deep learning.")
    
    # Sidebar
    st.sidebar.title("About")
    st.sidebar.info(
        "This app is based on the research paper: "
        "[Face to BMI: Using Computer Vision to Infer Body Mass Index on Social Media]"
        "(https://cdn.aaai.org/ojs/14923/14923-28-18442-1-2-20201228.pdf)"
    )
    
    # Check if model exists
    model_exists = os.path.exists(MODEL_PATH)
    if not model_exists:
        st.warning(f"Model not found at {MODEL_PATH}. You need to train the model first.")
    
    # App modes
    app_mode = st.sidebar.selectbox(
        "Choose the app mode",
        ["Show Instructions", "Train Model", "Predict from Image"]
    )
    
    # Show Instructions
    if app_mode == "Show Instructions":
        st.markdown("""
        ## Instructions
        
        This application uses deep learning to predict BMI (Body Mass Index) from facial images.
        
        ### How to use:
        
        1. **Train Model**: Use the provided dataset to train a custom model.
        2. **Predict from Image**: Upload an image to get a BMI prediction.
        
        ### Important Notes:
        
        - The model works best with clear, front-facing images.
        - The prediction is based solely on facial features and should be used for reference only.
        - Actual BMI calculation requires height and weight measurements.
        - This application is a demonstration of the technology and not a medical tool.
        """)
        
        st.markdown("""
        ## About the Model
        
        The model is based on the VGG-Face architecture, pre-trained on facial recognition tasks and fine-tuned for BMI regression.
        
        ### How it works:
        
        1. Face detection using MTCNN (Multi-task Cascaded Convolutional Networks)
        2. Image preprocessing for VGG-Face input
        3. Feature extraction using VGG-Face
        4. BMI prediction using custom regression layers
        
        ### Research Reference:
        
        This application is inspired by the paper "Face to BMI: Using Computer Vision to Infer Body Mass Index on Social Media"
        which examines the correlation between facial features and BMI.
        """)
    
    # Train Model
    elif app_mode == "Train Model":
        st.header("Train Model")
        
        # Check if data paths exist
        data_exists = os.path.exists(DATA_PATH)
        images_exist = os.path.exists(IMAGE_PATH)
        
        if not data_exists:
            st.error(f"Data file not found: {DATA_PATH}")
        
        if not images_exist:
            st.error(f"Images folder not found: {IMAGE_PATH}")
        
        if data_exists and images_exist:
            st.write("Ready to train the model using the provided dataset.")
            
            # Display dataset info if possible
            try:
                df = pd.read_csv(DATA_PATH)
                st.write(f"Dataset size: {len(df)} samples")
                
                # Show a few rows
                st.write("Dataset preview:")
                st.dataframe(df.head())
                
                # Show some statistics
                st.write("BMI Statistics:")
                stats = df['bmi'].describe()
                st.write(f"Min: {stats['min']:.2f}, Max: {stats['max']:.2f}, Mean: {stats['mean']:.2f}")
                
                # Count BMI categories
                df['category'] = pd.cut(
                    df['bmi'],
                    bins=[0, 18.5, 25, 30, 100],
                    labels=['Underweight', 'Normal', 'Overweight', 'Obese']
                )
                
                category_counts = df['category'].value_counts()
                
                # Show category distribution
                st.write("BMI Category Distribution:")
                st.bar_chart(category_counts)
                
            except Exception as e:
                st.warning(f"Could not analyze dataset: {str(e)}")
            
            if st.button("Start Training"):
                train_model()
        
        else:
            st.error("Cannot train the model because data files are missing.")
    
    # Predict from Image
    elif app_mode == "Predict from Image":
        st.header("Predict BMI from Image")
        
        if not model_exists:
            st.error(f"Model not found at {MODEL_PATH}. Please train the model first.")
            return
        
        # Upload image
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "bmp"])
        
        if uploaded_file is not None:
            # Read the image
            image = Image.open(uploaded_file)
            
            # Display the original image
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Process the image and predict
            predict_from_image(image)

if __name__ == "__main__":
    main()