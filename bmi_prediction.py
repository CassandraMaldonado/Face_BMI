# BMI Prediction from Face Images
# Based on the paper: https://cdn.aaai.org/ojs/14923/14923-28-18442-1-2-20201228.pdf

import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import tensorflow as tf
from mtcnn import MTCNN
import base64
from PIL import Image
from io import BytesIO

# Import our custom VGG-Face implementation
from vgg_face import get_vgg_face_model, preprocess_input

# Define image size (VGG-Face input size)
IMAGE_SIZE = (224, 224)

class BMIPredictor:
    """
    Class for BMI prediction from face images using a VGG-Face based model
    """
    
    def __init__(self, model_path=None):
        """
        Initialize the BMI predictor
        
        Args:
            model_path: Path to a trained model file (optional)
        """
        self.detector = MTCNN()
        self.model = None
        
        if model_path and os.path.exists(model_path):
            print(f"Loading model from {model_path}")
            self.model = load_model(model_path)
    
    def create_model(self):
        """
        Create a BMI prediction model based on VGG-Face
        
        Returns:
            The created model
        """
        # Load VGG-Face model with pre-trained weights
        base_model = get_vgg_face_model(include_top=False, input_shape=(224, 224, 3))
        
        # Freeze the base model layers
        for layer in base_model.layers:
            layer.trainable = False
        
        # Add custom regression layers
        x = base_model.output
        x = Flatten()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.5)(x)
        predictions = Dense(1, activation='linear')(x)  # BMI is a continuous value
        
        # Create the model
        model = Model(inputs=base_model.input, outputs=predictions)
        
        # Compile the model
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
        
        self.model = model
        return model
    
    def preprocess_face_image(self, image_path=None, img=None, method='mtcnn'):
        """
        Detect face in image and preprocess it for VGG-Face
        
        Args:
            image_path: Path to image file (optional)
            img: Numpy array image (optional)
            method: Face detection method ('mtcnn' or None)
            
        Returns:
            Preprocessed image array
        """
        try:
            # Load image if path is provided
            if image_path:
                img = cv2.imread(image_path)
                if img is None:
                    print(f"Error loading image: {image_path}")
                    return None
                
                # Convert BGR to RGB (VGG-Face expects RGB)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Make sure we have an image
            if img is None:
                print("No image provided")
                return None
            
            # Convert to RGB if needed
            if len(img.shape) == 2:  # Grayscale
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.shape[2] == 4:  # RGBA
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            
            if method == 'mtcnn':
                # Use MTCNN for face detection
                faces = self.detector.detect_faces(img)
                if not faces:
                    print(f"No face detected in the image")
                    # Use the entire image if no face is detected
                    face_img = cv2.resize(img, IMAGE_SIZE)
                else:
                    # Get the largest face
                    face = max(faces, key=lambda x: x['box'][2] * x['box'][3])
                    x, y, w, h = face['box']
                    
                    # Ensure coordinates are valid
                    x, y = max(0, x), max(0, y)
                    w = min(w, img.shape[1] - x)
                    h = min(h, img.shape[0] - y)
                    
                    # Crop the face
                    face_img = img[y:y+h, x:x+w]
                    face_img = cv2.resize(face_img, IMAGE_SIZE)
            else:
                # Use the entire image if not using MTCNN
                face_img = cv2.resize(img, IMAGE_SIZE)
            
            # Preprocess for VGG-Face
            face_img = preprocess_input(face_img, version=1)
            
            return face_img
            
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            return None
    
    def preprocess_base64_image(self, base64_string):
        """
        Process base64 image for prediction
        
        Args:
            base64_string: Base64 encoded image
            
        Returns:
            Preprocessed image array
        """
        try:
            # Decode base64 string
            img_data = base64.b64decode(base64_string)
            img = Image.open(BytesIO(img_data))
            img = np.array(img)
            
            # Process the face
            face_img = self.preprocess_face_image(img=img, method='mtcnn')
            
            if face_img is None:
                return None
                
            return np.expand_dims(face_img, axis=0)
            
        except Exception as e:
            print(f"Error processing base64 image: {str(e)}")
            return None
    
    def load_data(self, data_path, image_folder):
        """
        Load data from CSV and preprocess images
        
        Args:
            data_path: Path to CSV file with BMI data
            image_folder: Path to folder containing images
            
        Returns:
            X, y arrays for training
        """
        print("Loading and preprocessing dataset...")
        
        # Read the data.csv file
        df = pd.read_csv(data_path)
        
        # Initialize arrays for features and labels
        X = []
        y = []
        
        total = len(df)
        for idx, row in df.iterrows():
            print(f"Processing image {idx+1}/{total}: {row['name']}")
            
            image_path = os.path.join(image_folder, row['name'])
            
            # Check if image exists
            if not os.path.exists(image_path):
                print(f"Warning: Image {image_path} not found")
                continue
            
            # Load and preprocess image
            img = self.preprocess_face_image(image_path=image_path)
            if img is not None:
                X.append(img)
                y.append(row['bmi'])
        
        return np.array(X), np.array(y)
    
    def train(self, X, y, epochs=50, batch_size=4, validation_split=0.2, callbacks=None):
        """
        Train the model with the dataset
        
        Args:
            X: Input features
            y: Target values
            epochs: Number of training epochs
            batch_size: Batch size
            validation_split: Validation data proportion
            callbacks: List of Keras callbacks
            
        Returns:
            Training history
        """
        if self.model is None:
            self.create_model()
            
        # Train the model
        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=1,
            callbacks=callbacks
        )
        
        return history
    
    def predict_bmi(self, image_path=None, img=None, base64_string=None):
        """
        Predict BMI from an image
        
        Args:
            image_path: Path to image file (optional)
            img: Numpy array image (optional)
            base64_string: Base64 encoded image (optional)
            
        Returns:
            Predicted BMI value and category
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call create_model() or load a model first.")
        
        # Process the image
        if base64_string:
            processed_image = self.preprocess_base64_image(base64_string)
        elif image_path or img is not None:
            face_img = self.preprocess_face_image(image_path=image_path, img=img)
            if face_img is None:
                return None, None
            processed_image = np.expand_dims(face_img, axis=0)
        else:
            raise ValueError("No image provided. Provide image_path, img or base64_string.")
        
        if processed_image is None:
            return None, None
            
        # Make prediction
        predicted_bmi = float(self.model.predict(processed_image)[0][0])
        
        # Determine BMI category
        category = 'Unknown'
        if predicted_bmi < 18.5:
            category = 'Underweight'
        elif 18.5 <= predicted_bmi < 25:
            category = 'Normal weight'
        elif 25 <= predicted_bmi < 30:
            category = 'Overweight'
        else:
            category = 'Obesity'
        
        return predicted_bmi, category
    
    def save_model(self, model_path):
        """
        Save the model to a file
        
        Args:
            model_path: Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save. Call create_model() first.")
            
        self.model.save(model_path)
        print(f"Model saved to {model_path}")
    
    def save_training_plot(self, history, filename='training_history.png'):
        """
        Save the training history as a plot
        
        Args:
            history: Training history
            filename: Output file name
        """
        plt.figure(figsize=(12, 4))
        
        # Plot training & validation loss
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper right')
        
        # Plot training & validation mean absolute error
        plt.subplot(1, 2, 2)
        plt.plot(history.history['mae'])
        plt.plot(history.history['val_mae'])
        plt.title('Model Mean Absolute Error')
        plt.ylabel('MAE')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper right')
        
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()