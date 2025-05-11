# Face to BMI - Main Script
# Based on the paper: https://cdn.aaai.org/ojs/14923/14923-28-18442-1-2-20201228.pdf

import os
import argparse
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Import our BMI predictor
from bmi_prediction import BMIPredictor

# Define your specific paths
DATA_PATH = "/Users/casey/Documents/GitHub/Face_to_BMI/data.csv"
IMAGE_PATH = "/Users/casey/Documents/GitHub/Face_to_BMI/Images"
MODEL_PATH = "/Users/casey/Documents/GitHub/Face_to_BMI/bmi_predictor_model.h5"
HISTORY_PLOT_PATH = "/Users/casey/Documents/GitHub/Face_to_BMI/training_history.png"

def train_model():
    """Train the BMI prediction model using the specified data"""
    print("Starting model training...")
    
    # Create predictor
    predictor = BMIPredictor()
    
    # Load and preprocess the dataset
    X, y = predictor.load_data(DATA_PATH, IMAGE_PATH)
    
    if len(X) == 0:
        print("Error: No valid images were loaded. Check your data paths.")
        return False
    
    print(f"Successfully loaded {len(X)} images with corresponding BMI values")
    
    # Create callbacks for training
    checkpoint = ModelCheckpoint(
        MODEL_PATH,
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        verbose=1,
        restore_best_weights=True
    )
    
    # Create and train the model
    print("Creating and training the model...")
    model = predictor.create_model()
    history = predictor.train(
        X, y, 
        epochs=50,  # You can adjust this
        batch_size=4,  # You can adjust this based on your memory 
        validation_split=0.2,
        callbacks=[checkpoint, early_stopping]
    )
    
    # Save the model
    predictor.save_model(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    
    # Save training history plot
    predictor.save_training_plot(history, HISTORY_PLOT_PATH)
    print(f"Training history plot saved to {HISTORY_PLOT_PATH}")
    
    return True

def predict_bmi(image_path):
    """Predict BMI from an image file"""
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}. Train the model first.")
        return None, None
    
    # Create predictor with the trained model
    predictor = BMIPredictor(MODEL_PATH)
    
    # Predict BMI
    bmi, category = predictor.predict_bmi(image_path=image_path)
    
    if bmi is None:
        print("Error: Could not process the image")
        return None, None
    
    print(f"Predicted BMI: {bmi:.2f}")
    print(f"Category: {category}")
    
    return bmi, category

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Face to BMI Prediction')
    parser.add_argument('--mode', type=str, choices=['train', 'predict'], required=True,
                        help='Mode: train model or predict BMI')
    parser.add_argument('--image', type=str, help='Path to image for prediction (required in predict mode)')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        success = train_model()
        if success:
            print("Model training completed successfully!")
        else:
            print("Model training failed.")
            
    elif args.mode == 'predict':
        if not args.image:
            print("Error: --image argument is required in predict mode")
            return
            
        if not os.path.exists(args.image):
            print(f"Error: Image not found at {args.image}")
            return
            
        predict_bmi(args.image)

if __name__ == "__main__":
    main()