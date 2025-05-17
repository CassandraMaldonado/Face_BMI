# Face to BMI 

import os
import argparse
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import traceback

# Importing our BMI predictor
from bmi_prediction import BMIPredictor

# Relative paths
DATA_PATH = "data.csv"
IMAGE_PATH = "Images"
MODEL_PATH = "bmi_predictor_model.h5"
HISTORY_PLOT_PATH = "training_history.png"

def train_model():
    try:
        # BMI predictor
        predictor = BMIPredictor()
        
        # Dataset
        X, y = predictor.load_data(DATA_PATH, IMAGE_PATH)
        
        if len(X) == 0:
            print("Error: No valid images were loaded. Check your data paths.")
            return False
        
        print(f"Successfully loaded {len(X)} images with corresponding BMI values")
        
        print("Creating the model...")
        model = predictor.create_model()
        if model is None:
            print("Error: Failed to create model")
            return False
        
        # Model summary
        print("Model created successfully. Summary:")
        model.summary()
        
        # Checking the directory
        model_dir = os.path.dirname(os.path.abspath(MODEL_PATH))
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            print(f"Created directory: {model_dir}")
        
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
        
        # Model
        print("Training the model...")
        history = predictor.train(
            X, y, 
            epochs=50, 
            batch_size=4,  
            validation_split=0.2,
            callbacks=[checkpoint, early_stopping]
        )
        
        # Saving the model
        print(f"Attempting to save model to {MODEL_PATH}...")
        predictor.save_model(MODEL_PATH)
        
        # Verifying the model was saved
        if os.path.exists(MODEL_PATH):
            file_size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
            print(f"Model successfully saved! File size: {file_size_mb:.2f} MB")
        else:
            print(f"Warning: Model file not found at {MODEL_PATH} after saving")
        
        # History plots
        predictor.save_training_plot(history, HISTORY_PLOT_PATH)
        print(f"Training history plot saved to {HISTORY_PLOT_PATH}")
        
        return True
    
    except Exception as e:
        print(f"Error during model training: {str(e)}")
        traceback.print_exc()  
        return False

def predict_bmi(image_path):
    try:
        if not os.path.exists(MODEL_PATH):
            print(f"Error: Model not found at {MODEL_PATH}. Train the model first.")
            return None, None
        
        # Generated a predictor with the model
        print(f"Loading model from {MODEL_PATH}...")
        predictor = BMIPredictor(MODEL_PATH)
        
        # Predicting the BMI
        bmi, category = predictor.predict_bmi(image_path=image_path)
        
        if bmi is None:
            print("Error: Could not process the image")
            return None, None
        
        print(f"Predicted BMI: {bmi:.2f}")
        print(f"Category: {category}")
        
        return bmi, category
    
    except Exception as e:
        print(f"Error during BMI prediction: {str(e)}")
        traceback.print_exc()  
        return None, None

def main():
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