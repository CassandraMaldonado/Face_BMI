# Face to BMI

import os
import argparse
import pandas as pd  # Added this import for the test function
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Our BMI predictor
from bmi_prediction import BMIPredictor

# Define your specific paths
DATA_PATH = "/Users/casey/Documents/GitHub/Face_to_BMI/data.csv"
IMAGE_PATH = "/Users/casey/Documents/GitHub/Face_to_BMI/Images"
MODEL_PATH = "/Users/casey/Documents/GitHub/Face_to_BMI/bmi_predictor_model.h5"
HISTORY_PLOT_PATH = "/Users/casey/Documents/GitHub/Face_to_BMI/training_history.png"

def train_model():
    predictor = BMIPredictor()
    
    X, y = predictor.load_data(DATA_PATH, IMAGE_PATH)
    
    if len(X) == 0:
        print("Error: No images were loaded.")
        return False
    
    print(f"Successfully loaded {len(X)} images with corresponding BMI values")
    
    # Callbacks for training
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
    
    # Create the model
    model = predictor.create_model()
    history = predictor.train(
        X, y, 
        epochs=50, 
        batch_size=4, 
        validation_split=0.2,
        callbacks=[checkpoint, early_stopping]
    )
    
    predictor.save_model(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    
    # Training history
    predictor.save_training_plot(history, HISTORY_PLOT_PATH)
    print(f"Training history plot saved to {HISTORY_PLOT_PATH}")
    
    return True

def predict_bmi(image_path):
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}.")
        return None, None
    
    predictor = BMIPredictor(MODEL_PATH)
    
    # Predicting BMI
    bmi, category = predictor.predict_bmi(image_path=image_path)
    
    if bmi is None:
        print("Error: Could not process the image")
        return None, None
    
    print(f"Predicted BMI: {bmi:.2f}")
    print(f"Category: {category}")
    
    return bmi, category

def test_model():
    """Test the BMI prediction model using test images from the dataset"""
    print("Testing model on test images...")
    
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}. Train the model first.")
        return False
    
    # Predictor with the trained model
    predictor = BMIPredictor(MODEL_PATH)
    
    # Load the full dataset
    df = pd.read_csv(DATA_PATH)
    
    # Subset for testing with 20% of the data
    test_df = df.sample(frac=0.2, random_state=42)
    print(f"Selected {len(test_df)} images for testing")
    
    results = []
    
    # Testing each image
    for idx, row in test_df.iterrows():
        image_path = os.path.join(IMAGE_PATH, row['name'])
        
        if not os.path.exists(image_path):
            print(f"Warning: Image {image_path} not found")
            continue
        
        # Actual BMI
        actual_bmi = row['bmi']
        
        # BMI prediction
        predicted_bmi, category = predictor.predict_bmi(image_path=image_path)
        
        if predicted_bmi is not None:
            # Error
            error = abs(predicted_bmi - actual_bmi)
            
            results.append({
                'image': row['name'],
                'actual_bmi': actual_bmi,
                'predicted_bmi': predicted_bmi,
                'category': category,
                'error': error
            })
            
            print(f"Image: {row['name']}")
            print(f"  Actual BMI: {actual_bmi:.2f}")
            print(f"  Predicted BMI: {predicted_bmi:.2f}")
            print(f"  Error: {error:.2f}")
            print(f"  Category: {category}")
            print("---")
    
    # Metrics
    if results:
        errors = [r['error'] for r in results]
        mean_error = sum(errors) / len(errors)
        max_error = max(errors)
        min_error = min(errors)
        
        print("\nTest Results Summary:")
        print(f"Number of test images: {len(results)}")
        print(f"Mean Absolute Error: {mean_error:.2f}")
        print(f"Max Error: {max_error:.2f}")
        print(f"Min Error: {min_error:.2f}")
        
        # Save results to CSV
        results_df = pd.DataFrame(results)
        results_df.to_csv("test_results.csv", index=False)
        print("Detailed results saved to test_results.csv")
    
    return True

def main():
    # Argument
    parser = argparse.ArgumentParser(description='Face to BMI Prediction')
    parser.add_argument('--mode', type=str, choices=['train', 'predict', 'test'], required=True,
                        help='Mode: train model, predict BMI, or test model')
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
            print("Error: Image is required in predict mode")
            return
            
        if not os.path.exists(args.image):
            print(f"Error: Image not found at {args.image}")
            return
            
        predict_bmi(args.image)
    
    elif args.mode == 'test':
        test_model()

if __name__ == "__main__":
    main()