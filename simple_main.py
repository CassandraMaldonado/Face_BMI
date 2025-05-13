# Face to BMI - Simple Main Script (No arguments required)
# Based on the paper: https://cdn.aaai.org/ojs/14923/14923-28-18442-1-2-20201228.pdf

import os
import sys
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

def display_menu():
    print("\n===== Face to BMI Prediction =====")
    print("1. Train model")
    print("2. Predict BMI from image")
    print("0. Exit")
    choice = input("Enter your choice (0-2): ")
    return choice

def create_model(self):
    # VGG Face model with the pre-trained weights
    base_model = get_vgg_face_model(include_top=False, input_shape=(224, 224, 3))
    
    # We are freezing the base model layers
    for layer in base_model.layers:
        layer.trainable = False
    
    # Custom regression layers
    x = base_model.output
    
    # Add a unique name to the Flatten layer to avoid conflicts
    x = Flatten(name='flatten_regression')(x)
    x = Dense(512, activation='relu', name='dense_regression_1')(x)
    x = Dropout(0.5, name='dropout_regression_1')(x)
    x = Dense(128, activation='relu', name='dense_regression_2')(x)
    x = Dropout(0.5, name='dropout_regression_2')(x)
    predictions = Dense(1, activation='linear', name='bmi_output')(x)  # BMI is a continuous value
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    
    self.model = model
    return model

def main():
    while True:
        choice = display_menu()
        
        if choice == '0':
            print("Exiting...")
            break
            
        elif choice == '1':
            print("\n--- Training Model ---")
            success = train_model()
            if success:
                print("Model training completed successfully!")
            else:
                print("Model training failed.")
            
        elif choice == '2':
            print("\n--- Predict BMI from Image ---")
            if not os.path.exists(MODEL_PATH):
                print(f"Error: Model not found at {MODEL_PATH}. Train the model first.")
                continue
                
            image_path = input("Enter the path to the image file: ")
            
            if not os.path.exists(image_path):
                print(f"Error: Image not found at {image_path}")
                continue
                
            predict_bmi(image_path)
            
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
