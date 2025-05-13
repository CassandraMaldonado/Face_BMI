# Download VGG-Face Model Weights Script

import os
import sys
import tensorflow as tf
from tensorflow.keras.utils import get_file

def download_vggface_weights():
    """
    Download VGG-Face model weights and print their location
    """
    print("=== VGG-Face Model Weights Downloader ===")
    
    # URLs for the VGG-Face model weights
    WEIGHTS_PATH = 'https://github.com/rcmalli/keras-vggface/releases/download/v2.0/rcmalli_vggface_tf_vgg16.h5'
    WEIGHTS_PATH_NO_TOP = 'https://github.com/rcmalli/keras-vggface/releases/download/v2.0/rcmalli_vggface_tf_notop_vgg16.h5'
    
    # Get Keras cache directory
    keras_dir = os.path.expanduser(os.path.join('~', '.keras'))
    models_dir = os.path.join(keras_dir, 'models')
    
    print(f"Keras cache directory: {keras_dir}")
    print(f"Keras models directory: {models_dir}")
    
    # Download the weights
    print("\nDownloading VGG-Face model with top layers...")
    try:
        weights_path = get_file('rcmalli_vggface_tf_vgg16.h5',
                                WEIGHTS_PATH,
                                cache_subdir='models')
        print(f"Full model downloaded to: {weights_path}")
    except Exception as e:
        print(f"Error downloading full model: {str(e)}")
    
    print("\nDownloading VGG-Face model without top layers...")
    try:
        weights_path_no_top = get_file('rcmalli_vggface_tf_notop_vgg16.h5',
                                      WEIGHTS_PATH_NO_TOP,
                                      cache_subdir='models')
        print(f"Feature extraction model downloaded to: {weights_path_no_top}")
    except Exception as e:
        print(f"Error downloading feature extraction model: {str(e)}")
    
    # Check if files exist
    print("\nVerifying downloaded files:")
    if os.path.exists(os.path.join(models_dir, 'rcmalli_vggface_tf_vgg16.h5')):
        print("✓ Full model weights file exists")
    else:
        print("✗ Full model weights file NOT found")
    
    if os.path.exists(os.path.join(models_dir, 'rcmalli_vggface_tf_notop_vgg16.h5')):
        print("✓ Feature extraction model weights file exists")
    else:
        print("✗ Feature extraction model weights file NOT found")
    
    print("\nNOTE: These weights are used by your VGG-Face implementation in vgg_face.py")
    print("      They are NOT the same as your trained BMI prediction model (bmi_predictor_model.h5)")
    print("      Your BMI predictor model builds on top of these weights")

def check_bmi_predictor_model():
    """
    Check if the BMI predictor model has been trained and saved
    """
    print("\n=== BMI Predictor Model Check ===")
    
    # Current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Potential model locations
    model_paths = [
        os.path.join(current_dir, "bmi_predictor_model.h5"),
        os.path.join(current_dir, "..", "bmi_predictor_model.h5"),
        os.path.join(current_dir, "model", "bmi_predictor_model.h5"),
        os.path.join(current_dir, "models", "bmi_predictor_model.h5"),
    ]
    
    model_found = False
    for path in model_paths:
        if os.path.exists(path):
            print(f"✓ BMI predictor model found at: {path}")
            model_found = True
            break
    
    if not model_found:
        print("✗ BMI predictor model not found in common locations")
        print("\nYou need to train your BMI predictor model first. Here's how:")
        print("1. Make sure your data.csv and Images folder are in place")
        print("2. Run your training script:")
        print("   python main.py")
        print("   OR")
        print("   python simple_main.py (and select the train option)")
        print("\nAfter training, the model should be saved as bmi_predictor_model.h5")

def check_keras_imports():
    """
    Check if necessary Keras imports are working
    """
    print("\n=== Keras VGG-Face Import Check ===")
    
    try:
        # Try to import your custom implementation
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from vgg_face import get_vgg_face_model, preprocess_input
        print("✓ Successfully imported vgg_face module")
        
        # Try to create a model (without loading weights)
        model = get_vgg_face_model(include_top=False, weights=None)
        print("✓ Successfully created VGG-Face model architecture")
        
    except ImportError as e:
        print(f"✗ Import error: {str(e)}")
        print("  Make sure vgg_face.py is in the same directory as this script")
    
    except Exception as e:
        print(f"✗ Error creating model: {str(e)}")
        print("  This might indicate an issue with TensorFlow/Keras")

if __name__ == "__main__":
    download_vggface_weights()
    check_bmi_predictor_model()
    check_keras_imports()
    
    print("\n=== Summary ===")
    print("1. The VGG-Face base model weights should be downloaded to your Keras cache directory")
    print("2. Your BMI predictor model builds on top of this base model")
    print("3. You need to train your BMI predictor model using main.py or simple_main.py")
    print("4. After training, your model will be saved as bmi_predictor_model.h5")
    print("\nIf you've already trained your model but it's not being found,")
    print("make sure the model file is in the correct location and has the correct name.")