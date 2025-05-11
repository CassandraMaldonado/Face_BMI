# Face to BMI Guide

This project implements a BMI prediction system from facial images based on the paper: [Face to BMI: Using Computer Vision to Infer Body Mass Index on Social Media](https://cdn.aaai.org/ojs/14923/14923-28-18442-1-2-20201228.pdf)

## Project Structure

The project consists of several Python files:

1. `vgg_face.py`: Implementation of the VGG-Face model
2. `bmi_prediction.py`: BMI prediction functionality
3. `main.py`: Command-line interface for training and prediction
4. `simple_main.py`: Interactive menu-based interface 
5. `app.py`: Flask web application for BMI prediction
6. `streamlit_app.py`: Streamlit web application for BMI prediction (more user-friendly)

## Usage Options

### Interactive Menu

Run the simple_main.py script for an interactive menu-based interface:

This will display a menu with options to:
1. Train the model
2. Predict BMI from an image
0. Exit

### Command-line Arguments

Run the main.py script with appropriate command-line arguments:

To train the model:
```bash
python main.py --mode train
```

To predict BMI from an image:
```bash
python main.py --mode predict --image /path/to/your/image.jpg
```

### Web Interface with Streamlit
