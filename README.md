# Face to BMI Guide

This project implements a BMI prediction system from facial images based on the paper: [Face to BMI: Using Computer Vision to Infer Body Mass Index on Social Media](https://cdn.aaai.org/ojs/14923/14923-28-18442-1-2-20201228.pdf)

This application uses deep learning to predict Body Mass Index (BMI) from facial images. Built using a VGG-Face model with custom regression layers, it demonstrates how facial features can potentially correlate with BMI values. The app provides a user-friendly web interface through Streamlit, allowing users to train a model on their dataset and make predictions from uploaded images or webcam captures.

## Project Structure

The project consists of several Python files:

1. `vgg_face.py`: Implementation of the VGG-Face model
2. `bmi_prediction.py`: BMI prediction functionality
3. `main.py`: Command-line interface for training and prediction
4. `simple_main.py`: Interactive menu-based interface 
5. `streamlit_app.py`: Streamlit web application for BMI prediction (more user-friendly)

## How It Works
The system uses a transfer learning approach:

- Base Model: Pre-trained VGG-Face model that has learned facial feature extraction.
- Custom Layers: Additional regression layers to predict BMI from facial features.
- Training Process: Fine-tuning the regression layers while keeping the base model frozen.
