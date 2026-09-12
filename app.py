from flask import Flask, request, jsonify, render_template
# Flask is used to create the web backend and API.

import torch
# PyTorch is used to load and run the trained deep learning model.

import torch.nn.functional as F
# F.softmax() is used to convert model outputs into class probabilities.
from PIL import Image # PIL is used to open uploaded X-ray images.

from torchvision import transforms
# torchvision transforms are used for image preprocessing.

import io
# io is used to read the uploaded image from memory.

import numpy as np
# NumPy is used for numerical calculations.

import os
# os is used for handling file paths.


# ---------------------------------------------------------
# CREATE FLASK APPLICATION
# ---------------------------------------------------------

app = Flask(__name__)
# Creates the Flask application.


# ---------------------------------------------------------
# DEVICE CONFIGURATION
# ---------------------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Uses the GPU if CUDA is available; otherwise, uses the CPU.


# ---------------------------------------------------------
# CLASS NAMES
# ---------------------------------------------------------

CLASS_NAMES = {
    0: "normal",
    1: "pneumonia",
    2: "tuberculosis"
}
# These class numbers must match the labels used during training.


# ---------------------------------------------------------
# MODEL CHECKPOINT PATH
# ---------------------------------------------------------

MODEL_PATH = "best_chest_xray_model (2).pth"
# Specifies the exact filename of your trained model.


# ---------------------------------------------------------
# IMPORT YOUR MODEL ARCHITECTURE
# ---------------------------------------------------------

from model import ChestXRayModel
# Imports the same model architecture that was used during training.


# ---------------------------------------------------------
# CREATE MODEL
# ---------------------------------------------------------

model = ChestXRayModel(
    num_classes=3,
    dropout_rate=0.5
)
# Creates the DenseNet121 model with 3 output classes.
# IMPORTANT: dropout_rate must be the same value used during training.


# ---------------------------------------------------------
# LOAD TRAINED MODEL
# ---------------------------------------------------------

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)
# Loads the trained model checkpoint from the .pth file.


model.load_state_dict(
    checkpoint["model_state_dict"]
)
# Loads the learned weights into the model.


model = model.to(DEVICE)
# Moves the model to GPU or CPU.


model.eval()
# Puts the model into evaluation mode for prediction.


# ---------------------------------------------------------
# IMAGE PREPROCESSING
# ---------------------------------------------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    # Resizes the X-ray image to 224 x 224 pixels.

    transforms.Grayscale(num_output_channels=3),
    # Converts grayscale X-ray images into 3 channels
    # because DenseNet121 expects 3-channel input.

    transforms.ToTensor(),
    # Converts the image into a PyTorch tensor.

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    # Normalizes the image using ImageNet normalization.
])
# IMPORTANT: preprocessing should match the preprocessing used during training.


# ---------------------------------------------------------
# MC DROPOUT PREDICTION
# ---------------------------------------------------------

def mc_dropout_predict(image_tensor, num_samples=20):
    """
    Performs multiple stochastic predictions using
    Monte Carlo Dropout to estimate uncertainty.
    """

    model.train()
    # Enables Dropout layers so each prediction is slightly different.
    # BatchNorm behavior can also change here, so if your training setup
    # requires keeping BatchNorm in eval mode, adjust this accordingly.

    predictions = []
    # Creates an empty list to store probability predictions.


    with torch.no_grad():
        # Gradients are not required during prediction.

        for _ in range(num_samples):
            # Performs multiple forward passes.

            outputs = model(image_tensor)
            # Sends the X-ray image through the trained model.

            probabilities = F.softmax(outputs, dim=1)
            # Converts raw model outputs into probabilities.

            predictions.append(
                probabilities.cpu().numpy()
            )
            # Saves the probabilities for this prediction.


    model.eval()
    # Returns the model to normal evaluation mode.


    predictions = np.array(predictions)
    # Converts the list of predictions into a NumPy array.


    mean_probabilities = np.mean(
        predictions,
        axis=0
    )
    # Calculates the average probability across all MC Dropout predictions.


    prediction_variation = np.std(
        predictions,
        axis=0
    )
    # Calculates how much the predictions vary.


    predicted_class = np.argmax(
        mean_probabilities,
        axis=1
    )[0]
    # Selects the class with the highest average probability.


    confidence = float(
        mean_probabilities[0][predicted_class]
    )
    # Gets the probability of the predicted class.


    uncertainty = float(
        np.mean(prediction_variation[0])
    )
    # Calculates a simple uncertainty score from prediction variation.


    return (
        predicted_class,
        confidence,
        uncertainty,
        mean_probabilities[0]
    )
    # Returns the prediction and confidence/uncertainty information.


# ---------------------------------------------------------
# TRIAGE RECOMMENDATION
# ---------------------------------------------------------

def get_triage_recommendation(confidence, uncertainty):
    """
    Converts confidence and uncertainty into a simple
    prototype triage recommendation.
    """

    if confidence >= 0.80 and uncertainty < 0.10:
        # High confidence and low uncertainty.

        return "High Confidence - Screening Result"

    elif confidence >= 0.60 and uncertainty < 0.20:
        # Moderate confidence and moderate uncertainty.

        return "Radiologist Review"

    else:
        # Low confidence or high uncertainty.

        return "Specialist Review / Further Assessment"


# ---------------------------------------------------------
# HOME PAGE
# ---------------------------------------------------------

@app.route("/")
def home():
    # This route displays the frontend page.

    return render_template("index.html")
    # Loads templates/index.html.


# ---------------------------------------------------------
# PREDICTION API
# ---------------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():
    # This API receives an uploaded X-ray and returns prediction results.

    try:

        # -------------------------------------------------
        # CHECK WHETHER IMAGE WAS UPLOADED
        # -------------------------------------------------

        if "image" not in request.files:
            # Checks whether the frontend sent an image.

            return jsonify({
                "error": "No image uploaded."
            }), 400
            # Returns an error if no image was provided.


        file = request.files["image"]
        # Gets the uploaded image file.


        if file.filename == "":
            # Checks whether the filename is empty.

            return jsonify({
                "error": "No image selected."
            }), 400


        # -------------------------------------------------
        # OPEN IMAGE
        # -------------------------------------------------

        image = Image.open(
            io.BytesIO(file.read())
        )
        # Reads the uploaded image into memory and opens it with PIL.


        image = image.convert("RGB")
        # Converts the image to RGB format.


        # -------------------------------------------------
        # PREPROCESS IMAGE
        # -------------------------------------------------

        input_tensor = transform(image)
        # Applies resizing, grayscale conversion, tensor conversion
        # and normalization.


        input_tensor = input_tensor.unsqueeze(0)
        # Adds the batch dimension.
        # Shape changes from [3, 224, 224] to [1, 3, 224, 224].


        input_tensor = input_tensor.to(DEVICE)
        # Moves the image tensor to GPU or CPU.


        # -------------------------------------------------
        # RUN MC DROPOUT
        # -------------------------------------------------

        (
            predicted_class,
            confidence,
            uncertainty,
            probabilities
        ) = mc_dropout_predict(
            input_tensor,
            num_samples=20
        )
        # Performs 20 stochastic predictions.


        # -------------------------------------------------
        # GET CLASS NAME
        # -------------------------------------------------

        predicted_label = CLASS_NAMES[
            int(predicted_class)
        ]
        # Converts the predicted class number into its class name.


        # -------------------------------------------------
        # GET PROBABILITIES
        # -------------------------------------------------

        class_probabilities = {
            CLASS_NAMES[i]: float(probabilities[i])
            for i in range(len(CLASS_NAMES))
        }
        # Creates a readable probability dictionary for all classes.


        # -------------------------------------------------
        # GET TRIAGE RESULT
        # -------------------------------------------------

        triage_result = get_triage_recommendation(
            confidence,
            uncertainty
        )
        # Generates the prototype triage recommendation.


        # -------------------------------------------------
        # RETURN RESULT
        # -------------------------------------------------

        return jsonify({

            "prediction": predicted_label,
            # Predicted disease/class.

            "confidence": round(confidence, 4),
            # Confidence of the predicted class.

            "uncertainty": round(uncertainty, 4),
            # Estimated prediction uncertainty.

            "probabilities": class_probabilities,
            # Probability for Normal, Pneumonia and Tuberculosis.

            "triage": triage_result
            # Prototype triage recommendation.

        })


    except Exception as e:
        # Handles unexpected errors.

        return jsonify({
            "error": str(e)
        }), 500
        # Returns the error to the frontend.


# ---------------------------------------------------------
# RUN FLASK SERVER
# ---------------------------------------------------------

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )
    # Starts the Flask development server on port 5000.