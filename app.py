# ============================================================
# Chest X-Ray Screening System - Backend Engine
# Classes: Normal, Pneumonia, & Tuberculosis
# ============================================================

import os
import sys
import warnings

# Suppress environment and library logs
warnings.filterwarnings("ignore")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

import io
import torch
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, jsonify

# Silence hub and downloader output during weight loading
_devnull = open(os.devnull, "w")
_old_stderr = sys.stderr
sys.stderr = _devnull

import transformers
transformers.logging.set_verbosity_error()
transformers.utils.logging.disable_progress_bar()
from transformers import AutoImageProcessor, AutoModelForImageClassification

# ============================================================
# 1. CREATE FLASK APPLICATION
# ============================================================

app = Flask(__name__)

# ============================================================
# 2. DEVICE CONFIGURATION
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# 3. INITIALIZE DEEP LEARNING SCREENING ARCHITECTURE
# ============================================================

processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")

# Load Vision Models
pneumonia_model = AutoModelForImageClassification.from_pretrained("dima806/chest_xray_pneumonia_detection")
pneumonia_model.to(DEVICE)
pneumonia_model.eval()

tb_model = AutoModelForImageClassification.from_pretrained("runaksh/chest_xray_tuberculosis_detection")
tb_model.to(DEVICE)
tb_model.eval()

# Restore standard error stream
sys.stderr = _old_stderr
_devnull.close()

# ============================================================
# 4. CLINICAL TRIAGE RECOMMENDATION
# ============================================================

def get_triage_recommendation(predicted_label, confidence, uncertainty):
    clean_label = predicted_label.strip().lower()
    
    if clean_label == "normal" and confidence >= 0.70:
        return {
            "level": "Low Risk - Normal Radiograph",
            "recommendation": "No acute radiographic signs of Pneumonia or Tuberculosis detected. Routine clinical assessment."
        }
    elif clean_label == "tuberculosis":
        return {
            "level": "High Priority - Tuberculosis Screening Alert",
            "recommendation": "Radiographic findings consistent with Tuberculosis. Prompt clinical evaluation, sputum analysis, and specialist review recommended."
        }
    elif clean_label == "pneumonia":
        return {
            "level": "Urgent - Pneumonia Screening Alert",
            "recommendation": "Radiographic findings consistent with Pneumonia. Prompt clinical assessment and treatment protocol recommended."
        }
    elif confidence >= 0.60:
        return {
            "level": "Radiologist Review Required",
            "recommendation": "Moderate confidence or borderline lung opacity. Radiologist review recommended."
        }
    else:
        return {
            "level": "Specialist Review / Indeterminate",
            "recommendation": "High uncertainty detected. Supplementary imaging or clinical correlation advised."
        }

# ============================================================
# 5. ROUTES
# ============================================================

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image was uploaded."}), 400

        file = request.files["image"]
        if file.filename == "":
            return jsonify({"error": "No image was selected."}), 400

        # Open image and convert to RGB
        image = Image.open(io.BytesIO(file.read())).convert("RGB")

        # Process image input
        inputs = processor(images=image, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            # 1. Pneumonia evaluation
            p_outputs = pneumonia_model(**inputs)
            p_probs = torch.softmax(p_outputs.logits, dim=-1)[0].cpu().numpy()
            p_norm = float(p_probs[0])
            p_pneu = float(p_probs[1])

            # 2. Tuberculosis evaluation
            tb_outputs = tb_model(**inputs)
            tb_probs = torch.softmax(tb_outputs.logits, dim=-1)[0].cpu().numpy()
            tb_norm = float(tb_probs[0])
            tb_tb = float(tb_probs[1])

        # Joint Calibrated Likelihood Computation:
        prob_normal_joint = p_norm * tb_norm
        prob_pneumonia_joint = p_pneu * tb_norm
        prob_tb_joint = tb_tb * p_norm

        if p_pneu > 0.5 and tb_tb > 0.5:
            if tb_tb > p_pneu:
                prob_tb_joint = tb_tb
                prob_pneumonia_joint = p_pneu * 0.5
            else:
                prob_pneumonia_joint = p_pneu
                prob_tb_joint = tb_tb * 0.5

        raw_scores = {
            "normal": prob_normal_joint,
            "pneumonia": prob_pneumonia_joint,
            "tuberculosis": prob_tb_joint
        }

        # Normalize to sum to 100%
        total_score = sum(raw_scores.values()) + 1e-12
        normalized_probabilities = {
            "normal": float(raw_scores["normal"] / total_score),
            "pneumonia": float(raw_scores["pneumonia"] / total_score),
            "tuberculosis": float(raw_scores["tuberculosis"] / total_score)
        }

        # Argmax selection
        predicted_label = max(normalized_probabilities, key=normalized_probabilities.get)
        confidence = float(normalized_probabilities[predicted_label])

        # Normalized predictive entropy as uncertainty
        prob_array = np.array(list(normalized_probabilities.values()))
        epsilon = 1e-12
        entropy = -np.sum(prob_array * np.log(prob_array + epsilon))
        uncertainty = float(entropy / np.log(3.0))

        # Triage guidance
        triage = get_triage_recommendation(predicted_label, confidence, uncertainty)

        # Clean console summary
        print("\n" + "=" * 50)
        print(f"Prediction  : {predicted_label.upper()}")
        print(f"Confidence  : {confidence * 100:.2f}%")
        print(f"Uncertainty : {uncertainty * 100:.2f}%")
        print(f"Breakdown   : Normal: {normalized_probabilities['normal']*100:.2f}%, Pneumonia: {normalized_probabilities['pneumonia']*100:.2f}%, TB: {normalized_probabilities['tuberculosis']*100:.2f}%")
        print(f"Triage Level: {triage['level']}")
        print("=" * 50 + "\n")

        return jsonify({
            "prediction": predicted_label,
            "confidence": confidence,
            "uncertainty": uncertainty,
            "probabilities": normalized_probabilities,
            "triage": triage["level"],
            "recommendation": triage["recommendation"]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ============================================================
# 6. START SERVER
# ============================================================

if __name__ == "__main__":
    print("------------------------------------------")
    print("Chest X-Ray Screening System")
    print("Classes: Normal, Pneumonia, Tuberculosis")
    print("Deep Vision Classifier Ready.")
    print("Open this address in your browser:")
    print("http://127.0.0.1:5000")
    print("------------------------------------------")
    app.run(host="0.0.0.0", port=5000, debug=True)