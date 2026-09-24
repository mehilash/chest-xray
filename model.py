# ============================================================
# CHEST X-RAY CLASSIFICATION MODEL
# ============================================================
#
# This file contains the SAME model architecture that was used
# in the original Chest_X_ray.ipynb training notebook.
#
# Classes:
# 0 -> Normal
# 1 -> Pneumonia
# 2 -> Tuberculosis
# ============================================================


# Import PyTorch.
import torch


# Import PyTorch neural-network components.
import torch.nn as nn


# Import pretrained computer-vision models.
from torchvision import models


# ============================================================
# CREATE CHEST X-RAY MODEL
# ============================================================

class ChestXRayModel(nn.Module):
    # Creates the DenseNet121 model used for
    # Normal / Pneumonia / Tuberculosis classification.

    def __init__(
        self,
        num_classes=3,
        dropout_rate=0.3
    ):
        # num_classes = 3 because we have three classes.
        #
        # dropout_rate = 0.3 because this is the dropout
        # configuration used by the trained model.

        # Initialize the parent PyTorch neural-network class.
        super(ChestXRayModel, self).__init__()


        # ====================================================
        # LOAD DENSENET121
        # ====================================================

        self.backbone = models.densenet121(
            weights=models.DenseNet121_Weights.DEFAULT
        )

        # Loads the pretrained DenseNet121 architecture.
        #
        # The pretrained ImageNet weights provide the initial
        # feature-extraction layers used during transfer learning.


        # ====================================================
        # GET DENSENET FEATURE SIZE
        # ====================================================

        num_features = self.backbone.classifier.in_features

        # DenseNet121 produces 1024 features before its
        # original classification layer.
        #
        # Therefore:
        #
        # num_features = 1024


        # ====================================================
        # REPLACE ORIGINAL CLASSIFIER
        # ====================================================

        self.backbone.classifier = nn.Sequential(

            # ------------------------------------------------
            # Batch Normalization
            # ------------------------------------------------

            nn.BatchNorm1d(
                num_features
            ),

            # Normalizes the 1024 extracted DenseNet features.


            # ------------------------------------------------
            # Fully Connected Layer
            # ------------------------------------------------

            nn.Linear(
                num_features,
                512
            ),

            # Converts:
            #
            # 1024 features
            #
            # into:
            #
            # 512 features


            # ------------------------------------------------
            # ReLU Activation
            # ------------------------------------------------

            nn.ReLU(),

            # Adds a non-linear activation function.


            # ------------------------------------------------
            # Dropout
            # ------------------------------------------------

            nn.Dropout(
                p=dropout_rate
            ),

            # Randomly disables 30% of neurons during training.
            #
            # This also allows Monte Carlo Dropout to be used
            # later for uncertainty estimation.


            # ------------------------------------------------
            # Final Classification Layer
            # ------------------------------------------------

            nn.Linear(
                512,
                num_classes
            )

            # Produces three output values:
            #
            # output 0 -> Normal
            # output 1 -> Pneumonia
            # output 2 -> Tuberculosis
        )


    # ========================================================
    # FORWARD PASS
    # ========================================================

    def forward(self, x):
        # Receives a preprocessed chest X-ray image.

        # Pass the image through DenseNet121 and
        # the custom classification head.
        x = self.backbone(x)

        # Return the three classification scores.
        return x