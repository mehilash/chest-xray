import torch
# PyTorch is used to build and run the deep learning model.

import torch.nn as nn
# torch.nn provides neural-network layers such as Linear, ReLU, Dropout and BatchNorm.

from torchvision import models
# torchvision provides the pretrained DenseNet121 architecture.


# ---------------------------------------------------------
# CHEST X-RAY MODEL
# ---------------------------------------------------------

class ChestXRayModel(nn.Module):
    # Defines the custom chest X-ray classification model.

    def __init__(self, num_classes=3, dropout_rate=0.5):
        # num_classes = number of output classes.
        # dropout_rate = dropout probability used to reduce overfitting.

        super(ChestXRayModel, self).__init__()
        # Initializes the parent PyTorch neural-network class.


        # -------------------------------------------------
        # LOAD PRETRAINED DENSENET121
        # -------------------------------------------------

        self.backbone = models.densenet121(
            weights=models.DenseNet121_Weights.DEFAULT
        )
        # Loads DenseNet121 with pretrained ImageNet weights.
        # The pretrained network provides useful image features.


        # -------------------------------------------------
        # GET NUMBER OF FEATURES
        # -------------------------------------------------

        num_features = self.backbone.classifier.in_features
        # Gets the number of input features going into DenseNet's
        # original classification layer.


        # -------------------------------------------------
        # CUSTOM CLASSIFICATION HEAD
        # -------------------------------------------------

        self.backbone.classifier = nn.Sequential(

            nn.Linear(num_features, 512),
            # Converts DenseNet features into 512 features.

            nn.BatchNorm1d(512),
            # Normalizes the 512 features to improve training stability.

            nn.ReLU(),
            # Adds a nonlinear activation function.

            nn.Dropout(dropout_rate),
            # Randomly disables some neurons during training
            # to reduce overfitting.

            nn.Linear(512, num_classes)
            # Produces the final output for the 3 classes:
            # 0 = Normal
            # 1 = Pneumonia
            # 2 = Tuberculosis
        )


    # ---------------------------------------------------------
    # FORWARD PASS
    # ---------------------------------------------------------

    def forward(self, x):
        # Defines how an input X-ray image passes through the model.

        x = self.backbone(x)
        # Sends the image through DenseNet121 and the custom classifier.

        return x
        # Returns the raw model outputs (logits).