from torch import Tensor
from afsrr.utils.defaults import OTHER_KEY
from afsrr.models.midst import MIDST, get_activation_layer

import torch
import torch.nn as nn
from typing import Dict, Union


class JointRegressionModel(nn.Module):
    """
    Joint model for dynamics regression and classification.
    
    This model combines a MIDST model for regression
    with a classification head that operates on the singular values of the dynamics matrix.

    Args:
        dynamics_model_params (dict): Parameters for the MIDST model.
        n_classes (int, optional): Number of output classes. Defaults to 2
        n_classification_layers (int, optional): Number of layers in classifier. Defaults to 3
        classifier_activation (str, optional): Activation function for classifier. Defaults to 'leakyrelu'
        classifier_units (int, optional): Number of hidden units in classifier. Defaults to 256
    """

    def __init__(
            self,
            dynamics_model_params: dict,
            n_classes: int = 2,
            n_classification_layers: int = 3,
            classifier_activation: str = 'leakyrelu',
            classifier_units: int = 256,
    ) -> None:
        """Initialize the joint regression and classification model."""
        super().__init__()
        
        # Initialize the MIDST model for dynamics regression
        self.midst = MIDST(
            **dynamics_model_params
        )

        # Get dimension from dynamics model parameters
        svd_rank = dynamics_model_params['observable_dim']
        
        # Build classification layers
        layers = [
            nn.Linear(
                in_features=svd_rank,
                out_features=classifier_units,
                bias=True,
            ),
            get_activation_layer(classifier_activation)
        ]
        
        # Add hidden layers
        for _ in range(n_classification_layers - 2):
            layers.append(
                nn.Linear(
                    in_features=classifier_units,
                    out_features=classifier_units,
                    bias=True,
                ),
            )
            layers.append(
                get_activation_layer(classifier_activation)
            )

        # Add output layer
        layers.append(
            nn.Linear(
                in_features=classifier_units,
                out_features=n_classes,
                bias=True,
            ),
        )
        self.classifier = nn.Sequential(*layers)

    def __call__(self, x: Tensor) -> Dict[str, Union[Tensor, Dict[str, Tensor]]]:
        """
        Forward pass alias to match functional interface.

        Args:
            x (Tensor): Input tensor of shape (batch, channels, time)

        Returns:
            Same as forward() method
        """
        return self.forward(x=x)

    def forward(self, x: Tensor) -> Dict[str, Union[Tensor, Dict[str, Tensor]]]:
        """
        Forward pass of the model.

        Args:
            x (Tensor): Input tensor of shape (batch, channels, time)

        Returns:
            Dict containing:
                - Regression outputs from MIDST model
                - OTHER_KEY containing:
                    - dynamics: Learned dynamics matrices
                    - sv_class: Classification logits for singular values
        """
        # Get MIDST outputs including dynamics matrices
        out_midst = self.midst(x)
        
        # Extract first dynamics matrix and compute singular values
        dynamics = out_midst[OTHER_KEY]['dynamics'][0]
        singular_values = torch.linalg.svdvals(dynamics)
        
        # Classify singular values
        classification = self.classifier(singular_values)
        out_midst[OTHER_KEY]['sv_class'] = classification

        return out_midst
