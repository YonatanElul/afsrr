from torch import Tensor
from afsrr.utils.defaults import OTHER_KEY
from afsrr.models.midst import MIDST, get_activation_layer

import torch
import torch.nn as nn


class JointRegressionModel(nn.Module):
    def __init__(
            self,
            dynamics_model_params: dict,
            n_classes: int = 2,
            n_classification_layers: int = 3,
            classifier_activation: str = 'leakyrelu',
            classifier_units: int = 256,
    ):
        super().__init__()
        self.midst = MIDST(
            **dynamics_model_params
        )

        svd_rank = dynamics_model_params['observable_dim']
        layers = [
            nn.Linear(
                in_features=svd_rank,
                out_features=classifier_units,
                bias=True,
            ),
            get_activation_layer(classifier_activation)
        ]
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

        layers.append(
            nn.Linear(
                in_features=classifier_units,
                out_features=n_classes,
                bias=True,
            ),
        )
        self.classifier = nn.Sequential(*layers)

    def __call__(self, x: Tensor):
        return self.forward(x=x)

    def forward(self, x: Tensor):
        out_midst = self.midst(x)
        dynamics = out_midst[OTHER_KEY]['dynamics'][0]
        singular_values = torch.linalg.svdvals(dynamics)
        classification = self.classifier(singular_values)
        out_midst[OTHER_KEY]['sv_class'] = classification

        return out_midst
