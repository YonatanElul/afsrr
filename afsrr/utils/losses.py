from abc import ABC, abstractmethod
from typing import Sequence, Optional, Union, Dict
from torch import nn, Tensor, cat, from_numpy, eye, matmul
from afsrr.utils.defaults import (
    OTHER_KEY,
    GT_TENSOR_PREDICITONS_KEY,
    MODELS_TENSOR_PREDICITONS_KEY,
)

import torch
import numpy as np


class LossComponent(nn.Module, ABC):
    """
    An abstract API class for loss models
    """

    def __init__(self):
        super(LossComponent, self).__init__()

    @abstractmethod
    def forward(self, inputs: Union[Dict, Sequence]) -> Tensor:
        """
        The forward logic of the loss class.

        :param inputs: (Dict) Either a dictionary with the predictions from the forward pass and the ground truth
        outputs. The possible keys are specified by the following variables:
            MODELS_TENSOR_PREDICITONS_KEY
            MODELS_SEQUENCE_PREDICITONS_KEY
            GT_TENSOR_PREDICITONS_KEY
            GT_SEQUENCE_PREDICITONS_KEY
            GT_TENSOR_INPUTS_KEY
            GT_SEQUENCE_INPUTS_KEY
            OTHER_KEY

        Which can be found under .../DynamicalSystems/utils/defaults.py

        Or a Sequence of dicts, each where each element is a dict with the above-mentioned structure.

        :return: (Tensor) A scalar loss.
        """

        raise NotImplemented

    def __call__(self, inputs: Union[Dict, Sequence]) -> Tensor:
        return self.forward(inputs=inputs)

    def update(self, params: Dict[str, any]):
        for key in params:
            if hasattr(self, key):
                setattr(self, key, params[key])


class ModuleLoss(LossComponent):
    """
    A LossComponent which takes in a PyTorch loss Module and decompose the inputs according to the module's
    expected API.
    """

    def __init__(
            self, model: nn.Module,
            scale: float = 1.0,
            last_window_only: bool = False,
            last_window_pred_only: bool = False,
            trajectory_length: Optional[int] = None,
    ):
        """
        The constructor for the ModuleLoss class
        :param model: (PyTorch Module) The loss model, containing the computation logic.
        :param scale: (float) Scaling factor for the loss.
        """

        super().__init__()

        self.model = model
        self.scale = scale
        self._last_window_only = last_window_only
        self._last_window_pred_only = last_window_pred_only
        self._trajectory_length = trajectory_length

    def forward(self, inputs: Dict) -> Tensor:
        """
        Basically a wrapper around the forward of the inner model, which decompose the inputs to the expected
        structure expected by the PyTorch module.

        :param inputs: (dict) The outputs of the forward pass of the model along with the ground-truth labels.
        :return: (Tensor) A scalar Tensor representing the aggregated loss
        """

        y_pred = inputs[MODELS_TENSOR_PREDICITONS_KEY]
        y = inputs[GT_TENSOR_PREDICITONS_KEY]

        if self._last_window_only:
            y = y[..., -1, :]
            y_pred = y_pred[..., -1, :]

        if self._last_window_pred_only:
            y_pred = y_pred[..., (self._trajectory_length - 1)::self._trajectory_length, :]

        loss = self.scale * self.model(y_pred, y)

        return loss


class CompoundedLoss(LossComponent):
    """
    A wrapper class for handling multiple loss functions which should be applied
    together.
    """

    def __init__(
            self,
            losses: Sequence[LossComponent],
            losses_weights: Optional[Sequence[float]] = None,
            loss_inds_per_step: Optional[Sequence[Sequence[float]]] = None,
    ):
        """
        Constructor for the CompoundedLoss class.

        :param losses: (Sequence[LossComponent]) A sequence of loss
        modules to be applied.
        :param losses_weights: (Optional) A sequence of loss weights
        to be applied to each loss module.
        """

        if losses_weights is not None:
            assert len(losses) == len(losses_weights), \
                f"losses_weights should either specify a weight of type float for " \
                f"each loss in losses, or be left as None. Currently there are" \
                f"{len(losses)} and {len(losses_weights)} weights."

            self._losses_weights = losses_weights

        else:
            self._losses_weights = [1.0 for _ in range(len(losses))]

        super(CompoundedLoss, self).__init__()

        self._losses = losses
        self._loss_inds_per_step = loss_inds_per_step
        self._step = 0

    def forward(self, inputs: Union[Dict, Sequence]) -> Tensor:
        losses = [
            self._losses_weights[i] * self._losses[i](inputs)
            for i in range(len(self._losses))
            if (
                    self._loss_inds_per_step is None or
                    i in self._loss_inds_per_step[min(self._step, len(self._loss_inds_per_step) - 1)]
            )
        ]

        # Filter out nans from the loss
        losses = [
            loss
            for loss in losses
            if not torch.isnan(loss)
        ]

        # Sum all losses
        loss = sum(losses)
        self._step += 1

        return loss


class AFSRRLoss(LossComponent):
    def __init__(
            self,
            alpha: float = 1.0,
            beta: float = 1.0,
            dist_fn: nn.Module = nn.MSELoss(),
    ):
        super().__init__()
        self._alpha = alpha
        self._beta = beta

        self._dist = ModuleLoss(model=dist_fn)
        self._ce = nn.BCEWithLogitsLoss()

    def forward(self, inputs: Union[Dict, Sequence]) -> Tensor:
        midst_loss = self._dist(inputs)
        y_pred = inputs[OTHER_KEY]['sv_class']
        y = inputs['af_labels'][0, :][:, None]

        randomized_order = np.random.choice(
            a=np.arange(y.shape[0]),
            size=y.shape[0],
            replace=False,
        )
        ce_aux_loss = self._ce(y_pred[randomized_order], y[randomized_order])
        loss = (self._alpha * midst_loss) + (self._beta * ce_aux_loss)
        return loss


class ClassificationAccuracy(LossComponent):
    def __init__(
            self,
            apply_softmax: bool = True,
            pos_only: bool = False,
            multi_label: bool = True,
    ):
        super(ClassificationAccuracy, self).__init__()

        self._apply_softmax = apply_softmax
        self._multi_label = multi_label
        self._pos_only = pos_only
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, inputs: Dict) -> Tensor:
        y = inputs[GT_TENSOR_PREDICITONS_KEY]
        if self._apply_softmax:
            logits = inputs[MODELS_TENSOR_PREDICITONS_KEY]
            y_pred = self.softmax(logits)

        else:
            y_pred = inputs[MODELS_TENSOR_PREDICITONS_KEY]

        if self._multi_label:
            predictions = (y_pred > 0.5).int()

        else:
            predictions = torch.argmax(y_pred, dim=-1)

        if self._pos_only:
            accuracy = (y * (y == predictions).type(torch.float32)).mean()

        else:
            accuracy = (y == predictions).type(torch.float32).mean()

        return accuracy
