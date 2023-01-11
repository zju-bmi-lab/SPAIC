# encoding: utf-8
"""
@author: Yuan Mengwen
@contact: mwyuan94@gmail.com
@project: PyCharm
@filename: evaluation_utils.py
@time:2023/1/11 12:42
@description:
"""
import torch
from typing import Tuple, Optional

def all_activity(
    spikes: torch.Tensor, assignments: torch.Tensor, n_labels: int
) -> torch.Tensor:
    # language=rst
    """
    Classify data with the label with highest average spiking activity over all neurons.

    :param spikes: Binary tensor of shape ``(n_samples, time, n_neurons)`` of a layer's
        spiking activity.
    :param assignments: A vector of shape ``(n_neurons,)`` of neuron label assignments.
    :param n_labels: The number of target labels in the data.
    :return: Predictions tensor of shape ``(n_samples,)`` resulting from the "all
        activity" classification scheme.
    """
    n_samples = spikes.size(0)

    # Sum over time dimension (spike ordering doesn't matter).
    spikes = spikes.sum(1)

    rates = torch.zeros((n_samples, n_labels), device=spikes.device)
    for i in range(n_labels):
        # Count the number of neurons with this label assignment.
        n_assigns = torch.sum(assignments == i).float()

        if n_assigns > 0:
            # Get indices of samples with this label.
            indices = torch.nonzero(assignments == i).view(-1)

            # Compute layer-wise firing rate for this label.
            rates[:, i] = torch.sum(spikes[:, indices], 1) / n_assigns

    # Predictions are arg-max of layer-wise firing rates.
    return torch.sort(rates, dim=1, descending=True)[1][:, 0]


def proportion_weighting(
    spikes: torch.Tensor,
    assignments: torch.Tensor,
    proportions: torch.Tensor,
    n_labels: int,
) -> torch.Tensor:
    # language=rst
    """
    Classify data with the label with highest average spiking activity over all neurons,
    weighted by class-wise proportion.

    :param spikes: Binary tensor of shape ``(n_samples, time, n_neurons)`` of a single
        layer's spiking activity.
    :param assignments: A vector of shape ``(n_neurons,)`` of neuron label assignments.
    :param proportions: A matrix of shape ``(n_neurons, n_labels)`` giving the per-class
        proportions of neuron spiking activity.
    :param n_labels: The number of target labels in the data.
    :return: Predictions tensor of shape ``(n_samples,)`` resulting from the "proportion
        weighting" classification scheme.
    """
    n_samples = spikes.size(0)

    # Sum over time dimension (spike ordering doesn't matter).
    spikes = spikes.sum(1)

    rates = torch.zeros((n_samples, n_labels), device=spikes.device)
    for i in range(n_labels):
        # Count the number of neurons with this label assignment.
        n_assigns = torch.sum(assignments == i).float()

        if n_assigns > 0:
            # Get indices of samples with this label.
            indices = torch.nonzero(assignments == i).view(-1)

            # Compute layer-wise firing rate for this label.
            rates[:, i] += (
                torch.sum((proportions[:, i] * spikes)[:, indices], 1) / n_assigns
            )

    # Predictions are arg-max of layer-wise firing rates.
    predictions = torch.sort(rates, dim=1, descending=True)[1][:, 0]

    return predictions

def assign_labels(
    spikes: torch.Tensor,
    labels: torch.Tensor,
    n_labels: int,
    rates: Optional[torch.Tensor] = None,
    alpha: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # language=rst
    """
    Assign labels to the neurons based on highest average spiking activity.

    :param spikes: Binary tensor of shape ``(n_samples, time, n_neurons)`` of a single
        layer's spiking activity.
    :param labels: Vector of shape ``(n_samples,)`` with data labels corresponding to
        spiking activity.
    :param n_labels: The number of target labels in the data.
    :param rates: If passed, these represent spike rates from a previous
        ``assign_labels()`` call.
    :param alpha: Rate of decay of label assignments.
    :return: Tuple of class assignments, per-class spike proportions, and per-class
        firing rates.
    """
    n_neurons = spikes.size(2)

    if rates is None:
        rates = torch.zeros((n_neurons, n_labels), device=spikes.device)

    # Sum over time dimension (spike ordering doesn't matter).
    spikes = spikes.sum(1)

    for i in range(n_labels):
        # Count the number of samples with this label.
        n_labeled = torch.sum(labels == i).float()

        if n_labeled > 0:
            # Get indices of samples with this label.
            indices = torch.nonzero(labels == i).view(-1)

            # Compute average firing rates for this label.
            rates[:, i] = alpha * rates[:, i] + (
                torch.sum(spikes[indices], 0) / n_labeled
            )

    # Compute proportions of spike activity per class.
    proportions = rates / rates.sum(1, keepdim=True)
    proportions[proportions != proportions] = 0  # Set NaNs to 0

    # Neuron assignments are the labels they fire most for.
    assignments = torch.max(proportions, 1)[1]

    return assignments, proportions, rates