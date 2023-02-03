# encoding: utf-8
"""
@author: Yuan Mengwen
@contact: mwyuan94@gmail.com
@project: PyCharm
@filename: eth_mnist_int.py
@time:2023/1/6 22:06
@description:
"""

import os

os.chdir("../../")

import spaic
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from spaic.Learning.Learner import Learner
from torchvision import transforms
from torchvision.datasets import MNIST
from tqdm import tqdm

from spaic.Example.evaluation_utils import all_activity, proportion_weighting, assign_labels
from time import time as t
from spaic.Example.plot_utils import (
    plot_input,
    plot_spikes,
    plot_weights,
    plot_assignments,
    plot_performance,
    plot_voltages,
    get_square_weights,
    get_square_assignments,
)

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--n_neurons", type=int, default=100)
parser.add_argument("--n_epochs", type=int, default=1)
parser.add_argument("--n_test", type=int, default=10000)
parser.add_argument("--n_train", type=int, default=60000)
parser.add_argument("--n_workers", type=int, default=-1)
parser.add_argument("--time", type=int, default=250)
parser.add_argument("--dt", type=int, default=1)
parser.add_argument("--intensity", type=float, default=0.128)
parser.add_argument("--progress_interval", type=int, default=10)
parser.add_argument("--update_interval", type=int, default=250)
parser.add_argument("--train", dest="train", action="store_true")
parser.add_argument("--test", dest="train", action="store_false")
parser.add_argument("--plot", dest="plot", action="store_true")
parser.add_argument("--gpu", dest="gpu", action="store_true")

parser.set_defaults(plot=False)

args = parser.parse_args()
seed = args.seed
n_neurons = args.n_neurons
n_epochs = args.n_epochs
n_test = args.n_test
n_train = args.n_train
n_workers = args.n_workers
time = args.time
dt = args.dt
intensity = args.intensity
progress_interval = args.progress_interval
update_interval = args.update_interval
train = args.train
plot = args.plot
# gpu = args.gpu
gpu = False

# # Sets up Gpu use
# if torch.cuda.is_available():
#     device = torch.device("cuda")
#     torch.cuda.manual_seed_all(seed)
# else:
torch.manual_seed(seed)
device = torch.device('cpu' if not gpu else 'cuda')

torch.set_num_threads(os.cpu_count() - 1)
print("Running on Device = ", device)

# Determines number of workers to use
if n_workers == -1:
    n_workers = 0  # gpu * 4 * torch.cuda.device_count()

if not train:
    update_interval = n_test

n_sqrt = int(np.ceil(np.sqrt(n_neurons)))

n_inpt = 784
wmax_initial = 33
w_exc_inh = 2
w_inh_exc = -32767
wmin = 0
wmax = 109
backend = spaic.Torch_Backend(device)
backend.dt = dt
backend.data_type = torch.int32

root = './spaic/Datasets/MNIST'


class DiehlAndCook2015Int(spaic.Network):
    def __init__(self):
        super(DiehlAndCook2015Int, self).__init__()
        # coding
        self.input = spaic.Encoder(num=n_inpt, coding_method='bernoulli', unit_conversion=intensity)

        # neuron group
        self.exc_layer = spaic.NeuronGroup(n_neurons, model='diehlAndCook')
        self.inh_layer = spaic.NeuronGroup(n_neurons, model='lifint')

        # decoding
        # self.output = spaic.Decoder(num=n_neurons, dec_target=self.exc_layer, coding_method='spike_counts')

        # Connection
        self.connection1 = spaic.Connection(self.input, self.exc_layer, link_type='full',
                                            # weight=torch.load(r'D:\Projects\SPAIC\spaic\Example\w.pt').contiguous(),
                                            weight=np.random.randint(0, wmax_initial, (n_neurons, n_inpt)),
                                            w_min=wmin, w_max=wmax, is_parameter=False)
        # self.connection2 = spaic.Connection(self.exc_layer, self.inh_layer, link_type='full',
        #                                     weight=w_exc_inh * torch.ones(n_neurons, n_neurons, dtype=torch.int32),
        #                                     # weight=(np.diag(np.ones(n_neurons))) * w_exc_inh,
        #                                     w_min=w_exc_inh, w_max=w_exc_inh, is_parameter=False)
        # self.connection3 = spaic.Connection(self.inh_layer, self.exc_layer, link_type='full',
        #                                     # weight=w_inh_exc*(np.ones((n_neurons, n_neurons)) - np.diag(np.ones(n_neurons))),
        #                                     weight=w_inh_exc * (
        #                                             -torch.ones(n_neurons, n_neurons, dtype=torch.int32)
        #                                     ),
        #                                     w_min=w_inh_exc, w_max=w_inh_exc, is_parameter=False)
        self.connection2 = spaic.Connection(self.exc_layer, self.inh_layer, link_type='full',
                                            weight=np.diag(np.ones(n_neurons)) * w_exc_inh,
                                            w_min=0, w_max=w_exc_inh, is_parameter=False)
        self.connection3 = spaic.Connection(self.inh_layer, self.exc_layer, link_type='full',
                                            weight=w_inh_exc*(np.ones((n_neurons, n_neurons)) - np.diag(np.ones(n_neurons))),
                                            w_min=w_inh_exc, w_max=0, is_parameter=False)

        # Learner
        self._learner = Learner(algorithm='postpreint', trainable=self.connection1)

        # Minitor
        self.mon_weight = spaic.StateMonitor(self.connection1, 'weight', nbatch=-1)
        # # Voltage recording for excitatory and inhibitory layers.
        self.exc_voltage = spaic.StateMonitor(self.exc_layer, 'V', nbatch=-1)
        self.inh_voltage = spaic.StateMonitor(self.inh_layer, 'V', nbatch=-1)
        #
        # # Adaptive firing threshold recording for excitatory and inhibitory layers.
        self.exc_threshold = spaic.StateMonitor(self.exc_layer, 'thresh', nbatch=-1)
        #
        # # Set up monitors for spikes and voltages
        self.input_spike = spaic.StateMonitor(self.input, 'O', nbatch=-1)
        self.exc_spike = spaic.StateMonitor(self.exc_layer, 'O', nbatch=-1)
        self.inh_spike = spaic.StateMonitor(self.inh_layer, 'O', nbatch=-1)
        self.set_backend(backend)


network = DiehlAndCook2015Int()

# Load MNIST data.
train_dataset = MNIST(
    root=root,
    download=True,
    train=True,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Lambda(lambda x: x * intensity)]
    ),
)

# Record spikes during the simulation.
spike_record = torch.zeros((update_interval, int(time / dt), n_neurons), device=device)

# Neuron assignments and spike proportions.
n_classes = 10
assignments = -torch.ones(n_neurons, device=device)
proportions = torch.zeros((n_neurons, n_classes), device=device)
rates = torch.zeros((n_neurons, n_classes), device=device)

# Sequence of accuracy estimates.
accuracy = {"all": [], "proportion": []}

inpt_ims, inpt_axes = None, None
spike_ims, spike_axes = None, None
weights_im = None
assigns_im = None
perf_ax = None
voltage_axes, voltage_ims = None, None
adaptive_threshold_axes, adaptive_threshold_ims = None, None

# Train the network.
print("\nBegin training.\n")
start = t()
for epoch in range(n_epochs):
    labels = []

    if epoch % progress_interval == 0:
        print("Progress: %d / %d (%.4f seconds)" % (epoch, n_epochs, t() - start))
        start = t()

    # Create a dataloader to iterate and batch data
    dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, shuffle=True, num_workers=n_workers, pin_memory=gpu
    )

    for step, batch in enumerate(tqdm(dataloader)):
        if step > n_train:
            break
        # Get next input sample.
        inputs, label = batch
        inputs = inputs.view(1, -1)
        # inputs = {"X": batch["encoded_image"].view(int(time / dt), 1, 1, 28, 28)}
        # if gpu:
        #     inputs = {k: v.cuda() for k, v in inputs.items()}

        if step % update_interval == 0 and step > 0:
            # Convert the array of labels into a tensor
            label_tensor = torch.tensor(labels, device=device)

            # Get network predictions.
            all_activity_pred = all_activity(
                spikes=spike_record, assignments=assignments, n_labels=n_classes
            )
            proportion_pred = proportion_weighting(
                spikes=spike_record,
                assignments=assignments,
                proportions=proportions,
                n_labels=n_classes,
            )

            # Compute network accuracy according to available classification strategies.
            accuracy["all"].append(
                100
                * torch.sum(label_tensor.long() == all_activity_pred).item()
                / len(label_tensor)
            )
            accuracy["proportion"].append(
                100
                * torch.sum(label_tensor.long() == proportion_pred).item()
                / len(label_tensor)
            )

            print(
                "\nAll activity accuracy: %.2f (last), %.2f (average), %.2f (best)"
                % (
                    accuracy["all"][-1],
                    np.mean(accuracy["all"]),
                    np.max(accuracy["all"]),
                )
            )
            print(
                "Proportion weighting accuracy: %.2f (last), %.2f (average), %.2f"
                " (best)\n"
                % (
                    accuracy["proportion"][-1],
                    np.mean(accuracy["proportion"]),
                    np.max(accuracy["proportion"]),
                )
            )

            # Assign labels to excitatory layer neurons.
            assignments, proportions, rates = assign_labels(
                spikes=spike_record,
                labels=label_tensor,
                n_labels=n_classes,
                rates=rates,
            )

            labels = []

        # labels.append(batch["label"])
        labels.append(label)

        # Run the network on the input.
        network.input(inputs)
        network.run(time)

        # Get voltage recording.
        exc_voltages = torch.tensor(network.exc_voltage.values).permute(2, 0, 1)
        inh_voltages = torch.tensor(network.inh_voltage.values).permute(2, 0, 1)

        # Get adaptive threshold recording.
        exc_threshold = torch.tensor(network.exc_threshold.values).permute(2, 0, 1).squeeze()
        inh_threshold = torch.zeros_like(exc_threshold)

        # Get firing.
        exc_spike = torch.tensor(network.exc_spike.values, device=device, dtype=torch.int32).permute(2, 0, 1)
        inh_spike = torch.tensor(network.inh_spike.values, device=device, dtype=torch.int32).permute(2, 0, 1)
        input_spike = torch.tensor(network.input_spike.values, device=device, dtype=torch.int32).permute(2, 0, 1).view(
            250, 1, 1, 28, 28)

        # Add to spikes recording.
        spike_record[step % update_interval] = exc_spike.squeeze()  # spikes["Ae"].get("s").squeeze()
        #
        # Optionally plot various simulation information.
        if plot:
            inputs = torch.load(r'D:\Projects\SPAIC\spaic\Example\image.pt')
            image = inputs.view(28, 28)
            inpt = network.input.all_spikes.view(time, 784).sum(0).view(28, 28)
            input_exc_weights = network.connection1.weight
            square_weights = get_square_weights(
                input_exc_weights.view(784, n_neurons), n_sqrt, 28
            )
            square_assignments = get_square_assignments(assignments, n_sqrt)
            spikes_ = {"Ae": exc_spike, "X": input_spike, "Ai": inh_spike}
            voltages = {"Ae": exc_voltages, "Ai": inh_voltages}
            thresholds = {"Ae": exc_threshold, "Ai": inh_threshold}
            # inpt_axes, inpt_ims = plot_input(
            #     image, inpt, label=label, axes=inpt_axes, ims=inpt_ims
            # )
            spike_ims, spike_axes = plot_spikes(spikes_, ims=spike_ims, axes=spike_axes)
            weights_im = plot_weights(square_weights, im=weights_im, wmin=wmin, wmax=wmax)
            # assigns_im = plot_assignments(square_assignments, im=assigns_im)
            # perf_ax = plot_performance(accuracy, x_scale=update_interval, ax=perf_ax)
            voltage_ims, voltage_axes = plot_voltages(
                voltages, ims=voltage_ims, axes=voltage_axes, plot_type="line"
            )
            # adaptive_threshold_ims, adaptive_threshold_axes = plot_voltages(
            #     thresholds, ims=adaptive_threshold_ims, axes=adaptive_threshold_axes, plot_type="line"
            # )

            plt.pause(1e-8)
        #
        # network.reset_state_variables()  # Reset state variables.

print("Progress: %d / %d (%.4f seconds)" % (epoch + 1, n_epochs, t() - start))
print("Training complete.\n")

# Load MNIST data.
test_dataset = MNIST(
    root=root,
    download=True,
    train=False,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Lambda(lambda x: x * intensity)]
    ),
)

# Sequence of accuracy estimates.
accuracy = {"all": 0, "proportion": 0}

# Record spikes during the simulation.
spike_record = torch.zeros((1, int(time / dt), n_neurons), device=device)

# Train the network.
print("\nBegin testing\n")
network.train(mode=False)
start = t()

pbar = tqdm(total=n_test)
for step, batch in enumerate(test_dataset):
    if step >= n_test:
        break
    # Get next input sample.
    inputs, label = batch
    inputs = inputs.view(1, -1)
    # inputs = {"X": batch["encoded_image"].view(int(time / dt), 1, 1, 28, 28)}
    # if gpu:
    #     inputs = {k: v.cuda() for k, v in inputs.items()}

    # Run the network on the input.
    network.input(inputs)
    network.run(time)

    # Add to spikes recording.
    spikes = torch.tensor(network.exc_spike.values, device=device, dtype=torch.int32)
    spike_record[0] = spikes.permute(2, 0, 1).squeeze()  # spikes["Ae"].get("s").squeeze()

    # Convert the array of labels into a tensor
    label_tensor = torch.tensor(label, device=device)

    # Get network predictions.
    all_activity_pred = all_activity(
        spikes=spike_record, assignments=assignments, n_labels=n_classes
    )
    proportion_pred = proportion_weighting(
        spikes=spike_record,
        assignments=assignments,
        proportions=proportions,
        n_labels=n_classes,
    )

    # Compute network accuracy according to available classification strategies.
    accuracy["all"] += float(torch.sum(label_tensor.long() == all_activity_pred).item())
    accuracy["proportion"] += float(
        torch.sum(label_tensor.long() == proportion_pred).item()
    )

    # network.reset_state_variables()  # Reset state variables.
    pbar.set_description_str("Test progress: ")
    pbar.update()

print("\nAll activity accuracy: %.2f" % (accuracy["all"] / n_test))
print("Proportion weighting accuracy: %.2f \n" % (accuracy["proportion"] / n_test))

print("Progress: %d / %d (%.4f seconds)" % (epoch + 1, n_epochs, t() - start))
print("Testing complete.\n")
