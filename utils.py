import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np

import torch
import torchaudio
import os

def plot(X, Y, figsize=(8, 3), title=None, ax=None):
    if not ax:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.plot(X, Y)
    if title:
        ax.set_title(title)

def plot_mel(mel, title="Melspectrogram", ylabel="frequency", figsize=(8,3)):
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    image = ax.imshow(librosa.power_to_db(mel, ref=np.max))
    fig.colorbar(image, ax=ax, format="%+2.0f dB")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("frame")


def plot_spectro(spectro, rate, figsize=(8, 3)):
    fig, ax = plt.subplots(1,1, figsize=figsize)
    S_dB = librosa.power_to_db(spectro, ref=np.max)
    img = librosa.display.specshow(
        S_dB,
        x_axis="time",
        y_axis="mel",
        sr = rate,
        fmax = 12000,
        ax = ax
        )
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title("mel-spectrogram")

def print_results(CATEGORY, cat_title, time_data, loss_data, accuracy_data, epochs):
    print(f"Results per {cat_title}\n")
    print("|        | Loss    | Accuracy | Time   |")
    print("|--------|---------|----------|--------|")
    for c, l, a, t in zip(CATEGORY, loss_data, accuracy_data, time_data):
        print(f"|{c:7.3f} |{l[-1]:8.3f} |{a*100:8.3f}% |{t:7.3f} |")

def plot_results(CATEGORY, cat_title, time_data, loss_data, accuracy_data, epochs):
    fig, ax = plt.subplots(3, 1, figsize=(10,16))

    for l in loss_data:
        plot(list(range(epochs)), l, ax=ax[0])
    ax[0].legend(CATEGORY)
    ax[0].set_xlabel("epoch")
    ax[0].grid(True)
    ax[0].set_ylabel("loss")
    ax[0].set_title(f"Loss curves for {cat_title}")

    plot(CATEGORY, time_data, ax=ax[1])
    ax[1].set_xlabel(cat_title)
    ax[1].set_xticks(CATEGORY)
    ax[1].grid(True)
    ax[1].set_ylabel("time to run (seconds)")
    ax[1].set_xscale("log")
    ax[1].set_title(f"time to run for each {cat_title}")

    plot(CATEGORY, accuracy_data, ax=ax[2])
    ax[2].set_xlabel(cat_title)
    ax[2].set_xticks(CATEGORY)
    ax[2].grid(True)
    ax[2].set_ylabel("accuracy")
    ax[2].set_xscale("log")
    ax[2].set_title(f"Accuracy over {cat_title}")

    pass


def label_to_index(label, labels):
    return torch.tensor(labels.index(label))

def index_to_label(index, labels):
    return labels[index]

class SpeechCommandsSubset(torchaudio.datasets.SPEECHCOMMANDS):
    """Speech command subset class based on Pytorch sample"""
    def load_subset(self, subset):
        path = os.path.join(self._path, subset + "_list.txt")

        with open(path) as f:
            return [os.path.normpath(os.path.join(self._path, l.strip())) for l in f]

    def __init__(self, root, download, subset="training", rate=None, feature=None):
        super().__init__(root=root, download=download)
        self.rate = rate

        src_path = root + "/SpeechCommands/speech_commands_v0.02"
        self._labels = [x.name for x in os.scandir(src_path) if x.is_dir() and "_" not in x.name]
        self._labels = sorted(self.labels)

        if subset in ["validation", "testing"]:
            self._walker = self.load_subset(subset)
        elif subset == "training":
            skip = set(self.load_subset("validation") + self.load_subset("testing"))
            self._walker = [f for f in self._walker if f not in skip]

        # Save spectrogram transform
        self.feature = feature

    def __getitem__(self, idx):
        waveform, rate, label, speaker, id = super().__getitem__(idx)
        padding = 16000 - waveform.shape[1]
        waveform = torch.nn.functional.pad(waveform, (0, padding))
        rate = 16000
        if self.rate:
            waveform = torchaudio.functional.resample(waveform, orig_freq=rate, new_freq=self.rate)
            rate = self.rate
        if self.feature:
            waveform = self.feature(waveform).transpose(1, 2)
        return waveform, rate, label, speaker, id

    @property
    def labels(self):
        return self._labels

class RNNCell(torch.nn.Module):
    def __init__(self, input_size, hidden_size,
                 gate_nonlinearity, update_nonlinearity,
                 num_W_matrices, num_U_matrices, num_biases,
                 wRank=None, uRank=None,
                 wSparsity=1.0, uSparsity=1.0):
        super(RNNCell, self).__init__()
        self._input_size = input_size
        self._hidden_size = hidden_size
        self._gate_nonlinearity = gate_nonlinearity
        self._update_nonlinearity = update_nonlinearity
        self._num_W_matrices = num_W_matrices
        self._num_U_matrices = num_U_matrices
        self._num_biases = num_biases
        self._num_weight_matrices = [self._num_W_matrices, self._num_U_matrices,
                                     self._num_biases]
        self._wRank = wRank
        self._uRank = uRank
        self._wSparsity = wSparsity
        self._uSparsity = uSparsity
        self.oldmats = []


    @property
    def state_size(self):
        return self._hidden_size

    @property
    def input_size(self):
        return self._input_size

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def gate_nonlinearity(self):
        return self._gate_nonlinearity

    @property
    def update_nonlinearity(self):
        return self._update_nonlinearity

    @property
    def wRank(self):
        return self._wRank

    @property
    def uRank(self):
        return self._uRank

    @property
    def num_W_matrices(self):
        return self._num_W_matrices

    @property
    def num_U_matrices(self):
        return self._num_U_matrices

    @property
    def num_weight_matrices(self):
        return self._num_weight_matrices

    @property
    def name(self):
        raise NotImplementedError()

    def forward(self, input, state):
        raise NotImplementedError()

    def getVars(self):
        raise NotImplementedError()

    def get_model_size(self):
        '''
        Function to get aimed model size
        '''
        mats = self.getVars()
        endW = self._num_W_matrices
        endU = endW + self._num_U_matrices

        totalnnz = 2  # For Zeta and Nu
        for i in range(0, endW):
            device = mats[i].device
            totalnnz += countNNZ(mats[i].cpu(), self._wSparsity)
            mats[i].to(device)
        for i in range(endW, endU):
            device = mats[i].device
            totalnnz += countNNZ(mats[i].cpu(), self._uSparsity)
            mats[i].to(device)
        for i in range(endU, len(mats)):
            device = mats[i].device
            totalnnz += countNNZ(mats[i].cpu(), False)
            mats[i].to(device)
        return totalnnz * 4

    def copy_previous_UW(self):
        mats = self.getVars()
        num_mats = self._num_W_matrices + self._num_U_matrices
        if len(self.oldmats) != num_mats:
            for i in range(num_mats):
                self.oldmats.append(torch.FloatTensor())
        for i in range(num_mats):
            self.oldmats[i] = torch.FloatTensor(mats[i].detach().clone().to(mats[i].device))

    def sparsify(self):
        mats = self.getVars()
        endW = self._num_W_matrices
        endU = endW + self._num_U_matrices
        for i in range(0, endW):
            mats[i] = hardThreshold(mats[i], self._wSparsity)
        for i in range(endW, endU):
            mats[i] = hardThreshold(mats[i], self._uSparsity)
        self.W.data.copy_(mats[0])
        self.U.data.copy_(mats[1])
        # self.copy_previous_UW()

    def sparsifyWithSupport(self):
        mats = self.getVars()
        endU = self._num_W_matrices + self._num_U_matrices
        for i in range(0, endU):
            mats[i] = supportBasedThreshold(mats[i], self.oldmats[i])
