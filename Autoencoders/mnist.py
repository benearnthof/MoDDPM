"""
Interactive code segments for Regression & Autoencoders
"""

import torch
from torch import nn
import numpy as np
from matplotlib import pyplot as plt

from pathlib import Path
from torch.utils.data import Dataset
import torchvision.transforms as T
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
import functools
from tqdm import trange
from PIL import Image

# Basic Autoencoder on MNIST
# Basic Convolutional Autoencoder on Oxford Flowers
# Variational Autoencoder on Oxford Flowers & Faces Dataset

device = torch.device("cuda")

mnist_transform = T.Compose([
    T.ToTensor()
])

ds = MNIST(
    root="/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ru25jan4/data",
    download=True,
    transform = mnist_transform
)

ds_test = MNIST(
    root="/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ru25jan4/data",
    download=True,
    transform = mnist_transform,
    train=False
)


dl_test = DataLoader(ds_test, batch_size=len(ds_test), shuffle=False)
x_test, y_test = next(iter(dl_test))

def cycle(dl):
    while True:
        for batch in dl:
            yield batch


class Autoencoder(nn.Module):
    """
    Basic Linear Autoencoder that only uses fully connected layers & ReLU activation functions.
    """
    def __init__(self, in_channels, latent_dim=64):
        super(Autoencoder, self).__init__()
        self.encode = nn.Sequential(
            nn.Linear(in_channels, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, latent_dim),
        )
        self.decode = nn.Sequential(
            nn.BatchNorm1d(latent_dim),
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, in_channels)
        )
    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x



test_accs = []
weight_count = sum(t.numel() for t in model.parameters())
lossfunction = nn.MSELoss()
len(optimizer.state_dict()["param_groups"][0]["params"]), weight_count

def train_step():
    optimizer.zero_grad()
    samples, _ = next(dl)
    samples = samples.reshape([samples.shape[0], 28*28])
    samples = samples.to(device)
    pred = model(samples)
    loss = lossfunction(pred, samples)
    loss.backward()
    optimizer.step()
    return loss

x_test = x_test.reshape([x_test.shape[0], 28*28]).to(device)

# how to evaluate test performance? 
# test loss
def val_step(model):
    with torch.no_grad():
        reconstructions = model(x_test.to(device))
        pred_loss = lossfunction(reconstructions, x_test)
    return pred_loss

model.to(device)
model.train()

results = {}
batch_sizes = [2**x for x in range(1, 10)]
batch_sizes = [1024]
for b in batch_sizes:
    print(b)
    dl = DataLoader(ds, batch_size=b, shuffle=True, drop_last=True)
    dl = cycle(dl)
    model = Autoencoder(in_channels=28*28)
    model.to(device)
    optimizer = Adam(model.parameters(), lr=3e-4)
    model.train()
    # yields list of two tensors with batch of images at 0 and batch of labels at 1
    pred_loss = float("nan")
    test_accs = []
    for i in (t:=trange(10000)):
        loss = train_step()
        if i%10 == 9:
            pred_loss = val_step(model).detach().cpu().numpy()
            torch.cuda.empty_cache()
        test_accs.append(pred_loss)
        t.set_description(f"loss: {loss.item():6.2f}  val_loss: {pred_loss:5.2f}")
    results[b] = test_accs

from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection

fig, ax1 = plt.subplots(figsize=(12,6))
plt.yscale('log')
# ax2 = ax1.twinx()
for bs, test_accs in zip(results.keys(), results.values()):
    ax1.plot(test_accs, label=f"{bs}")

ax1.set_ylabel("Validation Loss")
plt.legend()
fig.savefig("mnist.png")



class ImageDataset(Dataset):
        def __init__(
        self,
        folder: str | Path,
        image_size: int,
        exts: List[str] = ['jpg', 'jpeg', 'png', 'tiff'],
        augment_horizontal_flip = False,
        convert_image_to = None
    ):
        super().__init__()
        if isinstance(folder, str):
            folder = Path(folder)

        assert folder.is_dir()

        self.folder = folder
        self.image_size = image_size

        self.paths = [p for ext in exts for p in folder.glob(f'**/*.{ext}')]

        def convert_image_to_fn(img_type, image):
            if image.mode == img_type:
                return image

            return image.convert(img_type)

        maybe_convert_fn = partial(convert_image_to_fn, convert_image_to) if exists(convert_image_to) else nn.Identity()

        self.transform = T.Compose([
            T.Lambda(maybe_convert_fn),
            T.Resize(image_size),
            T.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
            T.CenterCrop(image_size),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        return self.transform(img)


