"""
Convolutional Autoencoder on Celeba
"""

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import trange, tqdm
from MoDDPM.Autoencoders.datasets import ImageDataset

from matplotlib import pyplot as plt
import os
import shutil
from random import sample

device = torch.device("cuda")

# CELEBA contains over 200k images

# We train on CELEBA
CELEBA_PATH = "/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ru25jan4/data/CelebA/celeba/img_align_celeba/"
# Test on a subset of 20k images
TEST_PATH = "/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ru25jan4/data/CelebA/celeba/20k_test/"
# And validate our model on 70k FFHQ images
VAL_PATH = "/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ru25jan4/data/ffhq256/ffhq256/"


# First we sample 20k random images from CELEBA
# files = os.listdir(TEST_PATH)
# test_files = sample(files, 4218)
# # nove them to test directory so we can invoke Dataset constructor
# for file in tqdm(test_files):
#     shutil.move(
#          TEST_PATH + file, # to new subdir
#          CELEBA_PATH + file # from
#     )

image_size = 256

train_ds = ImageDataset(folder=CELEBA_PATH, image_size=image_size)
test_ds = ImageDataset(folder=TEST_PATH, image_size=image_size)
val_ds = ImageDataset(folder=VAL_PATH, image_size=image_size)


# input tensor to help us with finding the right number of channels
img = torch.randn([1, 3, 256, 256]).to(device)
img2 = torch.rand([1, 3, 256, 256]).to(device)

class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder,self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=5, stride=2, padding=2), # 8, 128, 128
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=5, stride=2, padding=2), # 16, 64, 64
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2), # 32, 32, 32
            nn.ReLU(),
            )
        self.decoder = nn.Sequential(             
            nn.ConvTranspose2d(32, 16, kernel_size=5, stride=2, padding=2, output_padding=1), # 16, 64, 64
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=5, stride=2, padding=2, output_padding=1), # 8, 128, 128
            nn.ReLU(),
            nn.ConvTranspose2d(8, 3, kernel_size=5, stride=2, padding=2, output_padding=1), # 3, 256, 256
            nn.ReLU(),
            )
    def forward(self,x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# lets compare the amount of parameters in each model:
# TODO make training a callable function in this file from MoDDPM.Autoencoders.mnist import Autoencoder
# conv_model = ConvAutoencoder()
# mnist_model = Autoencoder(28*28)
# naive_model = Autoencoder(3*256*256)

# n_params_conv = sum(t.numel() for t in conv_model.parameters())
# n_params_mnist = sum(t.numel() for t in mnist_model.parameters())
# n_params_naive = sum(t.numel() for t in naive_model.parameters())
# # That is an insane reduction of trainable parameters, maybe we need more layers for 256x256 images

def cycle(dl):
    while True:
        for batch in dl:
            yield batch

import lpips
lossfunction = nn.MSELoss().to(device)
lpips_loss_function = lpips.LPIPS(net="alex").to(device)

def train_step(model, lambda_lpips=0.1):
    optimizer.zero_grad()
    samples = next(train_dl).to(device)
    pred = model(samples)
    mse_loss = lossfunction(pred, samples)
    lpips_loss = torch.mean(lpips_loss_function(pred, samples))
    loss = mse_loss + lambda_lpips * lpips_loss
    loss.backward()
    optimizer.step()
    return loss


# how to evaluate test performance?
# test loss
def test_step(model, lambda_lpips=0.1):
    with torch.no_grad():
        sample = next(test_dl).to(device)
        reconstructions = model(sample)
        pred_mse = lossfunction(reconstructions, sample)
        lpips_loss = torch.mean(lpips_loss_function(reconstructions, sample))
        pred_loss = pred_mse + lambda_lpips * lpips_loss
    return pred_loss


# results = {}
# batch_sizes = [2**x for x in range(1, 10)]
# batch_sizes = [1024]
# for b in batch_sizes:
#     print(b)
    # dl = DataLoader(ds, batch_size=b, shuffle=True, drop_last=True)
    # dl = cycle(dl)

train_dl = cycle(DataLoader(train_ds, batch_size=32, shuffle=True))
test_dl = cycle(DataLoader(test_ds, batch_size=256, shuffle=True))
val_dl = DataLoader(val_ds, batch_size=8, shuffle=False)

conv_model = ConvAutoencoder()
conv_model.to(device)
optimizer = Adam(conv_model.parameters(), lr=3e-4)
conv_model.train()
# yields list of two tensors with batch of images at 0 and batch of labels at 1
pred_loss, val_loss = float("nan"), float("nan") 
test_losses, val_losses = [], []
for i in (t := trange(10000)):
    loss = train_step(conv_model)
    if i % 10 == 9:
        pred_loss = test_step(conv_model).detach().cpu().numpy()
        torch.cuda.empty_cache()
    test_losses.append(pred_loss)
    t.set_description(f"loss: {loss.item():6.4f}  val_loss: {pred_loss:5.4f}")
# results[b] = test_losses
conv_model.eval()
for x in tqdm(val_dl):
    with torch.no_grad():
        val_losses.append(
            lossfunction(conv_model(x.to(device)), x.to(device)).item()
        )

# TODO: KDE plot of values


from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection

fig, ax1 = plt.subplots(figsize=(12, 6))
plt.yscale("log")
# ax2 = ax1.twinx()
#for bs, test_accs in zip(results.keys(), results.values()):
ax1.plot(test_losses) #, label=f"{bs}")

ax1.set_ylabel("Validation Loss")
plt.legend()
fig.savefig(
    "/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ru25jan4/gitroot/MoDDPM/MoDDPM/assets/celeba_lpips.png"
)

# sampling an image and seeing its reconstruction
from torchvision.utils import save_image
from einops import rearrange

val_images = next(iter(val_dl))

plot_images = rearrange(val_images, "b c h w -> c h (b w)")
save_image(plot_images, "/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ru25jan4/gitroot/MoDDPM/MoDDPM/assets/val_image_lpips.png")
# pass this through the encoder decoder steps then plot again
recon_images = conv_model(val_images.to(device))

plot_images = rearrange(recon_images, "b c h w -> c h (b w)")
save_image(plot_images, "/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ru25jan4/gitroot/MoDDPM/MoDDPM/assets/val_recon_lpips.png")

