import os

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import Dataset, Subset
from torchvision import transforms
from torchvision.utils import save_image
import torch.autograd as autograd


class CelebADataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.files = [f for f in os.listdir(root) if f.endswith(".jpg")]
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.files[idx])
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img


class Generator(nn.Module):
    def __init__(self, nz=100, ngf=1024, nc=3):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # Input: Z latent vector (nz) going into a conv transpose
            nn.ConvTranspose2d(nz, ngf, 4, 1, 0, bias=False),  # 1x1 -> 4x4
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf, ngf // 2, 4, 2, 1, bias=False),  # 4x4 -> 8x8
            nn.BatchNorm2d(ngf // 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf // 2, ngf // 4, 4, 2, 1, bias=False),  # 8x8 -> 16x16
            nn.BatchNorm2d(ngf // 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf // 4, ngf // 8, 4, 2, 1, bias=False),  # 16x16 -> 32x32
            nn.BatchNorm2d(ngf // 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf // 8, ngf // 16, 4, 2, 1, bias=False),  # 32x32 -> 64x64
            nn.BatchNorm2d(ngf // 16),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf // 16, ngf // 32, 4, 2, 1, bias=False),  # 64x64 -> 128x128
            nn.BatchNorm2d(ngf // 32),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf // 32, nc, 4, 2, 1, bias=False),  # 128x128 -> 256x256
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self, nc=3, ndf=64):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # nc x 256 x 256 -> ndf x 128 x 128
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # -> (ndf*2) x 64 x 64
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # -> (ndf*4) x 32 x 32
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # -> (ndf*8) x 16 x 16
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # -> (ndf*16) x 8 x 8
            nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),

            # -> (ndf*32) x 4 x 4
            nn.Conv2d(ndf * 16, ndf * 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 32),
            nn.LeakyReLU(0.2, inplace=True),

            # -> 1 x 1 x 1
            nn.Conv2d(ndf * 32, 1, 4, 1, 0, bias=False)
        )

    def forward(self, input):
        return self.main(input).view(-1, 1)


def gradient_penalty(D, real_imgs, fake_imgs, device):
    batch_size = real_imgs.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)
    interpolates = (alpha * real_imgs + (1 - alpha) * fake_imgs).requires_grad_(True)

    d_interpolates = D(interpolates)
    fake = torch.ones(d_interpolates.size(), device=device, requires_grad=False)

    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    gradients = gradients.view(batch_size, -1)
    gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gp


if __name__ == '__main__':
    # Transform: convert to tensor + normalize [-1,1]
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ])

    dataset = CelebADataset("img_align_celeba/", transform=transform)
    dataset_small = Subset(dataset, range(30000))
    dataloader = torch.utils.data.DataLoader(dataset_small, batch_size=16, shuffle=True, num_workers=4)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.autograd.set_detect_anomaly(True)
    latent_dim = 100

    G = Generator(latent_dim).to(device)
    D = Discriminator().to(device)

    # summary(G, (4, 100, 1, 1))

    criterion = nn.BCELoss()
    optimizer_G = optim.Adam(G.parameters(), lr=0.0001, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))

    real_label = 0.9
    fake_label = 0.

    os.makedirs("generated_celebs", exist_ok=True)
    num_epochs = 30
    lambda_gp = 10
    n_critic = 5

    for epoch in range(num_epochs):
        for i, imgs in enumerate(dataloader):
            imgs = imgs.to(device)
            b_size = imgs.size(0)
            noise = torch.randn(b_size, latent_dim, 1, 1, device=device)

            # Discriminator
            optimizer_D.zero_grad()
            fake_imgs = G(noise).detach()
            loss_D = -torch.mean(D(imgs)) + torch.mean(D(fake_imgs))

            gp = gradient_penalty(D, imgs, fake_imgs, device)
            loss_D += lambda_gp * gp

            # --- R1 regularization ---
            gamma_r1 = 10
            imgs.requires_grad_()
            real_preds = D(imgs)
            grad_real = torch.autograd.grad(
                outputs=real_preds.sum(),
                inputs=imgs,
                create_graph=True
            )[0]
            r1 = (grad_real.pow(2).view(imgs.size(0), -1).sum(1).mean()) * (gamma_r1 / 2)
            loss_D += r1
            imgs.requires_grad_(False)

            # backward and step
            loss_D.backward()
            optimizer_D.step()

            # Generator
            optimizer_G.zero_grad()
            fake_imgs = G(noise)
            loss_G = -torch.mean(D(fake_imgs))
            loss_G.backward()
            optimizer_G.step()

            if i % 100 == 0:
                print(f"[Epoch {epoch}/{num_epochs}] [Batch {i}/{len(dataloader)}] "
                      f"Loss D: {loss_D.item():.4f}, loss G: {loss_G.item():.4f}")

        # Save sample images every epoch
        with torch.no_grad():
            fake_sample = G(torch.randn(16, latent_dim, 1, 1, device=device))
            fake_sample = (fake_sample + 1) / 2  # [-1,1] -> [0,1]
            save_image(fake_sample, f"generated_celebs/epoch_{epoch}.png", nrow=4)
