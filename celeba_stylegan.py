"""
Summary:
1. This implementation does not have a progressive growing due to time complexity for local training;
2. This implementation based on StyleGan1 papers;
3. This implementation should train for an appropriate time
"""

import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.utils.data import Dataset, Subset
from torchvision import transforms
from torchvision.utils import save_image


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


def accumulate(model1, model2, decay=0.999):
    """
    model1: EMA model
    model2: generator
    decay: EMA
    """
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)

    buf1 = dict(model1.named_buffers())
    buf2 = dict(model2.named_buffers())
    for k in buf1.keys():
        buf1[k].data.copy_(buf2[k].data)


class MappingNetwork(nn.Module):
    def __init__(self, z_dim=512, w_dim=512, num_layers=8):
        super().__init__()
        layers = []
        for i in range(num_layers):
            in_dim = z_dim if i == 0 else w_dim
            layers.append(nn.Linear(in_dim, w_dim))
            layers.append(nn.LeakyReLU(0.2))
        self.mapping = nn.Sequential(*layers)

    def forward(self, z):
        z = z / (z.norm(dim=1, keepdim=True) + 1e-8)
        return self.mapping(z)


def style_mixing(mapping_network, z1, z2, num_blocks, prob=0.9):
    w1 = mapping_network(z1)
    w2 = mapping_network(z2)

    if random.random() > prob:
        return [w1.clone() for _ in range(num_blocks)]

    cutoff = random.randint(1, num_blocks - 1)
    return [w1.clone() for _ in range(cutoff)] + [w2.clone() for _ in range(num_blocks - cutoff)]


# -----------------------
# Adaptive Instance Norm
# -----------------------
class AdaIN(nn.Module):
    def __init__(self, channels, dlatent_dim=512):
        super().__init__()
        self.norm = nn.InstanceNorm2d(channels)
        self.style_scale = nn.Linear(dlatent_dim, channels)
        self.style_bias = nn.Linear(dlatent_dim, channels)

    def forward(self, x, w):
        x = self.norm(x)
        style_scale = self.style_scale(w).unsqueeze(2).unsqueeze(3)
        style_bias = self.style_bias(w).unsqueeze(2).unsqueeze(3)
        return style_scale * x + style_bias


# -----------------------
# Styled Conv Block
# -----------------------
class StyledConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dlatent_dim=512):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.adain = AdaIN(out_channels, dlatent_dim)
        self.noise_strength = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x, w, noise=None):
        x = self.conv(x)
        if noise is None:
            noise = torch.randn(x.size(0), 1, x.size(2), x.size(3), device=x.device)
        x = x + self.noise_strength.view(1, -1, 1, 1) * noise
        x = self.adain(x, w)
        return F.leaky_relu(x, 0.2)


# -----------------------
# Generator (fixed 128x128)
# -----------------------
class Generator(nn.Module):
    def __init__(self, dlatent_dim=512, base_channels=64):
        super().__init__()
        self.const_input = nn.Parameter(torch.randn(1, base_channels * 16, 4, 4))

        self.blocks = nn.ModuleList([
            StyledConvBlock(base_channels * 16, base_channels * 16, dlatent_dim),
            StyledConvBlock(base_channels * 16, base_channels * 8, dlatent_dim),
            StyledConvBlock(base_channels * 8, base_channels * 4, dlatent_dim),
            StyledConvBlock(base_channels * 4, base_channels * 2, dlatent_dim),
            StyledConvBlock(base_channels * 2, base_channels, dlatent_dim),
        ])

        self.to_rgb = nn.Conv2d(base_channels, 3, 1)

    def forward(self, w_list, noise_list=None):
        batch_size = w_list[0].size(0)
        x = self.const_input.repeat(batch_size, 1, 1, 1)
        if noise_list is None:
            noise_list = [None] * len(self.blocks)

        for block, w, noise in zip(self.blocks, w_list, noise_list):
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            x = block(x, w, noise)

        return torch.tanh(self.to_rgb(x))


# -----------------------
# Discriminator (128x128)
# -----------------------
class Discriminator(nn.Module):
    def __init__(self, channels=64):
        super().__init__()

        self.convs = nn.Sequential(
            nn.Conv2d(3, channels, 3, stride=2, padding=1),  # 128→64
            nn.LeakyReLU(0.2),

            nn.Conv2d(channels, channels * 2, 3, stride=2, padding=1),  # 64→32
            nn.LeakyReLU(0.2),

            nn.Conv2d(channels * 2, channels * 4, 3, stride=2, padding=1),  # 32→16
            nn.LeakyReLU(0.2),

            nn.Conv2d(channels * 4, channels * 8, 3, stride=2, padding=1),  # 16→8
            nn.LeakyReLU(0.2),

            nn.Conv2d(channels * 8, channels * 16, 3, stride=2, padding=1),  # 8→4
            nn.LeakyReLU(0.2),
        )

        self.fc = nn.Linear(channels * 16 * 4 * 4, 1)

    def forward(self, img):
        x = self.convs(img)
        x = x.view(x.size(0), -1)
        return self.fc(x)


if __name__ == '__main__':
    # ==========================
    # Training Loop
    # ==========================
    latent_dim = 512
    batch_size = 16
    num_epochs = 50
    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    dataset = CelebADataset("img_align_celeba/", transform=transform)
    dataset_small = Subset(dataset, range(40000))
    dataloader = torch.utils.data.DataLoader(dataset_small, batch_size=32, shuffle=True, num_workers=4,
                                             pin_memory=(device == "cuda"))

    mapping = MappingNetwork(z_dim=latent_dim, w_dim=512).to(device)
    G = Generator(latent_dim).to(device)
    G_ema = Generator(latent_dim).to(device)
    G_ema.load_state_dict(G.state_dict())
    D = Discriminator().to(device)

    optimizer_G = optim.Adam(G.parameters(), lr=0.0002, betas=(0.0, 0.99))
    optimizer_D = optim.Adam(D.parameters(), lr=0.0001, betas=(0.0, 0.99))

    fixed_z = torch.randn(16, latent_dim, device=device)

    for epoch in range(num_epochs):
        for i, imgs in enumerate(dataloader):
            real_imgs = imgs.to(device)
            b_size = real_imgs.size(0)

            # --------------------
            # Train Discriminator
            # --------------------
            z1 = torch.randn(b_size, latent_dim, device=device)
            z2 = torch.randn(b_size, latent_dim, device=device)
            styles = style_mixing(mapping, z1, z2, num_blocks=len(G.blocks))

            fake_imgs = G(styles).detach()

            real_labels = torch.full((b_size, 1), 0.9, device=device).float()
            fake_labels = torch.full((b_size, 1), 0, device=device).float()
            real_labels_G = torch.ones((b_size, 1), device=device)

            output_real = D(real_imgs)
            output_fake = D(fake_imgs)

            loss_real = F.softplus(-output_real).mean()
            loss_fake = F.softplus(output_fake).mean()
            loss_D = loss_real + loss_fake

            # R1 Regularization

            r1_every = 16
            gamma = 10

            if i % r1_every == 0:
                real_imgs.requires_grad_(True)
                output_real_reg = D(real_imgs)
                grad_real = torch.autograd.grad(outputs=output_real_reg.sum(),
                                                inputs=real_imgs, create_graph=True)[0]
                grad_penalty = grad_real.pow(2).view(b_size, -1).sum(1).mean()
                r1_loss = (gamma / 2) * grad_penalty * r1_every
            else:
                r1_loss = torch.zeros((), device=device)

            loss_D = loss_D + r1_loss

            optimizer_D.zero_grad()
            loss_D.backward()
            optimizer_D.step()

            real_imgs.requires_grad_(False)

            # --------------------
            # Train Generator
            # --------------------
            g_steps = 2 if epoch > 2 else 1

            for _ in range(g_steps):
                z1 = torch.randn(b_size, latent_dim, device=device)
                z2 = torch.randn(b_size, latent_dim, device=device)

                styles = style_mixing(mapping, z1, z2, num_blocks=len(G.blocks))
                gen_imgs = G(styles)

                output_gen = D(gen_imgs)
                loss_G = F.softplus(-output_gen).mean()

                optimizer_G.zero_grad()
                loss_G.backward()
                optimizer_G.step()
                accumulate(G_ema, G, decay=0.999)

            if i % 100 == 0:
                print(f"[Epoch {epoch}/{num_epochs}] [Batch {i}/{len(dataloader)}] "
                      f"Loss D: {loss_D.item():.4f}, Loss G: {loss_G.item():.4f}")

        # --------------------
        # Save sample images
        # --------------------
        G_ema.eval()
        with torch.no_grad():
            w_fixed = mapping(fixed_z)
            samples = G_ema([w_fixed for _ in range(len(G.blocks))])
            samples = (samples + 1) / 2
            save_image(samples, f"generated_celebs/epoch_{epoch}.png", nrow=4)
        G_ema.train()
