import os

import torch
import torch.nn as nn
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
            nn.Conv2d(ndf * 32, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input).view(-1, 1)


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
    print(device)
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

    for epoch in range(num_epochs):
        for i, imgs in enumerate(dataloader):
            imgs = imgs.to(device)

            # --------------------
            # Train Discriminator
            # --------------------
            D.zero_grad()
            b_size = imgs.size(0)

            labels = torch.full((b_size,), real_label, device=device)
            # Real images
            output_real = D(imgs)
            loss_real = criterion(output_real.view(-1).squeeze(), labels)

            # Fake images
            noise = torch.randn(b_size, latent_dim, 1, 1, device=device)
            fake_imgs = G(noise)
            labels_fake = torch.full((b_size,), fake_label, device=device)
            output_fake = D(fake_imgs.detach())
            loss_fake = criterion(output_fake.view(-1).squeeze(), labels_fake)

            # Total discriminator loss
            loss_D = loss_real + loss_fake
            loss_D.backward()
            optimizer_D.step()

            # --------------------
            # Train Generator
            # --------------------
            G.zero_grad()
            labels.fill_(real_label)  # generator tries to fool D
            output = D(fake_imgs)
            loss_G = criterion(output.squeeze(), labels)
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
