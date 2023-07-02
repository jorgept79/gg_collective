# Version 0.3
#
# Incluye el guardado de las perdidas en un pickle cada 5 épocas


from __future__ import print_function
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import random
import torch
import time
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import freeze_support
from torchvision.utils import save_image
import pickle
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import gc
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

if __name__ == '__main__':
    freeze_support()
    manualSeed = random.randint(1, 10000)
    #print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    dataroot = "C:/Users/VTDA/Desktop/ArteDataSet/art2/abstracto" 
    workers =8
    batch_size = 128
    image_size = 128
# Numero de canales (RBG=3)
    nc = 3

# Tamaño de z (vestor latente)
    nz = 100

# Tamaño de los mapas de características en el generador
    ngf = 32

# Tamaño de los mapas de características en el discriminador
    ndf = 32
    num_epochs = 200
    lr = 0.0002
# Hiperparámetro Beta1 para optimizadores de Adam
    beta1 = 0.5
    ngpu = 1
    dataset = dset.ImageFolder(root=dataroot,
                               transform=transforms.Compose([
                                   transforms.Resize(image_size),
                                   transforms.CenterCrop(image_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                   ]))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)

    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    real_batch = next(iter(dataloader))
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:128], padding=2, normalize=True).cpu(),(1,2,0)))
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
    class Generator(nn.Module):
        def __init__(self, ngpu):
            super(Generator, self).__init__()
            self.ngpu = ngpu
            self.main = nn.Sequential(
                nn.ConvTranspose2d( nz, ngf * 16, 4, 1, 0, bias=False),
                nn.BatchNorm2d(ngf * 16),
                nn.ReLU(True),
                nn.ConvTranspose2d( ngf * 16, ngf * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 8),
                nn.ReLU(True),
                nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(True),
                nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 2),
                nn.ReLU(True),
                nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf),
                nn.ReLU(True),
                nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
                nn.Tanh()
            )		
        def forward(self, input):          
            return self.main(input)
    netG = Generator(ngpu).to(device)
    if (device.type == 'cuda') and (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu)))
    netG.apply(weights_init)
    print(netG)
    class Discriminator(nn.Module):
        def __init__(self, ngpu):
            super(Discriminator, self).__init__()
            self.ngpu = ngpu
            self.main = nn.Sequential(
                nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 16),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ndf * 16, 1, 4, 1, 0, bias=False),
                nn.Sigmoid()
                )
        def forward(self, input):
            return self.main(input)
    netD = Discriminator(ngpu).to(device)
    if (device.type == 'cuda') and (ngpu > 1):
        netD = nn.DataParallel(netD, list(range(ngpu)))
    netD.apply(weights_init)
    print(netD)
    criterion = nn.BCELoss()
    fixed_noise = torch.randn(100, nz, 1, 1, device=device)
    real_label = 1
    fake_label = 0
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0
    print("Comienza el entrenamiento:")
    for epoch in range(num_epochs):
        gc.collect()
        for i, data in enumerate(dataloader, 0):
            netD.zero_grad()
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            output = netD(real_cpu).view(-1)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            fake = netG(noise)
            label.fill_(fake_label)
            output = netD(fake.detach()).view(-1)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()
            netG.zero_grad()
            label.fill_(real_label)  
            output = netD(fake).view(-1)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f - %s'
                      % (epoch, num_epochs, i, len(dataloader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2,time.time()))
            G_losses.append(errG.item())
            D_losses.append(errD.item())
            if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                    img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
            if epoch % 5 == 0 and i == len(dataloader)-1:            
                if not os.path.exists(f"results/epoch_{epoch}"):
                    os.makedirs(f"results/epoch_{epoch}")
                torch.save(netG,f"results/epoch_{epoch}/generator_{epoch}.pkl")
                torch.save(netD,f"results/epoch_{epoch}/discriminator_{epoch}.pkl")
                with open(f"results/epoch_{epoch}/losses_{epoch}.pkl", 'wb') as losses_file:
                    pickle.dump([G_losses, D_losses], losses_file)
                    losses_file.close()
                with torch.no_grad():
                    for i in range(100):
                        fake_img = netG(torch.unsqueeze(fixed_noise[i],0)).detach().cpu()
                        save_image(fake_img, f"results/epoch_{epoch}/img_{i}.png", normalize=True)
                        
            iters += 1	 
    plt.close('all')
    real_batch = next(iter(dataloader))
#Imágenes reales
    plt.figure(figsize=(128,128))
    plt.subplot(1,2,1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:128], padding=5, normalize=True).cpu(),(1,2,0)))
#Imagenes fake
    plt.subplot(1,2,2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(img_list[-1],(1,2,0)))
    plt.show()


