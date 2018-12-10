import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import datasets
from torchvision import transforms


###############################################################################
####################### 1. LOAD THE TRAINING SET DATA #########################
###############################################################################
def get_dataloader(batch_size, image_size, data_dir='processed_celeba_small/'):
    
    transform = transforms.Compose([transforms.Resize(image_size),
                                    transforms.ToTensor()])
    dataset = datasets.ImageFolder(data_dir, transform = transform)
    print('# of training images: ', len(dataset))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, 
                                            num_workers = 0, shuffle = True)
    
    return dataloader


batch_size = 128
img_size = 32
celeba_train_loader = get_dataloader(batch_size, img_size)
    

###############################################################################
###################### 2. DEFINE DISCRIMINATOR NETWORK ########################
###############################################################################
import torch.nn as nn
import torch.nn.functional as F

def conv(in_channels, out_channels, kernel_size, stride = 2, padding = 1, batch_norm = True):
    
    layers = []
    layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias = False))
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
        
    return nn.Sequential(*layers)


class Discriminator(nn.Module):

    def __init__(self, conv_dim):    # conv_dim: Depth of the first convolutional layer

        super(Discriminator, self).__init__()
        self.conv_dim = conv_dim
        
        self.conv1 = conv(3, self.conv_dim, kernel_size = 4, batch_norm = False) 
        self.conv2 = conv(self.conv_dim, self.conv_dim*2, kernel_size = 4)    
        self.conv3 = conv(self.conv_dim*2, self.conv_dim*4, kernel_size = 4)   
        
        self.fc = nn.Linear(self.conv_dim*4 * 4 * 4, 1)
        

    def forward(self, x):
        #                   input image   :         (128, 3, 32, 32)
        x = F.leaky_relu(self.conv1(x))           # (128, 32, 16, 16)
        x = F.leaky_relu(self.conv2(x))           # (128, 64, 8, 8)
        x = F.leaky_relu(self.conv3(x))           # (128, 128, 4, 4)
        
        x = x.view(-1, self.conv_dim*4 * 4 * 4)   # (128, 1, 128*4*4)
        x = self.fc(x)                            # (128, 1, 1)
        
        return x
    
    
###############################################################################
######################### 3. DEFINE GENERATOR NETWORK #########################
###############################################################################
def trans_conv(in_channels, out_channels, kernel_size, stride = 2, padding = 1, batch_norm = True):
    
    layers = []
    layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias = False))
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
        
    return nn.Sequential(*layers)


class Generator(nn.Module):
    
    def __init__(self, z_size, conv_dim):       # conv_dim: Depth of the inputs to the LAST transpose convolutional layer

        super(Generator, self).__init__()
        self.z_size = z_size
        self.conv_dim = conv_dim
    
        self.fc = nn.Linear(self.z_size, 4 * 4 * self.conv_dim*4)   # 4, 4

        self.deconv1 = trans_conv(self.conv_dim*4, self.conv_dim*2, kernel_size = 4)    # 8, 8
        self.deconv2 = trans_conv(self.conv_dim*2, self.conv_dim, kernel_size = 4)       # 16, 16
        self.deconv3 = trans_conv(self.conv_dim, 3, kernel_size = 4, batch_norm = False)    # 32, 32
        
        
    def forward(self, x):     
        #                               input z : (128, 100)
        x = self.fc(x)                          # (128, 128*4*4)
        x = x.view(-1, self.conv_dim*4, 4, 4)   # (128, 128, 4, 4)
        
        x = F.relu(self.deconv1(x))             # (128, 64, 8, 8)
        x = F.relu(self.deconv2(x))             # (128, 32, 16, 16)
        x = F.tanh(self.deconv3(x))             # (128, 3, 32, 32)
        
        return x
    
###############################################################################
###################### 4. FUNCTION TO INITIALIZE WEIGHTS ######################
###############################################################################    
from torch.nn import init

def weights_init_normal(m):
    """
    Weights are initialized w/ a normal distribution (mean = 0, std dev = 0.02)
    - m: A module or layer in a network    
    """
    # classname will be something like:  `Conv`, `BatchNorm2d`, `Linear`, etc.
    classname = m.__class__.__name__
    if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
        init.normal_(m.weight.data, 0.0, 0.02)
        

###############################################################################
################# 5. BUILD D & G MODELS & INITIALIZE WEIGHTS ##################
###############################################################################
def build_network(d_conv_dim, g_conv_dim, z_size):
    # define discriminator and generator
    D = Discriminator(d_conv_dim)
    G = Generator(z_size = z_size, conv_dim = g_conv_dim)
    # initialize model weights
    D.apply(weights_init_normal)
    G.apply(weights_init_normal)
    
    print(D)
    print()
    print(G)
    
    return D, G


d_conv_dim = 32
g_conv_dim = 32
z_size = 100
D, G = build_network(d_conv_dim, g_conv_dim, z_size)

train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    print('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Training on GPU!')
    
    
###############################################################################
#################### 6. FUNCTION TO SCALE THE PIXEL VALUES ####################
############################################################################### 
# necessary because of generator's tanh activation function @ last layer
def scale(x, feature_range=(-1, 1)):
    # assume x is scaled to (0, 1) & scale to feature_range and return scaled x
    min, max = feature_range
    x = x * (max-min) - max
    
    return x


###############################################################################
####################### 7. DEFINE LOSSES & OPTIMIZERS #########################
###############################################################################
def real_loss(D_out):
    loss = torch.mean((D_out - 0.9)**2)
    return loss


def fake_loss(D_out):
    loss = torch.mean((D_out)**2)
    return loss


import torch.optim as optim
lr = 0.0002
beta1 = 0.5
beta2 = 0.999
d_optimizer = optim.Adam(D.parameters(), lr, [beta1, beta2])
g_optimizer = optim.Adam(G.parameters(), lr, [beta1, beta2])


###############################################################################
############################# 8. TRAIN THE NETWORK ############################
###############################################################################
def train(D, G, n_epochs, print_every=150):
   
    if train_on_gpu:
        D.cuda()
        G.cuda()

    samples = []
    losses = []

    # Get some fixed data for sampling (16)
    sample_size=16
    fixed_z = np.random.uniform(-1, 1, size=(sample_size, z_size))
    fixed_z = torch.from_numpy(fixed_z).float()
   
    if train_on_gpu:
        fixed_z = fixed_z.cuda()

    for epoch in range(n_epochs):
        for batch_i, (real_images, _) in enumerate(celeba_train_loader):

            batch_size = real_images.size(0)
            real_images = scale(real_images)
            if train_on_gpu:
                real_images = real_images.cuda()

            # =============================================== #
            #         YOUR CODE HERE: TRAIN THE NETWORKS      #
            # =============================================== #
            
            # 1. Train the discriminator on real and fake images
            d_optimizer.zero_grad()
            d_loss_real = real_loss(D(real_images))
            
            z = np.random.uniform(-1, 1, size = (batch_size, z_size))
            z = torch.from_numpy(z).float()
            if train_on_gpu:
                z = z.cuda()
            fake_images = G(z)
            d_loss_fake = fake_loss(D(fake_images))
            
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            d_optimizer.step()

            # 2. Train the generator with an adversarial loss
            g_optimizer.zero_grad()
            
            z = np.random.uniform(-1, 1, size = (batch_size, z_size))
            z = torch.from_numpy(z).float()
            if train_on_gpu:
                z = z.cuda()
            fake_images = G(z)
            
            g_loss = real_loss(D(fake_images))
            g_loss.backward()
            g_optimizer.step()
                     
            if batch_i % print_every == 0:
                losses.append((d_loss.item(), g_loss.item()))
                print('Epoch [{:5d}/{:5d}] | d_loss: {:6.4f} | g_loss: {:6.4f}'.format(
                        epoch+1, n_epochs, d_loss.item(), g_loss.item()))


        ## AFTER EACH EPOCH##    
        G.eval() 
        samples_z = G(fixed_z)
        samples.append(samples_z)
        G.train()
        
        if (epoch % 6) == 0:
            with open('train_samples.pkl', 'wb') as f:
                pkl.dump(samples, f)

    # After training, save training generator samples
    with open('train_samples.pkl', 'wb') as f:
        pkl.dump(samples, f)
    
    return losses


n_epochs = 50

losses = train(D, G, n_epochs=n_epochs)


###############################################################################
############################ 9. VISUALIZE SOME RESULTS ########################
###############################################################################
def view_samples(epoch, samples):
    fig, axes = plt.subplots(figsize=(16,4), nrows=2, ncols=8, sharey=True, sharex=True)
    for ax, img in zip(axes.flatten(), samples[epoch]):
        img = img.detach().cpu().numpy()
        img = np.transpose(img, (1, 2, 0))
        img = ((img + 1)*255 / (2)).astype(np.uint8)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        im = ax.imshow(img.reshape((32,32,3)))
        

with open('train_samples.pkl', 'rb') as f:
    samples = pkl.load(f)
_ = view_samples(-1, samples)