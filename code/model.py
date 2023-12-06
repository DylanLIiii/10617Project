import numpy as np 
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MLPEncoderBlock(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim, activation=nn.ReLU()):
        """_summary_

        Args:
            input_dim (_type_): _description_
            hidden_dims (_type_): _description_
            latent_dim (_type_): _description_
            activation (_type_, optional): _description_. Defaults to nn.ReLU(). Use nn.Identity() for linear activation.
        """
        super(MLPEncoderBlock, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.activation = activation
        self.mlp = nn.ModuleList()
        self.mlp.append(nn.Linear(self.input_dim, self.hidden_dims[0]))
        for i in range(1, len(self.hidden_dims)):
            self.mlp.append(nn.Linear(self.hidden_dims[i-1], self.hidden_dims[i]))
        self.latent_map = nn.Linear(self.hidden_dims[-1], self.latent_dim)
    
    def forward(self, x): 
        x = self.mlp[0](x)
        x = self.activation(x)
        for i in range(1, len(self.mlp)):
            x = self.mlp[i](x)
            x = self.activation(x)
        x = self.latent_map(x)
        return x
        
class MLPDecoderBlock(nn.Module):
    def __init__(self, latent_dim, hidden_dims, output_dim, activation=nn.ReLU()):
        super(MLPDecoderBlock, self).__init__()
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.activation = activation
        self.mlp = nn.ModuleList()
        self.mlp.append(nn.Linear(self.latent_dim, self.hidden_dims[0]))
        for i in range(1, len(self.hidden_dims)):
            self.mlp.append(nn.Linear(self.hidden_dims[i-1], self.hidden_dims[i]))
        self.output_map = nn.Linear(self.hidden_dims[-1], self.output_dim)
    
    def forward(self, z): 
        z = self.mlp[0](z)
        z = self.activation(z)
        for i in range(1, len(self.mlp)):
            z = self.mlp[i](z)
            z = self.activation(z)
        z = self.output_map(z)
        return z
    
    
class CNNEncoderBlock(nn.Module):
    def __init__(self, input_channels, hidden_channels, latent_dim, kernel_sizes, strides, paddings, activation=nn.ReLU()) -> None:
        super().__init__()
        self.activation = activation 
        self.conv = nn.ModuleList()
        self.input_channels = input_channels
        self.latent_dim = latent_dim
        self.hidden_channels = hidden_channels
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.paddings = paddings
        self.conv.append(nn.Sequential(nn.Conv2d(self.input_channels, self.hidden_channels[0], self.kernel_sizes[0], self.strides[0], self.paddings[0]),
                                    self.activation,
                                    nn.MaxPool2d(2, 2)))
        for i in range(1, len(self.hidden_channels)):
            self.conv.append(nn.Sequential(nn.Conv2d(self.hidden_channels[i-1], self.hidden_channels[i], self.kernel_sizes[i], self.strides[i], self.paddings[i]),
                                    self.activation,
                                    nn.MaxPool2d(2, 2)))
        self.latent_mlp = nn.Linear(84, self.latent_dim)
        
    def forward(self, x):
        x = self.conv[0](x)
        #print(f"Encode 0 {x.size()}")
        for i in range(1, len(self.conv)):
            x = self.conv[i](x)
            #print(f"Encode {i, x.size()}")
        x = x.view(x.size(0), -1)
        self.input_latent_dim = x.size(1)
        x = self.latent_mlp(x)
        return x
    
class CNNDecoderBlock(nn.Module):
    def __init__(self, latent_dim, hidden_channels, output_channels, kernel_sizes, strides, paddings, activation=nn.ReLU()) -> None:
        super().__init__()
        self.activation = activation 
        self.conv = nn.ModuleList()
        self.latent_dim = latent_dim
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.paddings = paddings
        # self.conv.append(nn.Sequential(nn.ConvTranspose2d(self.latent_dim, self.hidden_channels[0], self.kernel_sizes[0], self.strides[0], self.paddings[0]),self.activation))
        for i in range(1, len(self.hidden_channels)):
            self.conv.append(nn.Sequential(self.activation,
                                        nn.ConvTranspose2d(self.hidden_channels[i-1], self.hidden_channels[i], self.kernel_sizes[i], self.strides[i], self.paddings[i]),))
        self.conv.append(nn.Sequential(self.activation,
                                    nn.ConvTranspose2d(self.hidden_channels[-1], self.output_channels, self.kernel_sizes[-1], self.strides[-1], self.paddings[-1])))
        
        if self.latent_dim == 2: 
            self.latent_mlp = nn.Linear(1, 84)
        else:
            self.latent_mlp = nn.Linear(self.latent_dim, 84)
    def forward(self, z):
        z = self.latent_mlp(z)
        z = z.view(z.size(0), self.hidden_channels[0], 7, 3)
        
        z = F.interpolate(z, size=(15,7), mode='nearest')
        z = self.conv[0](z)
        #rint(f"Decode: {0 , z.size()}")
        
        z = F.interpolate(z, size=(31, 15), mode='nearest')
        z = self.conv[1](z)
        #print(f"Decode: {1 , z.size()}")
        
        z = F.interpolate(z, size=(62, 31), mode='nearest')
        z = self.conv[2](z)
        #print(f"Decode: {2 , z.size()}")
        
        z = F.interpolate(z, size=(125, 62), mode='nearest') # Not reverse of maxpooling. Generally upsampling is done by interpolation.
        z = self.conv[3](z)
        #print(f"Decode: {3 , z.size()}")
        
        z = F.interpolate(z, size=(250, 125), mode='nearest')
        z = self.conv[4](z)
        #print(f"Decode: {4 , z.size()}")
        
        z = F.interpolate(z, size=(501, 251), mode='nearest')
        z = self.conv[5](z)
        #print(f"Decode: {5 , z.size()}")
        return z
    
    
class DeepMatrixBlcok(nn.Module):
    def __init__(self, latent_dim, constant_dim, deepth) -> None: 
        super(DeepMatrixBlcok, self).__init__()
        self.latent_dim = latent_dim
        self.constant_dim = constant_dim
        self.deepth = deepth
        self.matrix = nn.ModuleList()
        for i in range(self.deepth):
            if i == 0: 
                self.matrix.append(nn.Linear(self.latent_dim, self.constant_dim))
            self.matrix.append(nn.Linear(self.constant_dim, self.constant_dim))
        self.final_layer  = nn.Linear(self.constant_dim, self.latent_dim)
    def forward(self, x):
        x = self.matrix[0](x)
        for i in range(1, self.deepth):
            x = self.matrix[i](x)
        x = self.final_layer(x)
        return x
    

class CNNAutoEncoder(nn.Module):
    def __init__(self, input_channels, hidden_channels=[16, 8, 8, 8, 4, 4], latent_dim=5, kernel_sizes=[3, 3, 3, 3, 3, 3], strides=[1,1,1,1,1,1], paddings=[1,1,1,1,1,1], activation=nn.Tanh()):
        super().__init__()
        self.encoder = CNNEncoderBlock(input_channels, hidden_channels, latent_dim, kernel_sizes, strides, paddings, activation)
        self.decoder = CNNDecoderBlock(latent_dim, hidden_channels[::-1], input_channels, kernel_sizes[::-1], strides[::-1], paddings[::-1], activation)
    
    def forward(self, x):
        if self.encoder.latent_dim == 2:
            z = self.encoder(x)
            z1 = z[:, 0].unsqueeze(1)
            z2 = z[:, 1].unsqueeze(1)
            x1 = self.decoder(z1)
            x2 = self.decoder(z2)
            x_hat = x1 + x2 
        else:
            z = self.encoder(x)
            x_hat = self.decoder(z)
        return x_hat, z 
    
    
    
class MLPAutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dims=[512, 256, 128, 64], latent_dim=5, activation=nn.ReLU()):
        super().__init__()
        self.encoder = MLPEncoderBlock(input_dim, hidden_dims, latent_dim, activation)
        self.decoder = MLPDecoderBlock(latent_dim, hidden_dims[::-1], input_dim, activation)
    
    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z
    
    
class MLPAutoEncoderWithDeepMatrix(nn.Module):
    def __init__(self, input_dim, hidden_dims=[512, 256, 128, 64], latent_dim=5, activation=nn.ReLU(), deepth=3):
        super().__init__()
        self.encoder = MLPEncoderBlock(input_dim, hidden_dims, latent_dim, activation)
        self.decoder = MLPDecoderBlock(latent_dim, hidden_dims[::-1], input_dim, activation)
        self.deep_matrix = DeepMatrixBlcok(latent_dim, latent_dim, deepth)
    
    def forward(self, x):
        z = self.encoder(x)
        z = self.deep_matrix(z)
        x = self.decoder(z)
        return x, z
    
    
class CNNAutoEncoderWithDeepMatrix(nn.Module):
    def __init__(self, input_channels, hidden_channels=[16, 8, 8, 8, 4, 4], latent_dim=5, kernel_sizes=[3, 3, 3, 3, 3, 3], strides=[1,1,1,1,1,1], paddings=[1,1,1,1,1,1], activation=nn.Tanh(), deepth=5):
        super().__init__()
        self.encoder = CNNEncoderBlock(input_channels, hidden_channels, latent_dim, kernel_sizes, strides, paddings, activation)
        self.decoder = CNNDecoderBlock(latent_dim, hidden_channels[::-1], input_channels, kernel_sizes[::-1], strides[::-1], paddings[::-1], activation)
        self.deep_matrix = DeepMatrixBlcok(latent_dim, latent_dim, deepth)
    
    def forward(self, x):
        z = self.encoder(x)
        z = self.deep_matrix(z)
        x = self.decoder(z)
        return x, z
    
    
def reparameterize(mu, std):
    eps = torch.randn_like(std)
    return mu + eps*std
    
    
class CNNVAE(nn.Module):
    def __init__(self, args, input_channels, hidden_channels=[16, 8, 8, 8, 4, 4], latent_dim=5, kernel_sizes=[3, 3, 3, 3, 3, 3], strides=[1,1,1,1,1,1], paddings=[1,1,1,1,1,1], activation=nn.Tanh()):
        super().__init__()
        self.args = args
        self.latent_dim = latent_dim
        self.encoder = CNNEncoderBlock(input_channels, hidden_channels, latent_dim * 2, kernel_sizes, strides, paddings, activation)
        self.decoder = CNNDecoderBlock(latent_dim, hidden_channels[::-1], input_channels, kernel_sizes[::-1], strides[::-1], paddings[::-1], activation)
        self.scaler =  Scaler()
        self.mean_norm = nn.BatchNorm1d(latent_dim, affine=False, eps=1e-8)
        self.std_norm = nn.BatchNorm1d(latent_dim, affine=False, eps=1e-8)
        
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        z = self.encoder(x)
        
        if self.args.vae_bn == 1:
            # BN
            mu = z[:, :self.latent_dim]
            mu = self.mean_norm(mu)
            mu = self.scaler(mu)
            logvar = z[:, self.latent_dim:]
            std = torch.exp(0.5*logvar)
            std = self.scaler(std)
            std = self.std_norm(std)
            z_bar = reparameterize(mu, std)
        else: 
            mu = z[:, :self.latent_dim]
            logvar = z[:, self.latent_dim:]
            std = torch.exp(0.5*logvar)
            z_bar = reparameterize(mu, std)
        x = self.sigmoid(self.decoder(z_bar))
        return x, z_bar, mu, logvar
    
class CNNVAEWithDeepMatrix(nn.Module):
    def __init__(self, args, input_channels, hidden_channels=[16, 8, 8, 8, 4, 4], latent_dim=5, kernel_sizes=[3, 3, 3, 3, 3, 3], strides=[1,1,1,1,1,1], paddings=[1,1,1,1,1,1], activation=nn.Tanh(), deepth=5):
        super().__init__()
        self.args = args
        self.latent_dim = latent_dim
        self.encoder = CNNEncoderBlock(input_channels, hidden_channels, latent_dim * 2, kernel_sizes, strides, paddings, activation)
        self.decoder = CNNDecoderBlock(latent_dim, hidden_channels[::-1], input_channels, kernel_sizes[::-1], strides[::-1], paddings[::-1], activation)
        self.deep_matrix = DeepMatrixBlcok(latent_dim, latent_dim, deepth)
        self.scaler =  Scaler()
        self.mean_norm = nn.BatchNorm1d(latent_dim, affine=False, eps=1e-8)
        self.std_norm = nn.BatchNorm1d(latent_dim, affine=False, eps=1e-8)
        
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        z = self.encoder(x)
        if self.args.vae_bn == 1:
            # BN
            mu = z[:, :self.latent_dim]
            mu = self.mean_norm(mu)
            mu = self.scaler(mu)
            logvar = z[:, self.latent_dim:]
            std = torch.exp(0.5*logvar)
            std = self.scaler(std)
            std = self.std_norm(std)
            z_bar = reparameterize(mu, std)
        else: 
            mu = z[:, :self.latent_dim]
            logvar = z[:, self.latent_dim:]
            std = torch.exp(0.5*logvar)
            z_bar = reparameterize(mu, std)
        x = self.sigmoid(self.decoder(z_bar))
        return x, z_bar, mu, logvar
    
class MLPVAE(nn.Module):
    def __init__(self, args, input_dim, hidden_dims=[512, 256, 128, 64], latent_dim=5, activation=nn.ReLU()):
        super().__init__()
        self.args = args
        self.latent_dim = latent_dim
        self.encoder = MLPEncoderBlock(input_dim, hidden_dims, latent_dim * 2, activation)
        self.decoder = MLPDecoderBlock(latent_dim, hidden_dims[::-1], input_dim, activation)
        self.scaler =  Scaler()
        self.mean_norm = nn.BatchNorm1d(latent_dim, affine=False, eps=1e-8)
        self.std_norm = nn.BatchNorm1d(latent_dim, affine=False, eps=1e-8)
        
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        z = self.encoder(x)
        
        if self.args.vae_bn == 1:
            # BN
            mu = z[:, :self.latent_dim]
            mu = self.mean_norm(mu)
            mu = self.scaler(mu)
            logvar = z[:, self.latent_dim:]
            std = torch.exp(0.5*logvar)
            std = self.scaler(std)
            std = self.std_norm(std)
            z_bar = reparameterize(mu, std)
        else: 
            mu = z[:, :self.latent_dim]
            logvar = z[:, self.latent_dim:]
            std = torch.exp(0.5*logvar)
            z_bar = reparameterize(mu, std)
        x = self.sigmoid(self.decoder(z_bar))
        return x, z_bar, mu, logvar
    
class MLPVAEWithDeepMatrix(nn.Module):
    def __init__(self, args, input_dim, hidden_dims=[512, 256, 128, 64], latent_dim=5, activation=nn.ReLU(), deepth=5):
        super().__init__()
        self.args = args
        self.latent_dim = latent_dim
        self.encoder = MLPEncoderBlock(input_dim, hidden_dims, latent_dim * 2, activation)
        self.decoder = MLPDecoderBlock(latent_dim, hidden_dims[::-1], input_dim, activation)
        self.deep_matrix = DeepMatrixBlcok(latent_dim, latent_dim, deepth)
        self.scaler =  Scaler()
        self.mean_norm = nn.BatchNorm1d(latent_dim, affine=False, eps=1e-8)
        self.std_norm = nn.BatchNorm1d(latent_dim, affine=False, eps=1e-8)
        
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        z = self.encoder(x)
        
        if self.args.vae_bn == 1:
            # BN
            mu = z[:, :self.latent_dim]
            mu = self.mean_norm(mu)
            mu = self.scaler(mu)
            logvar = z[:, self.latent_dim:]
            std = torch.exp(0.5*logvar)
            std = self.scaler(std)
            std = self.std_norm(std)
            z_bar = reparameterize(mu, std)
        else: 
            mu = z[:, :self.latent_dim]
            logvar = z[:, self.latent_dim:]
            std = torch.exp(0.5*logvar)
            z_bar = reparameterize(mu, std)
        x = self.sigmoid(self.decoder(z_bar))
        return x, z_bar, mu, logvar
    
class Scaler(nn.Module):
    """Special scale layer"""
    def __init__(self, tau=0.5):
        super(Scaler, self).__init__()
        self.tau = tau
        

    def forward(self, inputs, mode='positive'):
        self.scale = nn.Parameter(torch.zeros(inputs.shape[-1]))
        if mode == 'positive':
            scale = self.tau + (1 - self.tau) * torch.sigmoid(self.scale)
        else:
            scale = (1 - self.tau) * torch.sigmoid(-self.scale)
        return inputs * torch.sqrt(scale).to(inputs.device)