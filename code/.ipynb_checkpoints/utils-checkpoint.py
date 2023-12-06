import scipy.io 
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os 
import math 
import random
from tqdm import tqdm
import torch
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error as mse


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
def reparameterization(mu, logvar):
    std = torch.exp(0.5*logvar)
    eps = torch.randn_like(std)
    return mu + eps*std


def get_latent_space(model, data):
    z = model.encode(data)
    return z.detach().numpy()

def plot_loss(train_loss, val_loss, title='Loss'):
    sns.set_style('darkgrid')
    sns.set_context('paper')
    plt.figure(figsize=(8, 6))
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend()
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()
    
def plot_mode(z, X, Y, ):
    """
    Use this function to plot the modes of the latent space. 
    """
    if z.shape[1] < 5:
        num = int(z.shape[1])
    else: num = 5
    r = range(num)
    fig, axs = plt.subplots(nrows=num, ncols=1, figsize=(num*2,num*4)) 
    if int(z.shape[1]) == 1:
        axs = [axs]
    else: axs = axs.flatten()
    idx = 0
    for i in r:
        pcm = axs[idx].pcolormesh(X,Y,z[:,i].reshape(nx,ny).T,cmap = 'RdBu_r')
        fig.colorbar(pcm,ax=axs[idx])
        axs[idx].set_title(f'{args.model_name} mode {i+1} for $v$ velocity')
        axs[idx].set_aspect('equal')
        axs[idx].set_xlabel('x', fontsize = 14)
        axs[idx].set_ylabel('y',fontsize = 14)
        idx += 1
    fig.tight_layout()
    plt.clf()
    
def corr_matrix(z, mode_name):
    """
    Use this function to plot the correlation matrix of the latent space. 
    """
    # IF z.shape[1] larger than 5, then only plot the first 5 modes.
    if z.shape[1] < 5:
        num = int(z.shape[1])
    else: num = 5
    z = z[:, :num]
    corr = np.corrcoef(z.T)
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, cmap='RdBu_r', annot=True, vmin=-1, vmax=1)
    plt.title(f'Correlation Matrix for {mode_name}')
    plt.show()
    
def rank_z(args, model, original_data):
    """
    Use this function to rank the latent space. 
    """
    model.eval()
    model = model.to('cpu')
    # construct data
    z = model.encoder(torch.Tensor(original_data))
    if args.vae: 
        _, z, _, _  = model.forward(torch.Tensor(original_data))
    if args.deep_matrix:
        z = model.deep_matrix(z)
    z = z.detach().cpu().numpy()
    
    num = args.latent_dim if args.latent_dim < 5 else 5

    top_scores = [] 
    top_reconstruction = []
    top_modes = []
    prev_best = None 
    
    for i in range(1, num+1):
        current_best_score = None
        current_best_combo = None
        current_best_xhat = None
        current_best_mode = None

        for j in tqdm(range(int(z.shape[1])), desc=f'Searching for Rank {i}', unit='mode'):
            if prev_best is None:
                combo = (j,)
            else:
                if j in prev_best:
                    continue
                combo = prev_best + (j,)
            for index in combo:
                z_i = np.zeros_like(z)
                z_i[:, index] = z[:, index]
            x_hat = model.decoder(torch.Tensor(z_i))
            x_hat = x_hat.detach().cpu().numpy()
            # original energy
            score = mse(x_hat.reshape(-1,250), original_data.reshape(-1,250))
            # reconstruction energy
            #score = cal_energy(model, original_data, combo, z, 1, 1)
            if current_best_score is None or score < current_best_score:
                current_best_score = score
                current_best_combo = combo
                current_best_xhat = x_hat
                current_best_mode = z_i

        top_scores.append((current_best_score, current_best_combo))
        top_reconstruction.append(current_best_xhat)
        top_modes.append(current_best_mode)
        prev_best = current_best_combo

    return z, top_scores, top_reconstruction, top_modes
    

def cal_energy(model, original_data, combo, z, phi, V):
    for index in combo: 
        z_i = np.zeros_like(z)
        z_i[:, index] = z[:, index]
    x_hat = model.decoder(z_i)
    
    # original energy 
    E_o = 0.5 * phi * V * np.mean(original_data, axis=0)
    # reconstruction energy 
    E_r = 0.5 * phi * V * np.mean(x_hat, axis=0)
    # average percent 
    E_p = np.mean(E_r / E_o)
    
    return E_p

def plot_rank(args, model, z):
    """
    Use this function to plot the rank of the latent space. 
    """
    #z = z.detach().cpu().numpy()
    
    c = np.cov(z, rowvar=False)
    u, d, v = np.linalg.svd(c)
    
    d = d / d[0]
    
    plt.plot(range(args.latent_dim), d)

    plt.autoscale(enable=True, axis='y', tight=True)
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.ylim(0, 1)
    plt.xlim(0, args.latent_dim)
    plt.xlabel("Singular Value Rank")
    plt.ylabel("Singular Values")
    plt.title(f"{model.__class__.__name__} Singular Values of Covariance Matrix")
    plt.savefig(f'/home/dylan/repo/10617Project/results/physical/singular_values_{model.__class__.__name__}_{args.notes}_{args.latent_dim}.png')
    plt.show()
    plt.close()


def plot_reconstruction(args, model, X, Y, nx, ny, orginal_data, top_reconstruction):
    fig, axs = plt.subplots(nrows=len(top_reconstruction) + 2, ncols=1, figsize=(8,16))
    axs = axs.flatten()
    idx = 0
    norm = mcolors.Normalize(vmin=0, vmax=1)
    for reconstruction in top_reconstruction:
        if 'MLP' in args.model:
            reconstruction_t = reconstruction[:, 220]
        else:
            reconstruction_t = reconstruction[220, : ,: ,:].reshape(1,1,nx,ny) # use t = 20 
            reconstruction_t = reconstruction_t.transpose(2,3,0,1).reshape(nx*ny,-1)
        # normalize
        #norms = np.linalg.norm(reconstruction_t, axis=0)
        #normalized_reconstruction_t = reconstruction_t / np.power(norms, 0.5)
        pcm = axs[idx].pcolormesh(X,Y,reconstruction_t.reshape(nx,ny).T,cmap = 'RdBu_r')
        fig.colorbar(pcm,ax=axs[idx])
        axs[idx].set_title(f'Reconstruction for mode {idx+1}')
        axs[idx].set_aspect('equal')
        axs[idx].set_xlabel('x', fontsize = 14)
        axs[idx].set_ylabel('y',fontsize = 14)
        idx += 1
    
    # latent reconstruction data
    model.eval()
    x_hat, *_ = model.forward(torch.Tensor(orginal_data))
    x_hat = x_hat.detach().cpu().numpy()
    if 'MLP' in args.model:
        x_hat = x_hat[:, 220]
    else:
        x_hat = x_hat[220, : ,: ,:].reshape(1,1,nx,ny)
        x_hat = x_hat.transpose(2,3,0,1).reshape(nx*ny,-1)
    pcm = axs[idx].pcolormesh(X,Y,x_hat.reshape(nx,ny).T,cmap = 'RdBu_r')
    fig.colorbar(pcm, ax=axs[idx])
    axs[idx].set_title('whole latent reconstruction')
    axs[idx].set_aspect('equal')
    axs[idx].set_xlabel('x', fontsize = 14)
    axs[idx].set_ylabel('y', fontsize = 14)
    idx += 1 
    
    if 'MLP' in args.model:
        orginal_data_t = orginal_data[:, 220]
    else:
        orginal_data_t = orginal_data[220, : ,: ,:].reshape(1,1,nx,ny)
        orginal_data_t = orginal_data_t.transpose(2,3,0,1).reshape(nx*ny,-1)
    pcm = axs[idx].pcolormesh(X,Y,orginal_data_t.reshape(nx,ny).T,cmap = 'RdBu_r')
    fig.colorbar(pcm, ax=axs[idx])
    axs[idx].set_title('the original result')
    axs[idx].set_aspect('equal')
    axs[idx].set_xlabel('x', fontsize = 14)
    axs[idx].set_ylabel('y', fontsize = 14)
    fig.tight_layout()
    plt.savefig(f'/home/dylan/repo/10617Project/results/physical/reconstruction_{model.__class__.__name__}_{args.notes}_{args.latent_dim}.png')
    
    
def frange_cycle_linear(n_iter, start=0.0, stop=1.0, n_cycle=4, ratio_increase=0.5):
    l = np.ones(n_iter) * stop
    period = n_iter / n_cycle
    step = (stop - start) / (period * ratio_increase)
    for c in range(n_cycle):
        v, i = start, 0
        while v <= stop and (int(i + c * period) < n_iter):
            l[int(i + c * period)] = v
            v += step
            i += 1
    return l