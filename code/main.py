from model import MLPAutoEncoder, CNNAutoEncoder, MLPAutoEncoderWithDeepMatrix, CNNAutoEncoderWithDeepMatrix, CNNVAE, CNNVAEWithDeepMatrix, MLPVAEWithDeepMatrix, MLPVAE
from train import * 
from utils import * 
import wandb 
import copy
import argparse
import torch.nn as nn
import scipy.io as sio
import logging
import pandas as pd
import numpy as np
from ranger21 import Ranger21
logging.basicConfig(level=logging.INFO, format='%(asctime)s :: %(levelname)s :: %(message)s')

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('The Boolean value should be one of true/false or yes/no.')


parser = argparse.ArgumentParser()


# training params
parser.add_argument('--model', type=str, default='MLP')
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=25)
parser.add_argument('--latent_dim', type=int, default=5) # best with 3 8 and with deep matrix and Non vae
parser.add_argument('--ranger21', type=str2bool, default=True)
parser.add_argument('--lr_patience', type=int, default=10)
parser.add_argument('--stop_patience', type=int, default=20)


## DeepMatrix params
parser.add_argument('--deep_matrix', type=str2bool, default=True)
parser.add_argument('--deepth', type=int, default=8)

## VAE params 
parser.add_argument('--vae', type=str2bool, default=False)
parser.add_argument('--cycle', type=int, default=10)
parser.add_argument('--vae_bn', type=int, default=0)

## Experinment Notes 
parser.add_argument('--name', type=str, default='MLPDM_latent5', help='Experiment name')
parser.add_argument('--wandb', type=str2bool, default=True)
parser.add_argument('--notes', type=str, default='test')
parser.add_argument('--tags', type=list, default='None')

parser.add_argument('--model_dir', type=str, default='/home/dylan/repo/10617Project/model')

args = parser.parse_args()

# Print the arguments
print("\nConfigured Training Parameters:")
for arg in vars(args):
    print(f"{arg}: {getattr(args, arg)}")
print("\n")


def get_model(args, input_dim):
    if args.model == 'MLP':
        if args.vae:
            if args.deep_matrix:
                model = MLPVAEWithDeepMatrix(args, input_dim, latent_dim=args.latent_dim, activation=nn.ReLU(), deepth=args.deepth)
            else: model = MLPVAE(args, input_dim, latent_dim=args.latent_dim, activation=nn.ReLU())
        else:
            if args.deep_matrix:
                model = MLPAutoEncoderWithDeepMatrix(input_dim, latent_dim=args.latent_dim, activation=nn.ReLU(), deepth=args.deepth)
            else: model = MLPAutoEncoder(input_dim, latent_dim=args.latent_dim, activation=nn.ReLU())
    elif args.model == 'CNN':
        if args.vae:
            if args.deep_matrix:
                model = CNNVAEWithDeepMatrix(args, input_dim, latent_dim=args.latent_dim, activation=nn.ReLU(), deepth=args.deepth)
            else:
                model = CNNVAE(args, input_dim, latent_dim=args.latent_dim, activation=nn.ReLU())
        else: 
            if args.deep_matrix:
                model = CNNAutoEncoderWithDeepMatrix(input_dim, latent_dim=args.latent_dim, activation=nn.ReLU(), deepth=args.deepth)
            else:
                model = CNNAutoEncoder(input_dim, latent_dim=args.latent_dim, activation=nn.ReLU())
    else:
        raise ValueError('Invalid model name')
    print(f"Load model {model.__class__.__name__} successfully!")
    return model


def get_data(args):
    raise NotImplementedError

def make(args):
    # seed everything
    seed_everything(3407)
    
    # load data 
    Data = sio.loadmat('data/physical/CYLINDER.mat')
    U = Data['U']
    V = Data['V']
    VORTALL = Data['VORTALL']
    X = Data['X']
    Y = Data['Y']
    Dx = Data['dx'].item()  
    Dy = Data['dy'].item()
    nx = Data['nx'].item()
    ny = Data['ny'].item()
    print(colored("Load data successfully", 'green'))
    
    
    # define configs
    if args.model == 'MLP':
        input_dim = U.shape[1]
    else:
        input_dim = 1 # should be NCHW, where c = 1
    
    
    
    if args.wandb:
        wandb.login()
        run = wandb.init(project='10617project', name= args.name, config=args,  notes=args.notes, tags=args.tags)
    else :
        run = None
        
    # get device 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # get model
    model = get_model(args, input_dim)
    print(colored("Model architecture: ", 'green'))
    print(model)
    
    # create dataset
    if args.model == 'MLP':
        U_max, U_min = np.max(U,axis=0).reshape(1,-1), np.min(U,axis=0).reshape(1,-1)
        U_std = (U - U_min)/ (U_max - U_min)
        U_scaled = U_std * (1 - 0) + 0
        #U_max, U_min, U_scaled_max, U_scaled_min  = np.max(U), np.min(U), np.max(U_scaled), np.min(U_scaled)
        U = U_scaled
        
        train_dataset, test_dataset = make_dataset(U, test_size=0.2)
    elif args.model == 'CNN':
        U = U.reshape(nx, ny, -1, 1).transpose(2, 3, 0, 1)
        U_max = np.inf
        U_min = -np.inf
        U = U.reshape(250,-1)
        U_max, U_min = np.max(U, axis=1).reshape(-1, 1), np.min(U, axis=1).reshape(-1, 1)
        U_std = (U - U_min)/ (U_max - U_min)
        U_scaled = U_std * (1 - 0) + 0
        #U_max, U_min, U_scaled_max, U_scaled_min  = np.max(U), np.min(U), np.max(U_scaled), np.min(U_scaled)
        
        U = U_scaled.reshape(250, 1, nx, ny)
        train_dataset, test_dataset = make_dataset(U, test_size=0.2)
    print(colored(f'Create dataset successfully! Train dataset size: {len(train_dataset)}, Test dataset size: {len(test_dataset)}', 'green'))
    
    # create dataloader
    train_loader, test_loader = make_dataloader(train_dataset, test_dataset, args.batch_size)
    print(colored(f'Create dataloader successfully! Train dataloader size: {len(train_loader)}, Test dataloader size: {len(test_loader)}', 'green'))
    
    # get beta list 
    beta = frange_cycle_linear(args.epochs, start=0.0, stop=1.0, n_cycle=args.cycle, ratio_increase=0.5) if args.vae else 0
    if args.vae: print(colored(f'Create Cyclical Annealing Successfully! Beta list size: {len(beta)}', 'green'))
    
    
    # train model 
    optim = Ranger21(model.parameters(), lr=args.lr, num_epochs=args.epochs, num_batches_per_epoch=len(train_loader)) if args.ranger21 else torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss() # input and target should be the same shape
    logging.info(f"Start training model {args.model} using device {device}")
    trained_model, train_losses, val_losses = train(args, model, optim, criterion, train_loader, test_loader,  args.epochs, device, beta, wandb_logging=args.wandb)
    
    # plot reconstruction results 
    #plot_loss(train_losses, val_losses, title=f'Loss for {args.model}')
    
    # calculate rank of mode using 
    z, top_scores, top_reconstruction, top_modes = rank_z(args, model, U)
    for i in range(len(top_scores)):
        print(f"Mode {i}: Latent Index {top_scores[i][1]} with Score {top_scores[i][0]}")
    
    # calculate corr matrix 
    z_hat = z[:, :]
    np.save(f'/home/dylan/repo/10617Project/results/physical/latent/z_hat_{model.__class__.__name__}_{args.notes}_{args.latent_dim}.npy', z_hat)
    df = pd.DataFrame(z_hat)
    corr_matrix = df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title(f"{model.__class__.__name__} Correlation Matrix")
    plt.savefig(f'/home/dylan/repo/10617Project/results/physical/singular/corr_matrix_{model.__class__.__name__}_{args.notes}_{args.latent_dim}.png')
    plt.show()
    plt.close()
    
    
    ## rank of z 
    z_rank = np.linalg.matrix_rank(z_hat)
    print(f"Rank of z: {z_rank}")
        
    # plot rank
    plot_rank(args, model, z)
    
    # plot_reconstruction
    plot_reconstruction(args, model, X, Y, nx, ny, U, top_reconstruction)
    
    #finish wandb
    if args.wandb:
        # log reconstruction rank 
        wandb.summary['Reconstruction Rank Score'] = [score for score, _ in top_scores]
        wandb.summary['Reconstruction Rank Index'] = [idx for _, idx in top_scores]
        # save model 
        wandb.save(os.path.join(args.model_dir, f"{model.__class__.__name__}_best.pth"))
        # log rank of z 
        wandb.summary['Rank of z_hat'] = z_rank
        # log rank plot 
        wandb.log({"Singular Value Plot": wandb.Image(f'/home/dylan/repo/10617Project/results/physical/singular_values_{model.__class__.__name__}_{args.notes}_{args.latent_dim}.png')})
        # log modes plot 
        wandb.log({"Modes Plot": wandb.Image(f'/home/dylan/repo/10617Project/results/physical/reconstruction_{model.__class__.__name__}_{args.notes}_{args.latent_dim}.png')})
        # log corr plot
        wandb.log({"Correlation Matrix": wandb.Image(f'/home/dylan/repo/10617Project/results/physical/singular/corr_matrix_{model.__class__.__name__}_{args.notes}_{args.latent_dim}.png')})
        wandb.finish()

if __name__ == '__main__':
    make(args)