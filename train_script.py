import subprocess 
import os 

# For model MLP 

model = 'MLP'
names = ['MLPAE','MLPAEWithDeepMatrix']
epochs = 100
lr_patience = 30
stop_patience = 60
batch_size = 512
ranger21 = True 
wandb = False
cycle = 5
latent_dims = [5,10,20]
note = "Fourth_version_running"
deep_matrixs = [False, True]
vaes = [False, False]
tags = ['Third']

for name, deep_matrix, vae in zip(names, deep_matrixs, vaes):
    for latent_dim in latent_dims:
        name_ = f"{name}_Latent{latent_dim}"
        cmd = f"python /home/dylan/repo/10617Project/code/main.py --notes {note} --model {model} --name {name_} --epochs {epochs} --lr_patience {lr_patience} --stop_patience {stop_patience} --batch_size {batch_size} --ranger21 {ranger21} --wandb {wandb} --cycle {cycle} --latent_dim {latent_dim} --deep_matrix {deep_matrix} --vae {vae} --tags {tags}"
        print(cmd)
        subprocess.call(cmd, shell=True)



# For model CNN 
model = 'CNN'
names = ['CNNAE', 'CNNAEWithDeepMatrix']
epochs = 1000
lr_patience = 150
stop_patience = 300
batch_size = 25
ranger21 = True
wandb = True
cycle = 8
latent_dims = [5,10,20]
deep_matrixs = [ False, True]
note = "Fourth_version_running"
vaes = [False, False]
vae_bn = 0

for name, deep_matrix, vae in zip(names, deep_matrixs, vaes):
    for latent_dim in latent_dims:
        name_ = f"{name}_Latent{latent_dim}"
        cmd = f"python /home/dylan/repo/10617Project/code/main.py --vae_bn {vae_bn} --notes {note} --model {model} --name {name_} --epochs {epochs} --lr_patience {lr_patience} --stop_patience {stop_patience} --batch_size {batch_size} --ranger21 {ranger21} --wandb {wandb} --cycle {cycle} --latent_dim {latent_dim} --deep_matrix {deep_matrix} --vae {vae} --tags {tags}"
        print(cmd)
        subprocess.call(cmd, shell=True)

