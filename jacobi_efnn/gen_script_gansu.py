from pathlib import Path
# from decimal import Decimal, getcontext
import shutil
import numpy as np
import pandas as pd
import os


#this script generates slurm scripts on gansu

outPath="./bashFiles_gansu/"
if os.path.isdir(outPath):
    shutil.rmtree(outPath)
Path(outPath).mkdir(exist_ok=True,parents=True)

num_layers_vec=[3,5,7,9,11]
num_neurons_vec=[20,60,100,140,180]
N_vec=[800,1600,3200]
Q_vec=[2000,4000]
eps_vec=[1e-3,1e-4]
epochs=80000
layer_neuron_N_Q_eps_vec=[[layer,neuron,N,Q,eps] for layer in num_layers_vec
                      for neuron in num_neurons_vec
                      for N in N_vec
                      for Q in Q_vec
                      for  eps in eps_vec
                      ]
def contents_to_bash(num_layers,num_neurons,N,Q,eps,file_index):
    print(f"num_layers={num_layers}, num_neurons={num_neurons}")
    contents = [
        "#!/bin/bash\n",
        "#SBATCH -n 5\n",
        "#SBATCH -N 1\n",
        "#SBATCH -t 0-60:00\n",
        "#SBATCH -p lzicnormal\n",
        "#SBATCH --mem=50GB\n",
        f"#SBATCH -o out_jacobi_efnn_layer{num_layers}_neuron{num_neurons}.out\n",
        f"#SBATCH -e out_jacobi_efnn_layer{num_layers}_neuron{num_neurons}.err\n",
        "cd /public/home/hkust_jwliu_1/liuxi/Documents/pyCode/efnn_based_networks/jacobi_efnn\n",
        f"python3 -u train.py {epochs}  {num_layers} {num_neurons} {N} {Q} {eps}\n",
    ]
    outBashName = outPath + f"/train_hs_efnn_layer{num_layers}_neuron{num_neurons}.sh"
    with open(outBashName, "w+") as fptr:
        fptr.writelines(contents)



# Process each pair with its index
for file_index, val_item in enumerate(layer_neuron_N_Q_eps_vec):
    num_layers,num_neurons,N,Q,eps=val_item
    contents_to_bash(num_layers, num_neurons, N,Q,eps,file_index)