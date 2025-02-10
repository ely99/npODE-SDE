import os
import pandas as pd
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = False
import tensorflow as tf
sess = tf.InteractiveSession()

import numpy as np
from utils import gen_data, plot_model
from npde_helper import build_model, fit_model, save_model, load_model


def carica_dataset(percorso_dataset):
    print("LOADING")
    n_sim = 3
    n_data = 1 
    Nt = n_sim * n_data  # Numero totale di file (10 simulazioni * 10 file/simulazione)
    Y = np.zeros((Nt,), dtype=np.object)  # Inizializzazione di Y
    t = np.zeros((Nt,), dtype=np.object)  # Inizializzazione di t

    file_index = 0  # Indice per tenere traccia della posizione negli array
    for sim_id in range(n_sim):
        sim_folder = os.path.join(percorso_dataset, "SIM%d" % sim_id)
        for file_id in range(n_data):
            file_path = os.path.join(sim_folder, "data%d.csv" % file_id)
            df = pd.read_csv(file_path, header=None)
            traj = df.values[:, [2, 3]]  # Estrai la traiettoria
            Y[file_index] = traj  # Assegna la traiettoria a Y

            # Calcola il tempo per la traiettoria corrente e assegnalo a t
            n_rows = traj.shape[0]  # Numero di righe nella traiettoria
            t[file_index] = np.arange(0, n_rows * 0.02, 0.02)
            file_index += 1

    return Y, t



percorso_dataset = os.path.abspath(os.path.join(os.path.dirname(__file__), 'DATASET'))

#x0,t,Y,X,D,f,g = gen_data('vdp-cdiff', Ny=[35,40,30], tend=8, nstd=0.1)
Y, t = carica_dataset(percorso_dataset)


# t - observation times, a python array of numpy vectors
# Y - observations, a python array of numpy matrices, data points are in rows
# step size (eta) must be carefully tuned for different data sets
npde = build_model(sess, t, Y, model='sde', sf0=1.0, ell0=np.ones(2), W=6, ellg0=[1e5], ktype="id")
npde = fit_model(sess, npde, t, Y, Nw=100, num_iter=1000, print_every=50, eta=0.02, plot_=True)

save_model(npde,'npde_state_sde.pkl')
