import os, time
import pandas as pd
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = False


import tensorflow as tf
sess = tf.InteractiveSession()
FLAG= False
import numpy as np
from utils import gen_data, plot_model
from npde_helper import build_model, fit_model, save_model, load_model
def carica_dataset(percorso_dataset):
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

if FLAG:

    npde = load_model('npde_state_sde.pkl',sess)
    #plot_model(npde,t,Y)

    x0 = [0,0]
    t = np.linspace(0,20,100)
    Nw = 3 # number of samples

    start_time = time.time()
    samples = npde.sample(x0,t,Nw)
    samples = sess.run(samples)
    end_time = time.time()  
    tempo_esecuzione = end_time - start_time
    print("Tempo di esecuzione: %f secondi" % tempo_esecuzione)

    plt.figure(figsize=(12,5))
    for i in range(Nw):
        plt.plot(t,samples[i,:,0],'-k',linewidth=0.25)
        plt.plot(t,samples[i,:,1],'-r',linewidth=0.25)
    plt.xlabel('time',fontsize=12)
    plt.ylabel('states',fontsize=12)
    plt.title('npSDE samples',fontsize=16)
    plt.savefig('samples.png', dpi=200)
    plt.show()
else:
    npde = load_model('npde_state.pkl',sess)
    x0 = [0,0] # initial value
    t = np.linspace(0,20,100) # time points 
    path = npde.predict(x0,t)
    path = sess.run(path)

    plt.figure(figsize=(12,5))
    plt.plot(t,path)
    plt.xlabel('time',fontsize=12)
    plt.ylabel('states',fontsize=12)
    plt.title('npSDE mean future predictions',fontsize=16)
    plt.show()

