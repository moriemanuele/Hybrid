import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow import keras
from reservoir import *
import os
import sys
from aeon.datasets import load_classification
from sklearn.model_selection import train_test_split
from keras.datasets import mnist, fashion_mnist
from time import time, strftime, localtime
import matplotlib.pyplot as plt
from codecarbon import EmissionsTracker

from tslearn.datasets import UCR_UEA_datasets



print(f'sys.argv = {sys.argv}')
# Controlla se è stato passato un argomento
if len(sys.argv) <= 1:
    print("errore. non ho ricevuto input")
    exit(0)
print('num units = '+sys.argv[1])
print('dataset_name = '+sys.argv[2])
print('max_time = '+sys.argv[3])
print('gpu = '+sys.argv[4])
print('tentativo = '+sys.argv[5])




dataset_name = sys.argv[2]
max_time = int(sys.argv[3])
gpu = sys.argv[4]
tentativo = sys.argv[5]

os.environ["CUDA_VISIBLE_DEVICES"]=gpu #change this to the id of the GPU you can use (e.g., "2")


def permute_mnist(mnist, seed):
    np.random.seed(seed)
    perm_indices = np.random.permutation(784)
    return mnist[:, perm_indices]

def calcolo_gru(n_dim,n_units_gru):
    param_lib =3*((n_dim+n_units_gru)*n_units_gru+n_units_gru)
    return param_lib

def calcolo_param_add(n_units_mem,n_units_gru):
    n_param_add = n_units_mem + calcolo_gru(n_units_mem+1,n_units_gru)
    return n_param_add


#common experimental setting:
num_guesses = 3#1#3 #number of reservoir guesses for final evaluation after model selection
max_trials = 25#100#300#200#1000 #number of configurations to be sampled (randomly) during model selection
num_guesses_ms = 1 #number of guesses needed for model selection (in this case 1 is sufficient)

root_path = './'
datasets_path = os.path.join(root_path,'datasets')


num_units_3k = [
    (30,20), #0.9091673493385315, 137.55244255065918 seconds
    (25,20), #0.9167367815971375, 164.45119524002075 seconds
    (15,25), #0.9015979766845703, 175.69029426574707 seconds
    (22,22), #0.8948696255683899, 230.10365462303162 seconds solo 60 tentativi
    (5,30),
    (4,30),
    (3,30),
    (42,15),
    (180,5),
    (85,10)
] #la top è 30,20
num_units_8k = [ #la top è 35,35
    (35,35), #0.9133725762367249, 133.18904089927673
    (50,30), #0.9175778031349182, 470.11565232276917
    (25,40), # 0.8957107067108154, 364.47470808029175
    (10,45),
    (3,50),
    (500,5),
    (250,10)
    ] #20,40
num_units_16k = [
    (75,45), #0.8847771286964417, 420.3413972854614 seconds
    (50,50), #0.9201009273529053, 121.20062780380249
    (40,55), # 0.9217830300331116, 133.00883102416992
    (15,65),
    (95,40),
    (5,70),
    (10,67),
    (500,10),
    (1000,5)
    ]
num_units = [
    num_units_3k,
    num_units_8k,
    num_units_16k
    ]

hyperparameters_test_Beef_3k = {
    'best_units_m' : 4,
    'best_units_gru' : 30,
    'best_lr' : 0.01,
    'best_patience' : 12,
    'best_num_epochs' : 15,
    'best_batch_size' : 128,
    'best_batch_size_gru' : 4,
    'best_leaky': 0.1,
    "best_memory_scaling" : 1.0,
    "best_input_scaling" : 0.1,
    "best_bias_scaling" : 0.5,
    "best_spectral_radius" : 0.8,
    "best_leaky_m" : 1,
    "best_input_scaling_m" : 0.5,
    "best_spectral_radius_m" : 0.9,
    "best_bias_scaling_m" : 1.0
}

hyperparameters_test_Beef_8k = {
    'best_units_m' : 50,
'best_units_gru' : 30,
'best_lr' : 0.01,
'best_patience' : 50,
'best_num_epochs' : 200,
'best_batch_size' : 128,
'best_batch_size_gru' : 16,
'best_leaky': 0.1,
"best_memory_scaling" : 0.1,
"best_input_scaling" : 1.0,
"best_bias_scaling" : 1.0,
"best_spectral_radius" : 0.8,
"best_leaky_m" : 1,
"best_input_scaling_m" : 0.5,
"best_spectral_radius_m" : 0.8,
"best_bias_scaling_m" : 0.5
}

hyperparameters_test_Beef_16k = {
    'best_units_m' : 5,
'best_units_gru' : 70,
'best_lr' : 0.0001,
'best_patience' : 40,
'best_num_epochs' : 50,
'best_batch_size' : 128,
'best_batch_size_gru' : 16,
'best_leaky': 1,
"best_memory_scaling" : 0.5,
"best_input_scaling" : 0.1,
"best_bias_scaling" : 0.1,
"best_spectral_radius" : 0.9,
"best_leaky_m" : 1,
"best_input_scaling_m" : 0.5,
"best_spectral_radius_m" : 1.0,
"best_bias_scaling_m" : 0.1
}

hyperparameters_test_Meat_3k = {
    'best_units_m' : 85,
'best_units_gru' : 10,
'best_lr' : 0.0001,
'best_patience' : 375,
'best_num_epochs' : 500,
'best_batch_size' : 128,
'best_batch_size_gru' : 4,
'best_leaky': 0.01,
"best_memory_scaling" : 0.5,
"best_input_scaling" : 0.5,
"best_bias_scaling" : 0.1,
"best_spectral_radius" : 0.8,
"best_leaky_m" : 1,
"best_input_scaling_m" : 1.0,
"best_spectral_radius_m" : 1.0,
"best_bias_scaling_m" : 0.5,
}

hyperparameters_test_Meat_8k = {
    'best_units_m' : 3,
'best_units_gru' : 50,
'best_lr' : 1e-05,
'best_patience' : 375,
'best_num_epochs' : 500,
'best_batch_size' : 128,
'best_batch_size_gru' : 64,
'best_leaky': 0.1,
"best_memory_scaling" : 1.0,
"best_input_scaling" : 0.1,
"best_bias_scaling" : 0.1,
"best_spectral_radius" : 0.8,
"best_leaky_m" : 1,
"best_input_scaling_m" : 0.1,
"best_spectral_radius_m" : 0.8,
"best_bias_scaling_m" : 0.1,
}

hyperparameters_test_Meat_16k = {
    'best_units_m' : 1000,
'best_units_gru' : 5,
'best_lr' : 0.001,
'best_patience' : 12,
'best_num_epochs' : 15,
'best_batch_size' : 128,
'best_batch_size_gru' : 64,
'best_leaky': 1,
"best_memory_scaling" : 0.5,
"best_input_scaling" : 1.0,
"best_bias_scaling" : 0.1,
"best_spectral_radius" : 0.8,
"best_leaky_m" : 1,
"best_input_scaling_m" : 0.5,
"best_spectral_radius_m" : 0.8,
"best_bias_scaling_m" : 0.5,
}

hyperparameters_test_Wine_3k = {
'best_units_m' : 3,
'best_units_gru' : 30,
'best_lr' : 0.0001,
'best_patience' : 12,
'best_num_epochs' : 15,
'best_batch_size' : 128,
'best_batch_size_gru' : 8,
'best_leaky': 0.01,
"best_memory_scaling" : 0.1,
"best_input_scaling" : 0.5,
"best_bias_scaling" : 0.5,
"best_spectral_radius" : 0.9,
"best_leaky_m" : 0.1,
"best_input_scaling_m" : 1.0,
"best_spectral_radius_m" : 0.8,
"best_bias_scaling_m" : 0.1,
}
hyperparameters_test_Wine_8k = {
'best_units_m' : 10,
'best_units_gru' : 45,
'best_lr' : 0.0001,
'best_patience' : 40,
'best_num_epochs' : 50,
'best_batch_size' : 128,
'best_batch_size_gru' : 32,
'best_leaky': 0.1,
"best_memory_scaling" : 0.5,
"best_input_scaling" : 0.1,
"best_bias_scaling" : 1.0,
"best_spectral_radius" : 0.9,
"best_leaky_m" : 0.1,
"best_input_scaling_m" : 0.5,
"best_spectral_radius_m" : 0.8,
"best_bias_scaling_m" : 1.0,
}
hyperparameters_test_Wine_16k = {
'best_units_m' : 1000,
'best_units_gru' : 5,
'best_lr' : 0.1,
'best_patience' : 375,
'best_num_epochs' : 500,
'best_batch_size' : 128,
'best_batch_size_gru' : 16,
'best_leaky': 1,
"best_memory_scaling" : 0.5,
"best_input_scaling" : 0.5,
"best_bias_scaling" : 0.5,
"best_spectral_radius" : 1.0,
"best_leaky_m" : 0.1,
"best_input_scaling_m" : 0.5,
"best_spectral_radius_m" : 0.8,
"best_bias_scaling_m" : 0.5,
}

hyperparameters_test_Trace_3k = {
'best_units_m' : 4,
'best_units_gru' : 30,
'best_lr' : 0.0001,
'best_patience' : 100,
'best_num_epochs' : 200,
'best_batch_size' : 128,
'best_batch_size_gru' : 8,
'best_leaky': 0.1,
"best_memory_scaling" : 0.5,
"best_input_scaling" : 0.1,
"best_bias_scaling" : 0.5,
"best_spectral_radius" : 1.0,
"best_leaky_m" : 0.1,
"best_input_scaling_m" : 1.0,
"best_spectral_radius_m" : 1.0,
"best_bias_scaling_m" : 1.0,
}
hyperparameters_test_Trace_8k = {
'best_units_m' : 35,
'best_units_gru' : 35,
'best_lr' : 0.0001,
'best_patience' : 150,
'best_num_epochs' : 200,
'best_batch_size' : 128,
'best_batch_size_gru' : 4,
'best_leaky': 0.1,
"best_memory_scaling" : 1.0,
"best_input_scaling" : 0.5,
"best_bias_scaling" : 0.1,
"best_spectral_radius" : 0.8,
"best_leaky_m" : 0.01,
"best_input_scaling_m" : 0.5,
"best_spectral_radius_m" : 0.8,
"best_bias_scaling_m" : 1.0,
}
hyperparameters_test_Trace_16k = {
'best_units_m' : 95,
'best_units_gru' : 40,
'best_lr' : 0.001,
'best_patience' : 4,
'best_num_epochs' : 5,
'best_batch_size' : 128,
'best_batch_size_gru' : 128,
'best_leaky': 0.1,
"best_memory_scaling" : 0.1,
"best_input_scaling" : 1.0,
"best_bias_scaling" : 0.1,
"best_spectral_radius" : 1.0,
"best_leaky_m" : 0.1,
"best_input_scaling_m" : 0.5,
"best_spectral_radius_m" : 0.8,
"best_bias_scaling_m" : 0.1,
}



if sys.argv[1] == '3':
    units = num_units_3k
    if dataset_name == 'Beef':
        nome_hyperparameters = hyperparameters_test_Beef_3k
    elif dataset_name == 'Wine':
        nome_hyperparameters = hyperparameters_test_Wine_3k
    elif dataset_name == 'Trace':
        nome_hyperparameters = hyperparameters_test_Trace_3k
    elif dataset_name == 'Meat':
        nome_hyperparameters = hyperparameters_test_Meat_3k   
elif sys.argv[1] == '8': 
    units = num_units_8k
    if dataset_name == 'Beef':
        nome_hyperparameters = hyperparameters_test_Beef_8k
    elif dataset_name == 'Wine':
        nome_hyperparameters = hyperparameters_test_Wine_8k
    elif dataset_name == 'Trace':
        nome_hyperparameters = hyperparameters_test_Trace_8k
    elif dataset_name == 'Meat':
        nome_hyperparameters = hyperparameters_test_Meat_8k 
elif sys.argv[1] == '16': 
    units = num_units_16k
    if dataset_name == 'Beef':
        nome_hyperparameters = hyperparameters_test_Beef_16k
    elif dataset_name == 'Wine':
        nome_hyperparameters = hyperparameters_test_Wine_16k
    elif dataset_name == 'Trace':
        nome_hyperparameters = hyperparameters_test_Trace_16k
    elif dataset_name == 'Meat':
        nome_hyperparameters = hyperparameters_test_Meat_16k 

#num_unit = units[np.random.randint(low = 0, high=len(units))]


#units_m = num_unit[0]
#units_gru = num_unit[1]
#num_unit = units[0]

Nh_ms = 5
Nh_assessment = 10
"""units_gru = 15
units_m = 20"""

if dataset_name == 'psMNIST' or dataset_name == 'sMNIST' or dataset_name == 'FordA' or dataset_name == 'FordB':
    (x_train_all, y_train_all), (x_test, y_test) = mnist.load_data()
    x_train_all  = np.reshape(x_train_all,(x_train_all.shape[0], -1, 1))
    x_test  = np.reshape(x_test,(x_test.shape[0], -1, 1))

    x_train_all = (x_train_all / 255.0)
    x_test = (x_test / 255.0)

    if dataset_name == 'psMNIST':
        x_train_all = permute_mnist(x_train_all, seed=0)
        x_test = permute_mnist(x_test, seed=0)

    x_train, x_val, y_train, y_val = train_test_split(x_train_all, y_train_all, test_size=0.33, random_state=42, stratify = y_train_all)



elif dataset_name == 'Trace' or dataset_name == 'KeplerLightCurves' or dataset_name == 'Meat' or dataset_name == 'Wine' or dataset_name == 'Beef':
    # Istanza dell'oggetto UCR_UEA_datasets
    ucr = UCR_UEA_datasets()

    # Scaricare i dataset
    dataset = ucr.load_dataset(dataset_name)
    if dataset is None:
        raise ValueError("Il dataset non è stato caricato correttamente.")
        

    x_train_all, y_train_all, x_test, y_test = dataset
    x_train, x_val, y_train, y_val = train_test_split(x_train_all, y_train_all, test_size=0.33, random_state=42, stratify = y_train_all)




experiment_name = 'Hybrid'
model_type = 'Hybrid'
pid = os.getpid()
#create the results directory
results_path = os.path.join(root_path, 'results',dataset_name)
#create the results path if it does not exists
if not os.path.exists(results_path):
    os.makedirs(results_path)
root_path = './'
# Stampa il PID
#nome = str(calcolo_param_add(units_m,units_gru))+'_'+str(units_m)+'_'+str(units_gru)+'_'+tentativo
nome = sys.argv[1]+'_'+tentativo

res = "Sono Hybrid_"+dataset_name+'_'+nome+", pid "+ str(pid)
print(res)
outputs_path = os.path.join(root_path, 'outputsTEST',dataset_name)
#create the results path if it does not exists
if not os.path.exists(outputs_path):
    os.makedirs(outputs_path)

nomefileoutput = os.path.join(outputs_path,'Hybrid_'+dataset_name+'_'+nome+'.txt')
output = open(nomefileoutput,'w')
print(res,file=output, flush=True)
results_logger_filename = os.path.join(results_path,'results_logger_'+model_type+nome+'.txt')

results_logger = open(results_logger_filename,'w')
results_logger.write('Experiment with '+model_type+' on dataset '+ dataset_name +nome+ ' starting now\n')
time_string_start = strftime("%Y/%m/%d %H:%M:%S", localtime())
results_logger.write('** local time = '+ time_string_start+'\n')
results_logger.flush()


emissions_path = os.path.join(root_path,'emissions',dataset_name)
if not os.path.exists(emissions_path):
    os.makedirs(emissions_path)
nomefileemissioni = os.path.join(emissions_path,dataset_name+'_'+sys.argv[1]+'_'+tentativo)
fileemissions = open(nomefileemissioni,'w')


initial_model_selection_time = time() #start the timer for the model selection
best_val_score = 0
model_type = 'Hybrid'
tentativi = 0
accuracy = []
required_time = []

tracker = EmissionsTracker(project_name='Hybrid_'+dataset_name+'_'+sys.argv[1]+str(tentativo), measure_power_secs=5, tracking_mode = 'process')
tracker.start_task('Hybrid_'+dataset_name+'_'+sys.argv[1]+'_'+sys.argv[5])
for i in range(num_guesses):
    tempo_inizio_trial = strftime("%Y/%m/%d %H:%M:%S", localtime())
    begin_time = time()
    print("tempo inizio trial numero "+str(i)+": " + tempo_inizio_trial,file=output, flush=True)

    if (i % 100 == 0):
        print('* testing the {}-th {} configuration'.format(i, model_type))
    
    if (time()-initial_model_selection_time > max_time):
        print('--> terminating the model selection for exceeding max time after {} configurations.\n'.format(i))
        results_logger.write('--> terminating the model selection for exceeding max time after {} configurations.\n'.format(i))
        results_logger.flush()
        break
    #num_unit = units[np.random.randint(low = 0, high=len(units))]
    units_m = nome_hyperparameters['best_units_m']
    units_gru = nome_hyperparameters['best_units_gru']
    leaky = nome_hyperparameters['best_leaky']
    spectral_radius = nome_hyperparameters['best_spectral_radius']
    input_scaling = nome_hyperparameters['best_input_scaling']
    memory_scaling = nome_hyperparameters['best_memory_scaling']
    bias_scaling = nome_hyperparameters['best_bias_scaling']
    lr = nome_hyperparameters['best_lr']
    batch_size_gru = nome_hyperparameters['best_batch_size_gru']
    batch_size = nome_hyperparameters['best_batch_size']
    num_epochs = nome_hyperparameters['best_num_epochs']
    patience = nome_hyperparameters['best_patience']
    leaky_m = nome_hyperparameters['best_leaky_m']
    bias_scaling_m = nome_hyperparameters['best_bias_scaling_m']
    spectral_radius_m = nome_hyperparameters['best_spectral_radius_m']
    input_scaling_m = nome_hyperparameters['best_input_scaling_m']

    output_units = max(y_train_all)+1
    features_dim = 1
    if (output_units==2):
        output_units = 1
    if (output_units==1):
        output_activation = 'sigmoid'
        loss_function = 'binary_crossentropy'
    else:
        output_activation = 'softmax'#'sigmoid'
        loss_function = 'sparse_categorical_crossentropy'
                
    val_score = 0
    
    tempo_inizio_guess = strftime("%Y/%m/%d %H:%M:%S", localtime())

    memory = keras.Sequential([
            #linear memory component
            keras.Input(batch_input_shape = (batch_size, None, features_dim)),
            keras.layers.RNN(cell = RingReservoirCell(units = units_m,
                                                    input_scaling = input_scaling_m,
                                                    bias_scaling = bias_scaling_m,
                                                    spectral_radius = spectral_radius_m,
                                                    leaky = leaky_m,
                                                    activation = None), #linear layer
                                return_sequences = True, stateful=True)
    ])

    
    trainable = keras.Sequential([
        keras.layers.GRU(units = units_gru),#, input_shape=(500, 6)),
        keras.layers.Dense(output_units, activation = output_activation)
    ])

    trainable.compile(optimizer=keras.optimizers.Adam(learning_rate = lr),loss=loss_function,metrics=['accuracy'])
        
    import io
    from contextlib import redirect_stdout

    """# Cattura il sommario del modello in una stringa
    with io.StringIO() as buf, redirect_stdout(buf):
        trainable.summary()
        trainable_summary = buf.getvalue()

    print(trainable_summary, file=output, flush=True)"""

    sys.setrecursionlimit(10000)

    x = x_train_all
    y = y_train_all
    batch_size = batch_size #128
    num_batches = int(np.ceil(x.shape[0] / batch_size)) # 19

    #x_train_mem_all = np.zeros(shape=(num_batches,x.shape[1],self.units_m+1)) #shape nb x n time steps x nm+1, che è l'input da dare alla gru
    x_train_mem_all = np.zeros(shape=(x.shape[0],x.shape[1],units_m+1)) #shape nb x n time steps x nm+1, che è l'input da dare alla gru



    #scorro i batches
    for i in range(num_batches):
        #reset the state of self.memory recurrent layer
        memory.reset_states()
        #self.trainable.reset_states() #commentato pk mi serve allenato
        xlocal = x[i*batch_size:(i+1)*batch_size,:,:]
        
        original_shape = xlocal.shape
        if xlocal.shape[0] < batch_size:
            xlocal = np.concatenate((xlocal, np.zeros((batch_size - xlocal.shape[0],xlocal.shape[1],xlocal.shape[2]))), axis = 0)
        
        tensors = []
        for t in range(xlocal.shape[1]):
            memory_reservoir_states = memory(xlocal[:,t:t+1,:]) #(128,1,5)
            concatenated_input = np.concatenate((memory_reservoir_states,xlocal[:,t:t+1,:]), axis = -1) #(128,1,6)
            tensors.append(concatenated_input)
        result_x = np.concatenate(tensors, axis = 1) #(128,500,6)
        result_y = y[i*batch_size:(i+1)*batch_size]

        #xlocal = x[i*batch_size:(i+1)*batch_size,:,:]
        
        #original_shape = xlocal.shape
        #if xlocal.shape[0] < batch_size:
            #xlocal = np.concatenate((xlocal, np.zeros((batch_size - xlocal.shape[0],xlocal.shape[1],xlocal.shape[2]))), axis = 0)
        
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        if end_idx <= x_train_mem_all.shape[0]:  # Controlla se l'indice è entro i limiti
            x_train_mem_all[start_idx:end_idx, :, :] = result_x
        
        if xlocal.shape[0]<batch_size:
            x_train_mem_all[-xlocal.shape[0]:, :, :] = result_x[:xlocal.shape[0], :, :]
        
        """trainable.fit(x_train_mem_all, y,
                            verbose = 0, 
                            epochs = num_epochs,
                            batch_size = batch_size_gru,
                            callbacks=[keras.callbacks.EarlyStopping(monitor='loss'#'val_loss'
                                                                    , patience = patience)]
                        )"""
    


    x = x_test
    y = y_test
    x_val_mem_all = np.zeros(shape=(x.shape[0],x.shape[1],units_m+1)) #shape nb x n time steps x nm+1, che è l'input da dare alla gru
    for i in range(num_batches):
        #print('batch * ', i)
        #reset the state of self.memory recurrent layer
        memory.reset_states()
        #self.trainable.reset_states() #È CAMBIATA, lo commento pk mi serve allenato x lo score
        xlocal = x[i*batch_size:(i+1)*batch_size,:,:]
        #if xlocal is smaller tahn the batch_size we need to pad it
        original_shape = xlocal.shape
        if xlocal.shape[0] < batch_size:
            xlocal = np.concatenate((xlocal, np.zeros((batch_size - xlocal.shape[0],xlocal.shape[1],xlocal.shape[2]))), axis = 0)

        tensors = []
        for t in range(xlocal.shape[1]):
            memory_reservoir_states = memory(xlocal[:,t:t+1,:]) #(128,1,5)
            concatenated_input = np.concatenate((memory_reservoir_states,xlocal[:,t:t+1,:]), axis = -1) #(128,1,6)
            tensors.append(concatenated_input)
        result_x = np.concatenate(tensors, axis = 1) #(128,500,6)
        result_y = y[i*batch_size:(i+1)*batch_size]

        #xlocal = x[i*batch_size:(i+1)*batch_size,:,:]
        
        #original_shape = xlocal.shape
        #if xlocal.shape[0] < batch_size:
            #xlocal = np.concatenate((xlocal, np.zeros((batch_size - xlocal.shape[0],xlocal.shape[1],xlocal.shape[2]))), axis = 0)
        
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        if end_idx <= x_val_mem_all.shape[0]:  # Controlla se l'indice è entro i limiti
            x_val_mem_all[start_idx:end_idx, :, :] = result_x
        
        if xlocal.shape[0]<batch_size:
            x_val_mem_all[-xlocal.shape[0]:, :, :] = result_x[:xlocal.shape[0], :, :]
    
    history = trainable.fit(x_train_mem_all, y_train_all,
                        verbose = 0, 
                        epochs = num_epochs,
                        batch_size = batch_size_gru,
                        validation_data = (x_val_mem_all,y_test),
                        callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss'#'val_loss'
                                                                , patience = patience)]
                    )
    loss, val_accuracy = trainable.evaluate(x_val_mem_all, y_test, verbose=0) 
    #_,val_score_new = trainable.evaluate(x_val_mem_all,y_val, verbose = 0) 
    print(f"Accuracy sul test set: {val_accuracy:.2f}", file=output, flush=True)

    # Accesso ai valori di accuracy per epoca
    train_accuracy = history.history['accuracy']  # Accuracy durante il training
    val_accuracy = history.history['val_accuracy']  # Accuracy durante il validation

    # Stampa o visualizza graficamente l'accuracy
    print(f"Accuracy finale su training set: {train_accuracy[-1]:.2f}", file=output, flush=True)
    print(f"Accuracy finale su test set: {val_accuracy[-1]:.2f}", file=output, flush=True)

    # Visualizzazione grafica dell'accuracy
    plt.plot(train_accuracy, label='Training Accuracy')
    plt.plot(val_accuracy, label='Validation Accuracy')
    plt.title('Accuracy durante il training')
    plt.xlabel('Epoca')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    plt.savefig('accuracy_plot.png')  # Specifica il nome del file e il formato

    #val_score = val_score + val_score_new
    tempo_fine_guess = strftime("%Y/%m/%d %H:%M:%S", localtime())
    val_score = val_accuracy[0]#val_score / num_guesses_ms
    end_time = time()  
    durata = end_time-begin_time

    print(f'val_score = {val_score}, time = {durata}', file=output, flush=True)
    accuracy.append(val_score)
    required_time.append(durata)

        
print('--'+model_type+' on {}--'.format(dataset_name),file=output, flush=True)
print('Results: MEAN {} STD {}'.format(np.mean(accuracy),np.std(accuracy)),file=output, flush=True)
print('----- required time: MEAN {} STD {}'.format(np.mean(required_time),np.std(required_time)),file=output, flush=True)
emissions = tracker.stop_task()
_ = tracker.stop()
print(emissions,file = fileemissions, flush = True)
output.close()
results_logger.close()

