import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import qiskit.quantum_info as qi
from src.dip_pdip import Dip_Pdip
from qiskit.quantum_info import Statevector
import os
from qiskit import transpile
from qiskit.visualization import circuit_drawer
from qiskit_aer import Aer
from qiskit import QuantumCircuit
from scipy.optimize import minimize
from src.result import Result
import matplotlib.pyplot as plt
import time
from datetime import datetime
import random

class Main:

    def __init__(self,start_time):
        self.start_time = start_time
        self.min = 1
        self.res = Result()
        self.c1Array1Layer = []
        self.DLambda = []
        self.totLayer = 5
        self.testTwoStates()

    def testTwoStates(self):
        a = np.array([1/np.sqrt(2), 0 ,0,1/np.sqrt(2)])
        b = np.array([0,1/np.sqrt(2), 1/np.sqrt(2),0])
        c = np.array([1/np.sqrt(2),0, 1/np.sqrt(2) ,0])
        d = np.array([1/np.sqrt(3), 0,1/np.sqrt(3) ,1/np.sqrt(3)])
        e = np.array([1/np.sqrt(2), 1/np.sqrt(2) ,0,0])
        self.imgIniziale = a
        batchBinary = [a]
        self.rho = self.getRhoMath(batchBinary)
        self.num_qubits = 2
        self.res.num_qubits = self.num_qubits
        self.toFind = True
        self.num_layer = 2
        self.res.num_layers = self.num_layer
        self.timeRun = time.time()
        self.a = self.paramsStandard()
        self.counter = 1
        self.min = 1
        while(self.toFind is True):
                #print("Inizio un ciclo di ottimizzazione con n layer: ", self.num_layer)
            self.c1_optimization(batchBinary)
            self.c1Array1Layer.append(self.min)
        c1Array1Layer = np.array(self.c1Array1Layer)  # Converte in array NumPy
        print("array:", c1Array1Layer)

    


    def MNISTTest(self):
        batchStates, batchBinary = self.res.load_img(65)
        self.rho = self.getRhoMath(batchBinary)
        #y = batchBinary[0].shape[0] * batchBinary[0].shape[0]
        self.num_qubits = int(np.log2(len(batchBinary[0])))
        #self.num_qubits = int(np.log2(y))
        self.res.num_qubits = self.num_qubits
        self.num_layer = 3
        self.toFind = True
        self.res.num_layers = self.num_layer
        self.a = self.paramsStandard()
        self.timeRun = time.time()
            #print("Ora lavoro con n_layer: ", self.num_layer)
        self.counter = 1
        while(self.toFind is True):
            print("Num layer: ", self.num_layer)
            self.c1_optimization(batchBinary)
            self.c1Array1Layer.append(self.min) 
        c1Array1Layer = np.array(self.c1Array1Layer)  
        print("array:", c1Array1Layer)

    def pca_from_T(self,T):
        
        eigenvalues, U_T = np.linalg.eigh(T)
        
        
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        U_T = U_T[:, idx]
        
        Lambda_T = np.diag(eigenvalues)
        
        print('Autovettori: ')
        print(U_T)
        print('Autovalori: ', eigenvalues)
        print('Lambda_T:')
        print(Lambda_T )

    def paramsStandard(self):
        
        if self.num_qubits == 6 and self.num_layer == 3:#Se uso lo 0 da MNSIT
            #per 0 e 7
            data_list = [-3.4303, 0.0529, 1.5043, -0.9789, -1.3242, 2.5626, 0.1839, -2.9897, 1.6978, 1.2098, -1.2082, 2.9435, 2.7008, 0.7135, 2.7236, 1.3226, -1.5448, 2.3656, 0.2288, 2.8213, 2.2226, -1.797, -0.7605, 0.644, -1.1354, -2.8242, 2.5641, -1.1049, -2.7954, -1.8391, -1.5919, -2.106, 1.0983, -1.8143, 2.7647, -0.6653, 2.4634, 0.3005, -1.5956, -1.1268, -1.8003, 2.6279, 2.7314, -1.2663, -1.411, -1.7982, 1.6135, 0.3495, -1.8663, -1.7149, -0.8408, -1.0719, -0.1266, -2.251, -0.126, 0.0997, -1.4532, 1.8221, 0.5958, 2.3247, -0.1079, -2.1351, 0.1534, 2.456, -3.2946, -2.1624, -2.5935, 1.4077, 2.9927, -1.6982, -0.078, 3.0321, -2.0341, 0.4735, -1.5401, 2.7443, -1.7639, 2.2846, -1.205, -2.2631, -2.0621, 0.5173, -3.1594, -1.9718, -1.5924, 1.2644, -0.4314, -0.3493, -2.9163, -1.422, -0.7807, 1.0163, 4.0447, 0.5578, 0.7594, -3.1219, 2.6851, -0.3403, -0.9537, -1.6858, 0.2895, 2.6581, -2.8574, 2.0926, 1.2358, 0.3165, -0.0318, 0.7374, -0.5303, 2.2192, -2.0581, -1.1776, 0.3661, 1.7587, 2.7502, 2.7938, -1.8958, -1.3114, 1.6004, 2.1901, 2.4713, 1.2514, -0.0098, 0.9056, -1.3445, -1.0965, -1.3511, -1.5336, 1.3074, 2.1535, -1.597, 2.9764, -2.3068, 0.9977, -1.6774, 1.1786, -2.1408, -0.3592, -1.4986, 1.6022, -1.9729, 0.5384, 2.7768, -2.7119, 1.4904, 2.1244, 0.0211, 1.6448, 0.2469, 2.6461, 1.788, 3.0451, -2.0095, -0.6389, -0.6078, -2.2146, 1.951, 2.3453, 0.5079, -1.1223, -2.591, -1.4493, 1.1616, 0.9114, 2.0688, -2.6265, 1.9473, 1.3555, -1.4972, 0.8273, -0.3099, 0.4214, 1.6014, -1.3026, 1.2826, -0.7133, 2.3366, -1.2302, -1.5197, -1.1355, 1.5725, 1.2358, -0.0042, 0.506, -0.2098, 1.0108, -0.0666, -2.0169, 4.048, -0.0451, -1.0237, 3.232, -2.5393, 1.9612, 2.3981, 1.4633, 0.4017, 2.4867, -2.3338, 2.5855, 2.5112, 1.5824, 1.2811, -2.589, 3.0261, -0.1715, 4.486, 0.7788, 1.8908, -2.6274, -0.0595, 2.6808, 1.8305, 0.6877, -3.2681, 0.9944]

            #data_list = [ 1.51386077, -1.70203324, 1.85607109, 2.30895355, -1.36081905, -1.25963548,3.42352944, -1.44315902, 1.31116192, 1.27885328, 15.09812813, 1.47683429,0.8990097, -3.14073329, -1.14785431, 0.47205453, -3.17577445, -1.57255897, 0.64787145, 0.29711341, -0.64651245, -1.5391103, -1.65680379, -1.5301107, 22.46992636, 1.5394186, 5.6831592, 1.70814087, 1.97813529, 2.71828155,3.2970164, -3.17469452, 12.13407952, 0.38293945, -0.62746048, 9.27192198,-0.19312788, 0.51474739, 1.07061587, 1.57391989, -2.85726607, -2.50930985,-1.5919183, 0.27346589, 7.23699726, 2.39371459, 3.20194722, 6.82350133]

            x = np.array(data_list).reshape(self.num_layer,2 ,self.num_qubits//2 ,12)
            x_reshaped = x.reshape(self.num_layer, 2, self.num_qubits // 2, 4, 3)
            return x_reshaped
        
        elif self.num_qubits == 4 and self.num_layer == 1:#semrpe con lo 0 scritto su a mano
            data_list = [
  1.33286165, -1.6360426,  1.89911712,  2.54946114, -1.33457995, -1.22788508, 
  3.06576776, -1.41017854,  1.62244655,  1.25314897, 13.12578229,  1.57175219, 
  0.8089642, -3.11592479, -1.17070079,  0.51786867, -3.1820048,  -1.51360945, 
  0.64535919,  0.19976472, -0.64992413, -1.78013715, -1.76122462, -1.24101147, 
  16.55343939,  1.52994246,  2.75772913,  1.68809309,  1.96429889,  2.76975141, 
  3.28671663, -3.08838081, 10.17562557,  0.37828878, -0.68246838,  7.64837006, 
  -0.31682686,  0.48200567,  0.999498,    1.63484914, -2.82263382, -2.51945769, 
  -1.57851123,  0.19782738,  5.0980062,   2.40993532,  3.11738655,  3.72040611
]

            x = np.array(data_list).reshape(self.num_layer,2 ,self.num_qubits//2 ,12)
            x_reshaped = x.reshape(self.num_layer, 2, self.num_qubits // 2, 4, 3)
            return x_reshaped
        elif self.num_qubits == 2 and self.num_layer == 1:
            data_list = [
    0.30769997, 3.16212, -0.04130119,
    2.53405638, -1.70093959, -1.72562181,
    2.63348584, 2.13674309, 0.57736903,
    -1.23230544, -0.78641796, -2.16476545,
    -0.46965461, 1.52427227, -2.37878391,
    -2.79052192, 2.02044089, 0.51667296,
    -1.33488655, -0.46251497, 1.15919535,
    -0.45364059, -0.28087855, -1.08811782
]
            x = np.array(data_list).reshape(self.num_layer,2 ,self.num_qubits//2 ,12)
            x_reshaped = x.reshape(self.num_layer, 2, self.num_qubits // 2, 4, 3)
            return x_reshaped
        else:
            return self.res.get_params(self.num_qubits, self.num_layer)


    def plot(self):
        # Creazione del plot
        plt.figure(figsize=(10, 6))
        # Plot per C1 finale

        c1Array1Layer = np.array(self.c1Array1Layer)  # Converte in array NumPy
        c1Array2Layer = np.array(self.c1Array2Layer)  # Converte in array NumPy
        c1Array3Layer = np.array(self.c1Array3Layer)  # Converte in array NumPy

        max_dim = max(c1Array1Layer.shape[0], c1Array2Layer.shape[0], c1Array3Layer.shape[0])

        # Riempi l'array più piccolo con l'ultimo valore dell'array più grande
        if c1Array1Layer.shape[0] < max_dim:
            c1Array1Layer = np.pad(c1Array1Layer, (0, max_dim - c1Array1Layer.shape[0]), mode='edge')
        elif c1Array2Layer.shape[0] < max_dim:
            c1Array2Layer = np.pad(c1Array2Layer, (0, max_dim - c1Array2Layer.shape[0]), mode='edge')
        elif c1Array3Layer.shape[0] < max_dim:
            c1Array3Layer = np.pad(c1Array3Layer, (0, max_dim - c1Array3Layer.shape[0]), mode='edge')    

# Crea iterations partendo da 0 fino alla lunghezza dell'array
        iterations = np.arange(len(c1Array1Layer))  # Questo crea [0, 1, 2, ...]

        print("c1Array1Layer:", c1Array1Layer)
        print("c1Array2Layer:", c1Array2Layer)
        print("c1Array3Layer:", c1Array3Layer)

        plt.plot(iterations, c1Array1Layer.real, marker='o', linestyle='-', color='b', label='C con 1 layer')
        plt.plot(iterations, c1Array2Layer.real, marker='o', linestyle='-', color='g', label='C con 2 layer')
        plt.plot(iterations, c1Array3Layer.real, marker='o', linestyle='-', color='r', label='C con 3 layer')

        #plt.plot(layers, DMaggioreArray.real, marker='s', linestyle='--', color='r', label='Elementi di D maggiori')
        #plt.plot(layers, OverlapArray.real, marker='o', linestyle='-', color='g', label='Overlap')

        # Dettagli del grafico
        plt.xlabel('Numero di Layer')
        plt.ylabel('Valore')
        plt.title('Quantità in funzione del numero di layer')
        plt.grid(True)
        plt.legend()

        # Mostra il grafico
        plt.show()

    def plot10Array(self):
        pass
        
    def testDipPurity(self,vector):        
        vectorized_matrix = vector.flatten()
        print(vectorized_matrix)
        y = self.res.c1_calc(vectorized_matrix)
        print("Risultato matematico C1: ",y)
        return y


    def getUnitaria(self,qc):
        #self.res.printCircuit(qc)
        return qi.Operator(qc)
    
    def c1_optimization(self, batch):
        # Appiattisci i parametri solo per la fase di ottimizzazione
        
        flat_params = np.ravel(self.a)
        
        # Passa i parametri appiattiti a minimize con i vincoli
        """Devo rendere stati i miei batch"""
        k = []  # Inizializzi come lista, non come dizionario
        for i in range(len(batch)):
            a = batch[i].flatten()
            k.append(self.getQc(a)) 
            #self.printCircuit(self.getQc(a))
         # Usa append sulla lista
        result = minimize(self.c1, flat_params, args=(k,), method="Powell", tol=1e-2) 
        print("terminato ciclo con: ", int((time.time() - self.timeRun)/60) , " minuti con ", self.num_layer, " layer con C1 = ", self.min, "\n sono le ", datetime.now().time())
        #print("Il minimo ottenuto corrisponde a: ", self.min, " il tempo di esecuzione corrisponde a: ",int((time.time() - self.start_time)/60), " minuti")
        #Set num max di ripe o delta di errore, mettere che se trova C1 = 0 si ferma
        #result = minimize(self.testCirc, result.x, args=(vector,), method="cobyla") 
        # Puoi riconvertire result.x alla forma originale, se necessario
        optimized_params = result.x.reshape(self.a.shape)
        self.a = optimized_params
        return result, optimized_params
    
    def c1(self,params,batchStates):
        if self.toFind is True:
            parametri = params.reshape(self.a.shape)
            purity = self.load_purity(parametri,batchStates,1)
            dip = self.res.dip_parallel(parametri,batchStates)
            c1 = purity - dip
            if self.min > c1:
                print("Parametri ottimi trovati: ", parametri)
                print("C: ", purity, " - ", dip, " = ", c1,  "\nIl tempo di esecuzione corrisponde a: ",int((time.time() - self.timeRun)/60), " minuti, sono le ", datetime.now().time(), "\nil miglioramento corrisponde a: ", self.min - c1)
                self.min = c1
                
                self.a = parametri#DA PROVARE SE VA
            epsilon = 10e-2
            timeRunning = 60*1
            #epsilon = 0.5
            """or (time.time() - self.timeRun) > 60*timeRunning"""
            if c1 < epsilon:
                self.toFind = False or (time.time() - self.timeRun) > 60*timeRunning
                print("C1 = ",c1)
                #self.c1Array1Layer.append(purity-dip)
                optimized_params = params.reshape(self.a.shape)
                file_path = "output.txt"  # Sostituisci con il percorso del file che desideri
                #print("Parametri ottimi trovati: ", optimized_params)
                #exit()
                x = Dip_Pdip(optimized_params,batchStates[0],self.num_layer)
                self.unitaria = self.getUnitaria(x.unitaria2)
                self.work()
                return 0
            return purity - dip + np.random.normal(0, 0.1)
        else:
            return 1
        
    def printCircuit(self, circuit):
        current_dir = os.path.dirname(os.path.realpath(__file__))        
        # Salva il circuito come immagine
        image_path = os.path.join(current_dir, 'PrepStatePassato.png')
        circuit_drawer(circuit, output='mpl', filename=image_path)
        
        # Apri automaticamente l'immagine
        img = Image.open(image_path)
        img.show()
    
    def getQc(self,qs):#Il circuito
        quantum_state = Statevector(qs)
        qc = QuantumCircuit(quantum_state.num_qubits)
        qc.initialize(quantum_state, range(quantum_state.num_qubits))
        return qc
    
    def getRhoMath(self, batch):
        val = 0.0
        for a in batch:
            val += np.outer(a, a.conj())
        print("Rho calcolato:\n", val/len(batch))
        return val/len(batch)

    def load_purity(self,params,batch,nrep):
        ov = 0.0
        for ii in range(len(batch)):
            circuit = Dip_Pdip(params,batch[ii],1)
            ov += circuit.obj_ds(circuit.getFinalCircuitDS())    
            #print("OV: ",ov)     
        f = ov/(len(batch) * nrep)
        return (2*f)-1
    
    def work(self):
        L = self.unitaria.data
        rho = self.rho
        matrice = np.where(rho < 0.1, 0, rho)
        #print("Rho approssimato:")
        #print(matrice)
        #print("Matrice unitaria:", unitary)
        
        #print("Rho da input: ",rho)
        U = L.T.conj()
        U_conj_T = L
        D = np.matmul(np.matmul(U_conj_T,rho),U)
        rho = np.matmul(np.matmul(U,D),U_conj_T)
        np.set_printoptions(precision=2)
        matrice = np.where(D < 0.1, 0, D)
        file_path = "output.txt"  # Sostituisci con il percorso del file che desideri

        with open(file_path, 'w') as f:
            # Scrivi l'array D
            f.write("D a precisione 2:\n")
            np.savetxt(f, D, fmt='%.2f')  # Formato a 2 decimali

            # Scrivi l'array L
            f.write("\nL:\n")
            np.savetxt(f, L, fmt='%.2f')  # Formato a 2 decimali

            f.write("\nD approssimato:\n")
            np.savetxt(f, matrice, fmt='%.2f')  # Formato a 2 decimali
        #print("D e' diaognale?", self.is_diagonale(D))
        #print("GLi 1 si trovano")
        #print(self.trova_posizioni_uno(D))
        
        #print("------------------------------------------------------------------")
        risultati = []
        for i in range(0,D[0].size):
            lambda_ = L[i,:]
            if D[i][i] != 0:
                lambda_ = L[i,:]
                self.add_value(D[i][i],lambda_)
                
        for i in (self.get_nonzero_indices(D)):
            lambda_ = L[i,:]
            print("Autovalore: ", D[i][i], "\nAutovettore: ", lambda_)
            print("Autovettore convertito: ", self.conversione(lambda_))

        """array_finale = np.array(risultati)
        print("Array ottenuto nuovamente:")
        print(array_finale)
        flat_array = array_finale.flatten()"""

    def is_diagonale(self,matrice):
    # Confronta la matrice con la sua versione diagonale estratta usando np.diag
        return np.all(matrice == np.diag(np.diag(matrice)))

        

    def density_matrix(self, rho, params, num_layers, n_rep):
        #print("parametri trovati matematicamente:" ,params)
        circ = Dip_Pdip(params,rho,num_layers)
        a = circ.layer
        a.save_density_matrix()
        simulator = Aer.get_backend('aer_simulator')
        transpiled_circuit = transpile(a, simulator)
        result = simulator.run(transpiled_circuit, shots=1000).result()
        # Ottieni la matrice densità dallo stato finale
        self.matrice_dens = result.data(0)['density_matrix']
        print("Numero Layer: ", self.num_layer)
        #print(self.matrice_dens.data)
        print()
        print("Matrice densità: ")
        print(self.matrice_dens)
        
        #print(np.round(self.matrice_dens.data, 2)

    def add_value(self,D, autovettore):
        self.DLambda.append((D, autovettore))
        self.DLambda.sort(key=lambda x: x[0], reverse=True) 

    def conversione(self,eigenvector):
        norm = np.linalg.norm(eigenvector)
        normalized_vector = eigenvector / norm

        # Pixelizzazione
        pixelized_vector = np.where(np.abs(normalized_vector) > 0.1, 1, 0)

        return pixelized_vector
    
    def get_nonzero_indices(self, D):
        """
        Restituisce gli indici dei valori non nulli nella matrice diagonale D,
        considerando una soglia per valori estremamente piccoli.
        """
        # Ottieni i valori dalla diagonale
        diagonal_values = np.diag(D)
        #diagonal_values = np.sort(diagonal_values)[::-1]
        # Trova gli indici dei valori non nulli
        indici = [i for i, elem in enumerate(diagonal_values) if 0.1 <= elem < 1 and not (0 < elem < 0.1)]
        print("Indici: ", indici)
        return indici

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()  # Solo per sistemi Windows
    start_time = time.time()
    main = Main(start_time)
    end_time = time.time()

    print(f"Tempo di esecuzione: {end_time - start_time:.2f} secondi")

        
