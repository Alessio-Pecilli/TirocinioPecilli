import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import qiskit.quantum_info as qi
from src.dip_pdip import Dip_Pdip
from qiskit.quantum_info import Statevector
import os
from qiskit.circuit import Parameter
from qiskit import transpile
from qiskit.visualization import circuit_drawer
from qiskit_aer import Aer
from qiskit import QuantumCircuit
from scipy.optimize import minimize
from src.result import Result
import matplotlib.pyplot as plt
import time

class Main:

    def __init__(self,start_time):
        self.start_time = start_time
        self.min = 1
        self.res = Result()
        self.c1Array1Layer = []
        self.c1Array2Layer = []
        self.c1Array3Layer = []
        self.RhoArray = []
        self.DArray = []
        self.LArray = []
        self.DLambda = []
        self.totLayer = 1
        #Se bloccato per troppo aumento N Layer
        #Implementare ansatz per pari
        #Plot cost -> Iterazioni(Confronto anche tra Lyaer)
        #Max 1 ora di run
        #PER TESI
        #Simulatori quantistici + verifiche sperimentali(simulazione materiali complessi)
        self.testTwoStates()

    def testTwoStates(self):
        a = np.array([1/np.sqrt(2), 0 ,0,1/np.sqrt(2)])
        b = np.array([0,1/np.sqrt(2), 1/np.sqrt(2),0])
        c = np.array([1/np.sqrt(2),0, 1/np.sqrt(2) ,0])
        d = np.array([1/np.sqrt(3), 0,1/np.sqrt(3) ,1/np.sqrt(3)])
        e = np.array([1/np.sqrt(2), 1/np.sqrt(2) ,0,0])
        print("TURNO 1")
        self.imgIniziale = a
        batchBinary = [a]
        self.rho = self.getRhoMath(batchBinary)
        self.num_qubits = 2
        self.res.num_qubits = 2
        self.toFind = True
        self.num_layer = 1
        self.res.num_layers = self.num_layer
        self.timeRun = time.time()
            #print("Ora lavoro con n_layer: ", self.num_layer)
        self.a = self.paramsStandard()
        self.counter = 1
        while(self.toFind is True):
            self.c1_optimization(batchBinary)
            self.counter+=1
            self.c1Array1Layer.append(self.min)
        c1Array1Layer = np.array(self.c1Array1Layer)  # Converte in array NumPy
        print("Con 2 Layer: ", c1Array1Layer)
        
        print("TURNO 2")
        self.min = 1
        self.timeRun = time.time()
        countTOT = self.counter-1
        self.toFind = True
        self.num_layer+=1  
        self.res.num_layers = self.num_layer
            #print("Ora lavoro con n_layer: ", self.num_layer)
        self.a = self.paramsStandard()
        self.counter = 0
        print("counter: ", self.counter, " tot: ", countTOT)
        while(self.counter != countTOT):
            self.c1_optimization(batchBinary)
            self.counter+=1
            self.c1Array2Layer.append(self.min)
        c2Array1Layer = np.array(self.c1Array2Layer)  # Converte in array NumPy
        print("Con 2 Layer: ", c2Array1Layer)
        print("TURNO 3")
        self.min = 1
        self.timeRun = time.time()
        self.toFind = True
        self.num_layer+=1  
        self.res.num_layers = self.num_layer
            #print("Ora lavoro con n_layer: ", self.num_layer)
        self.a = self.paramsStandard()
        self.counter = 0
        print("counter: ", self.counter, " tot: ", countTOT)
        print("---------------------------------")
        while(self.counter != countTOT):
            self.c1_optimization(batchBinary)
            self.counter+=1
            self.c1Array3Layer.append(self.min)
        self.plot()
        

    def testOneStase(self):
        a = np.array([1/np.sqrt(3), 1/np.sqrt(3) ,0,1/np.sqrt(3)])
        batchBinary = [a]
        self.imgIniziale = a
        self.rho = self.getRhoMath(batchBinary)
        self.num_qubits = 2
        self.res.num_qubits = 2
        for self.num_layer in range(1,self.totLayer + 1):
            self.toFind = True
            self.res.num_layers = self.num_layer
            print("Ora lavoro con n_layer: ", self.num_layer)
            self.a = self.paramsStandard()
            while(self.toFind is True):
                self.c1_optimization(batchBinary)


    def MNISTTest(self):
        batchStates, batchBinary = self.res.load_img(1)
        
        a = np.array([
    [0, 1, 1, 0],
    [1, 0, 0, 1],
    [1, 0, 0, 1],
    [0, 1, 1, 0]
])
        norm_factor = np.sqrt(np.sum(a**2))

        # Normalizza l'array
        normalized_array1 = a / norm_factor
        b = np.array([
    [1, 1, 1, 0],
    [0, 0, 1, 0],
    [0, 1, 0, 0],
    [0, 1, 0, 0]
])
        norm_factor = np.sqrt(np.sum(b**2))
        # Normalizza l'array
        normalized_array2 = b / norm_factor
        #batchBinary = [normalized_array1,normalized_array2]
        self.rho = self.getRhoMath(batchBinary)
        #y = batchBinary[0].shape[0] * batchBinary[0].shape[0]
        self.num_qubits = int(np.log2(batchBinary[0].shape[0]))
        #self.num_qubits = int(np.log2(y))
        self.res.num_qubits = self.num_qubits
        print("TURNO 1")
        self.num_layer = 1
        self.toFind = True
        self.res.num_layers = self.num_layer
        self.a = self.paramsStandard()
        self.timeRun = time.time()
            #print("Ora lavoro con n_layer: ", self.num_layer)
        self.counter = 1
        while(self.toFind is True):
            self.c1_optimization(batchBinary)
            self.counter+=1
            self.c1Array1Layer.append(self.min)
        c1Array1Layer = np.array(self.c1Array1Layer)  # Converte in array NumPy
        print("Con 1 Layer: ", c1Array1Layer)
        print("TURNO 2")
        self.min = 1
        self.timeRun = time.time()
        countTOT = self.counter-1
        self.toFind = True
        self.num_layer+=1  
        self.res.num_layers = self.num_layer
            #print("Ora lavoro con n_layer: ", self.num_layer)
        self.a = self.paramsStandard()
        self.counter = 0
        print("counter: ", self.counter, " tot: ", countTOT)
        while(self.counter != countTOT):
            self.c1_optimization(batchBinary)
            self.counter+=1
            self.c1Array2Layer.append(self.min)
        c1Array2Layer = np.array(self.c1Array2Layer)  # Converte in array NumPy
        print("Con 2 Layer: ", c1Array2Layer)
        
        print("TURNO 3")
        self.min = 1
        self.timeRun = time.time()
        self.toFind = True
        self.num_layer+=1  
        self.res.num_layers = self.num_layer
            #print("Ora lavoro con n_layer: ", self.num_layer)
        self.a = self.paramsStandard()
        self.counter = 0
        print("counter: ", self.counter, " tot: ", countTOT)
        while(self.counter != countTOT):
            self.c1_optimization(batchBinary)
            self.counter+=1
            self.c1Array3Layer.append(self.min)
        self.plot()

    def pca_from_T(self,T):
        # 1. Calcolo autovalori e autovettori della matrice T
        eigenvalues, U_T = np.linalg.eigh(T)
        
        # 2. Ordina autovalori e autovettori in ordine decrescente
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        U_T = U_T[:, idx]
        
        # 3. Costruisci la matrice diagonale Lambda_T
        Lambda_T = np.diag(eigenvalues)
        
        print('Autovettori: ')
        print(U_T)
        print('Autovalori: ', eigenvalues)
        print('Lambda_T:')
        print(Lambda_T )

    def paramsStandard(self):
        if self.num_qubits == 6 and self.num_layer == 1:#Se uso lo 0 da MNSIT
            #per 0 e 7
            data_list = [4.8271, -0.2266, -0.2307, 2.5973, -3.1726, 0.4472, 2.062, 2.7945, -1.6858, 1.7472, -2.0093, 2.4955, 2.767, -2.5138, 0.6689, -0.9537, -0.0695, 0.3661, 0.6099, -1.7153, -1.7218, -0.0639, 3.0658, 0.5961, 3.0716, 1.3141, 2.055, 2.5826, 1.829, 1.4141, -0.5007, 2.454, 2.9092, -2.7698, -1.5272, 2.6841, 0.3352, -1.0254, 0.553, 0.1876, 1.7407, 1.9197, -0.7875, 2.5172, -0.2287, 0.5562, 2.9556, 1.2559, 2.5721, -2.413, 2.4267, -0.8998, -0.3189, -2.9932, 2.9629, -2.9214, -2.1814, 2.2012, 2.1024, -2.7828, 2.5091, -1.021, -0.5078, -0.8287, -0.9777, 2.5078, 0.3045, -2.1724, -0.2387, 0.8422, 1.3412, -1.2194]
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

        result = minimize(self.c1, flat_params, args=(k,), method="cobyla") 
        #print("terminato run numero: ", self.counter, " con tempo passato: ", int((time.time() - self.timeRun)/60) , " minuti con valore minimo trovato: ", self.min)
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
            dip = self.res.dip(parametri,batchStates)
            purity = self.load_purity(parametri,batchStates,100)

            c1 = purity - dip
            
            c1 = purity - dip
            if self.min > c1:
                #print("Parametri ottimi trovati: ", parametri)
                self.min = c1
                #print("C: ", purity, " - ", dip, " = ", c1,  "\nIl tempo di esecuzione corrisponde a: ",int((time.time() - self.timeRun)/60), " minuti")
                self.a = parametri#DA PROVARE SE VA
            epsilon = 1.5e-1
            timeRunning = {
                1: 60,
                2: 120,
                3: 300
            }.get(self.num_layer, 0)
            #epsilon = 0.5
            if c1 < epsilon or (time.time() - self.timeRun) > 60*timeRunning:
                self.toFind = False
                #self.c1Array1Layer.append(purity-dip)
                optimized_params = params.reshape(self.a.shape)
                file_path = "output.txt"  # Sostituisci con il percorso del file che desideri
                #print("Parametri ottimi trovati: ", optimized_params)
                #exit()
                x = Dip_Pdip(optimized_params,batchStates[0],self.num_layer)
                self.unitaria = self.getUnitaria(x.unitaria2)
                self.work()
                return 0
            return purity - dip
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
    
    def load_purity(self,params,batch,nrep):
        ov = 0.0
        for ii in range(len(batch)):
            circuit = Dip_Pdip(params,batch[ii],1)
            ov += circuit.obj_ds(circuit.getFinalCircuitDS())    
            #print("OV: ",ov)     
        f = ov/(len(batch) * nrep)
        return (2*f)-1
    
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
    
    def work(self):
        L = self.unitaria.data
        rho = self.rho
        matrice = np.where(rho < 0.1, 0, rho)
        #print("Rho approssimato:")
        #print(matrice)
        #print("Matrice unitaria:", unitary.shape)
        
        #print("Rho da input: ",rho)
        U = L.T.conj()
        U_conj_T = L
        D = np.matmul(np.matmul(U_conj_T,rho),U)
        rho = np.matmul(np.matmul(U,D),U_conj_T)
        #print("Rho calcolato a partire da U e D: ", rho)
        self.RhoArray.append(rho)
        self.DArray.append(D)
        self.DArray.append(L)
        #print("D a precisione 2:")
        np.set_printoptions(precision=2)
        #print(D)
        #print("L:")
        #print(L)
        #print(L)
        # Applica la condizione: se l'elemento è < 0.9, diventa 0; altrimenti 1
        matrice = np.where(D < 0.1, 0, D)
        #print("D approssimato:")
        #print(matrice)
        #print("------------------")
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
        pixelized_vector = np.where(np.abs(normalized_vector) > 0.25, 1, 0)

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


start_time = time.time()
main = Main(start_time)
end_time = time.time()

print(f"Tempo di esecuzione: {end_time - start_time:.2f} secondi")
        
