import pandas as pd
import cirq
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import qiskit
import qiskit.quantum_info as qi
from dip_pdip import Dip_Pdip
from prepState import StatePreparation
import readMINST
from qiskit.circuit import ParameterVector
import random
from qutip import Qobj, ket2dm
from qiskit.quantum_info import Statevector
import os
from qiskit.circuit import Parameter
import math
from qiskit import QuantumRegister, transpile
from qiskit.visualization import circuit_drawer
import qiskit_aer
from qiskit_aer import Aer
from qiskit_aer import AerSimulator
from qiskit import QuantumCircuit
from qiskit.visualization import plot_histogram
from scipy.optimize import minimize
from result import Result
import matplotlib.pyplot as plt
import time

class Main:

    def __init__(self):
        self.min = 1
        self.res = Result(1)
        self.c1Array = []
        self.RhoArray = []
        self.DArray = []
        self.LArray = []
        self.DLambda = []
        self.totLayer = 1
        
        self.MNISTTest()

    def testTwoStates(self):
        a = np.array([1/np.sqrt(2), 0 ,0,1/np.sqrt(2)])
        b = np.array([0,1/np.sqrt(2), 1/np.sqrt(2),0])
        batchBinary = [a,b]
        self.rho = self.getRhoMath(batchBinary)
        self.num_qubits = 2
        for self.num_layer in range(1,self.totLayer + 1):
            self.toFind = True
            print("Ora lavoro con n_layer: ", self.num_layer)
            self.a = self.res.get_params(self.num_qubits, self.num_layer)
            while(self.toFind is True):
                self.c1_optimization(batchBinary)

    def testOneStase(self):
        a = np.array([1, 0])
        batchBinary = [a]
        self.rho = self.getRhoMath(batchBinary)
        self.num_qubits = 1
        for self.num_layer in range(1,self.totLayer + 1):
            self.toFind = True
            print("Ora lavoro con n_layer: ", self.num_layer)
            self.a = self.res.get_params(self.num_qubits, self.num_layer)
            while(self.toFind is True):
                self.c1_optimization(batchBinary)


    def MNISTTest(self):
        batchStates, batchBinary = self.res.load_img(2)
        print("-----------------------------")
        print(batchBinary[0])
        print(batchBinary[1])
        self.rho = self.getRhoMath(batchBinary)
        self.num_qubits = int(np.log2(batchBinary[1].shape[0]))
        for self.num_layer in range(1,self.totLayer + 1):
            self.toFind = True
            print("Ora lavoro con n_layer: ", self.num_layer)
            self.a = self.res.get_params(self.num_qubits, self.num_layer)
            while(self.toFind is True):
                self.c1_optimization(batchBinary)
        return

    def plot(self):
        layers = np.arange(1, self.totLayer + 1)  # Numero di layer da 1 a 5
        # Creazione del plot
        plt.figure(figsize=(10, 6))
        # Plot per C1 finale
       
        c1Array = np.array(self.c1Array)  # Converte in array NumPy

        # Ora puoi usare .real
        plt.plot(layers, c1Array.real, marker='o', linestyle='-', color='b', label='C1 finale')
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

        # Crea uno stato quantistico Statevector dal vettore
    def testCirc(self, parametri, vector):
        if self.toFind is True:
            params = parametri.reshape(self.a.shape)
            vectorized_matrix = vector.flatten()
            quantum_state = Statevector(vectorized_matrix)
            qc = QuantumCircuit(quantum_state.num_qubits)
            qc.initialize(quantum_state, range(quantum_state.num_qubits))
            
            b = [qc]
            dip = self.res.dip(params, b)
            qc = QuantumCircuit(quantum_state.num_qubits)
            qc.initialize(quantum_state, range(quantum_state.num_qubits))
            b = [qc]
            purity = self.load_purity(params, b,100)
            #print("C1: ", purity, " - ", dip, " = ", purity - dip)
            if(purity - dip) < 0.5:
                print("Arriva sotto 0.5")
            if(purity - dip) < 0.5:
                print("Arriva sotto 0.3")
            if purity - dip == 0:
                self.toFind = False
                #print("SONO ENTRATO NEL GIUSTO BLOCCO")
                self.c1Array.append(purity-dip)     
                optimized_params = parametri.reshape(self.a.shape)
                print("Parametri ottimi trovati: ", optimized_params)
                x = Dip_Pdip(optimized_params,b[0],self.num_layer)
                self.unitaria = self.getUnitaria(x.unitaria2)
                #self.density_matrix(self.getQc(vectorized_matrix),optimized_params,self.num_layer,1000)
                self.work()
                return 0
            return purity - dip
        else:
            return 1

    def getUnitaria(self,qc):
        #self.res.printCircuit(qc)
        return qi.Operator(qc)
    
    def c1_optimizationOLD(self, vector):
        # Appiattisci i parametri solo per la fase di ottimizzazione
        flat_params = np.ravel(self.a)
        # Passa i parametri appiattiti a minimize con i vincoli
        result = minimize(self.testCirc, flat_params, args=(vector,), method="cobyla")    
        #Set num max di ripe o delta di errore, mettere che se trova C1 = 0 si ferma
        #result = minimize(self.testCirc, result.x, args=(vector,), method="cobyla") 
        # Puoi riconvertire result.x alla forma originale, se necessario
        optimized_params = result.x.reshape(self.a.shape)
        
        return result, optimized_params
    
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
        print("Il minimo ottenuto corrisponde a: ", self.min)
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
            if self.min > c1:
                self.min = c1
            #print("C1: ", purity, " - ", dip, " = ", c1)
            epsilon = 1e-2
            if c1 < epsilon:
                self.toFind = False
                self.c1Array.append(purity-dip)
                optimized_params = params.reshape(self.a.shape)
                print("Parametri ottimi trovati: ", optimized_params)
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
        #print("Matrice unitaria:", unitary.shape)
        
        #print("Rho da input: ",rho)
        U = L.T.conj()
        U_conj_T = L
        D = np.matmul(np.matmul(U_conj_T,rho),U)
        rho = np.matmul(np.matmul(U,D),U_conj_T)
        print("Rho calcolato a partire da U e D: ", rho)
        self.RhoArray.append(rho)
        self.DArray.append(D)
        self.DArray.append(L)
        print("D a precisione 5:")
        np.set_printoptions(precision=2)
        print(D)
        print("L:")
        #print(L)
        print(L)
        # Applica la condizione: se l'elemento è < 0.9, diventa 0; altrimenti 1
        matrice = np.where(D < 0.1, 0, D)
        print("D approssimato:")
        print(matrice)
        print("------------------")
        #print("D e' diaognale?", self.is_diagonale(D))
        #print("GLi 1 si trovano")
        #print(self.trova_posizioni_uno(D))
        
        #print("------------------------------------------------------------------")
        risultati = []
        for i in range (0,D[0].size):
            lambda_ = L[i,:]
            if D[i][i] != 0:
                lambda_ = L[i,:]
                self.add_value(D[i][i],lambda_)
            risultati.append(self.conversione(lambda_))

        array_finale = np.array(risultati)
        print("Array ottenuto nuovamente:")
        print(array_finale)

        """
        print(self.convert_eigenvectors_to_images(D))
        nonzero_indices = self.get_nonzero_indices(D)
        i = np.argmax(np.diag(D))
        lambda_ = L[i,:]
        self.conversione(lambda_)
        lambda_conj = np.conjugate(lambda_)
        #print("Lambda trovato: ", lambda_, " lambda coniugato: ", lambda_conj, " psi vale: ", self.psi)
        dot_product = np.dot(self.psi, lambda_conj)
        #print(" Il dot product = ", dot_product)
        mod_quadro = np.abs(dot_product)**2

        self.OverlapArray.append(mod_quadro)"""

        #psi è a
    def trova_posizioni_uno(self,matrice):
        # Trova le posizioni degli elementi uguali a 1
        posizioni = np.argwhere(matrice == 1)
        # Converti le posizioni in una lista di tuple
        lista_posizioni = [tuple(posizione) for posizione in posizioni]
        return lista_posizioni

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
        pixelized_vector = np.where(np.abs(normalized_vector) > 0.5, 1, 0)
        return pixelized_vector
    
    def get_nonzero_indices(self, D):
        """
        Restituisce gli indici dei valori non nulli nella matrice diagonale D,
        considerando una soglia per valori estremamente piccoli.
        """
        # Ottieni i valori dalla diagonale
        diagonal_values = np.diag(D)
        
        # Trova gli indici dei valori non nulli
        indici = [i for i, elem in enumerate(diagonal_values) if 0.1 <= elem < 1 and not (0 < elem < 0.1)]
        print("Indici: ", indici)
        return indici


start_time = time.time()
main = Main()
end_time = time.time()

print(f"Tempo di esecuzione: {end_time - start_time:.2f} secondi")
        