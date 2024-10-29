import pandas as pd
import cirq
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import qiskit
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
from sklearn.decomposition import PCA

class Test:

    def __init__(self):
        self.res = Result(1)
        vector = np.array([1/2, 1/2 ,1/2,1/2])
        self.num_layer = 50
        # Calcolo degli autovalori e degli autovettori
        rho = np.outer(vector, vector)
        autovalori, autovettori = np.linalg.eigh(rho)
        
        idx = np.argsort(autovalori)[::-1]  # Ordine decrescente
        autovalori_ordinati = autovalori[idx]
        autovettori_ordinati = autovettori[:, idx]
        vectorized_matrix = vector.flatten()
        self.h = self.testDipPurity(vector)
        #print("Test per: ", vector)
        #print("Autovalori ordinati:", autovalori_ordinati)
        #print("Autovettori ordinati:\n", autovettori_ordinati)
        #print("RUN--------------------------------------------")
        
        
        self.a = self.res.get_params(2,self.num_layer)
        #print("Prima dell'ottimo ---------------------------------")           
        #self.c1_optimization()
           
        #self.density_matrix(self.getRho(vectorized_matrix),self.a,2,2)
        x, paramStar = self.c1_optimization(vector)
        print("Dopo l'ottimo --------------------------------------")
        self.testCirc(paramStar,vector)  
        print("Ottimizzo-----")
        self.density_matrix(self.getRho(vectorized_matrix),paramStar,self.num_layer,1000)

    def testDipPurity(self,vector):        
        vectorized_matrix = vector.flatten()
        print(vectorized_matrix)
        y = self.res.c1_calc(vectorized_matrix)
        print("Risultato matematico C1: ",y)
        return y

        # Crea uno stato quantistico Statevector dal vettore
    def testCirc(self, parametri, vector):
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
        purity = self.res.load_purity(params, b,100)
        print("C1: ", purity, " - ", dip, " = ", purity - dip)
        if(purity - dip == 0):
            #print("SONO ENTRATO NEL GIUSTO BLOCCO")
            optimized_params = parametri.reshape(self.a.shape)
            #print(optimized_params)
            self.density_matrix(self.getRho(vectorized_matrix),optimized_params,self.num_layer,1000)
        return purity - dip

    def c1_optimization(self, vector):
        # Appiattisci i parametri solo per la fase di ottimizzazione
        flat_params = np.ravel(self.a)
        # Passa i parametri appiattiti a minimize con i vincoli
        result = minimize(self.testCirc, flat_params, args=(vector,), method="cobyla")        
        # Puoi riconvertire result.x alla forma originale, se necessario
        optimized_params = result.x.reshape(self.a.shape)
        
        return result, optimized_params
    
    def getRho(self,rho):#Il circuito
        
        quantum_state = Statevector(rho)
        qc = QuantumCircuit(quantum_state.num_qubits)
        qc.initialize(quantum_state, range(quantum_state.num_qubits))
        return qc
    
    def getRhoMath(self,batch):
        #print("Entro in RHO")
        val = 0.0
        for a in batch:
            #print(a)
            val += a
        val/=len(batch)    
        # Appiattisci la matrice in un vettore
        norm_squared = np.sum(np.abs(val) ** 2)
        # Normalizza il vettore per la radice quadrata della norma dei quadrati degli amplitudi
        normalized_params = val / np.sqrt(norm_squared)
        
        vectorized_matrix = normalized_params.flatten()
        # Crea uno stato quantistico Statevector dal vettore
        return vectorized_matrix
    
    def density_matrix(self, rho, params, num_layers, n_rep):
        print("Entro qui")
        #print(params)
        circ = Dip_Pdip(params,rho,num_layers)
        a = circ.layer
        #self.res.printCircuit(a)
        a.save_density_matrix()
        simulator = Aer.get_backend('aer_simulator')
        transpiled_circuit = transpile(a, simulator)
        result = simulator.run(transpiled_circuit, shots=1).result()
        # Ottieni la matrice densità dallo stato finale
        density_matrix = result.data(0)['density_matrix']

        print(density_matrix)
        """
        # Convertire l'oggetto DensityMatrix in un array NumPy
        density_matrix_np = np.asarray(density_matrix)
        # Separare la matrice in parte reale e immaginaria
        real_part = density_matrix_np.real
        imag_part = density_matrix_np.imag

        # Concatenare parte reale e immaginaria
        combined_matrix = np.hstack([real_part, imag_part])

        # Applichiamo la PCA
        pca = PCA()
        pca.fit(combined_matrix)

        # Gli autovettori corrispondenti
        eigenvectors = pca.components_

        # Gli autovalori associati alle componenti principali
        eigenvalues = pca.explained_variance_

        print("Autovettori (componenti principali) trovati con PCA sulla matrice densita':")
        print(eigenvectors)

        print("\nAutovalori (varianza spiegata) trovati con PCA sulla matrice densita':")
        print(eigenvalues)

        # Calcolo degli autovalori e autovettori per matrici complesse
        eigenvalues, eigenvectors = np.linalg.eig(density_matrix)

        print("Autovalori con libreria np usando la matrice densita':")
        print(eigenvalues)

        print("\nAutovettori con libreria np usando la matrice densita':")
        print(eigenvectors)
        vector = np.array([1/np.sqrt(2), 0, 0,1/np.sqrt(2)])
        rho = np.outer(vector, vector)
        autovalori, autovettori = np.linalg.eigh(rho)
        
        idx = np.argsort(autovalori)[::-1]  # Ordine decrescente
        print("Autovalori con libreria np lavorando direttamente col vettore: ", autovalori)
        print("Autovettori con libreria np lavorando direttamente col vettore: ", autovettori)
        """
        

    def dot_product(self,a, b):
        return np.sum(np.conj(a) * b)


        
test = Test()