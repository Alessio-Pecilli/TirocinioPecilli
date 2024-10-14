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

class Test:

    def __init__(self):
        self.res = Result(1)
        a = np.array([1/np.sqrt(2), 0 ,0,1/np.sqrt(2)])
        b = np.array([0,1/np.sqrt(2), 1/np.sqrt(2),0])

# Calcolo del prodotto esterno |psi><psi| (prodotto tra vettore colonna e vettore riga)
        stato1 = np.outer(a, a.conj())  # Usare np.outer per il prodotto esterno
        stato2 = np.outer(b, b.conj())
        print("Stato 1", stato1)
        density_matrix = (stato1+stato2)/2
        print("Vector:", density_matrix )
        self.num_layer = 1
        # Calcolo degli autovalori e degli autovettori
        #rho = np.outer(vector, vector)
        #autovalori, autovettori = np.linalg.eigh(rho)
        
        #idx = np.argsort(autovalori)[::-1]  # Ordine decrescente
        #autovalori_ordinati = autovalori[idx]
        #autovettori_ordinati = autovettori[:, idx]
        #self.h = self.testDipPurity(vector)
        #print("Test per: ", vector)
        #print("Autovalori ordinati:", autovalori_ordinati)
        #print("Autovettori ordinati:\n", autovettori_ordinati)
        #print("RUN--------------------------------------------")
        
        batch = [stato1,stato2]
        self.a = self.res.get_params(4,self.num_layer)
        #print("Ottengo come parametri casuali:", self.a)
        #print("Prima dell'ottimo ---------------------------------")           
        vectorized_matrix = density_matrix.flatten()
        #self.density_matrix(self.getRho(vectorized_matrix),self.a,self.num_layer,1000)
           
        #self.density_matrix(self.getRho(vectorized_matrix),self.a,2,2)
        x, paramStar = self.c1_optimizationNUOVO(batch)
        print("Test dopo che ho ottenuto l'ottimo --------------------------------------")
        #self.testCirc(paramStar,vector)  
        print("Matrice densita' con i parametri ottimizzati-----")
        
        #self.density_matrix(self.getRho(vectorized_matrix),paramStar,self.num_layer,1000)

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
        print("Lui usa:", vectorized_matrix)
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
        #Set num max di ripe o delta di errore, mettere che se trova C1 = 0 si ferma
        #result = minimize(self.testCirc, result.x, args=(vector,), method="cobyla") 
        # Puoi riconvertire result.x alla forma originale, se necessario
        optimized_params = result.x.reshape(self.a.shape)
        
        return result, optimized_params
    
    def c1_optimizationNUOVO(self, batch):
        # Appiattisci i parametri solo per la fase di ottimizzazione
        flat_params = np.ravel(self.a)
        # Passa i parametri appiattiti a minimize con i vincoli
        """Devo rendere stati i miei batch"""
        k = []  # Inizializzi come lista, non come dizionario
        for i in range(len(batch)):
            print(batch[i])
            a = batch[i].flatten()
            k.append(Statevector(a))  # Usa append sulla lista


        result = minimize(self.c1, flat_params, args=(k,), method="cobyla")    
        #Set num max di ripe o delta di errore, mettere che se trova C1 = 0 si ferma
        #result = minimize(self.testCirc, result.x, args=(vector,), method="cobyla") 
        # Puoi riconvertire result.x alla forma originale, se necessario
        optimized_params = result.x.reshape(self.a.shape)
        
        return result, optimized_params
    
    def c1(self,params,batchStates):
        parametri = params.reshape(self.a.shape)
        dip = self.dip(parametri,batchStates)
        purity = self.load_purity(parametri,batchStates,100)
        c1 = purity - dip
        #print("C1: ", purity, " - ", dip, " = ", c1)
        return c1
    
    def load_purity(self,params,batch,nrep):
        ov = 0.0
        for ii in range(len(batch)):
            circuit = Dip_Pdip(params,batch[ii],1)
            ov += circuit.obj_ds(circuit.getFinalCircuitDS())    
            #print("OV: ",ov)     
        f = ov/(len(batch) * nrep)
        return (2*f)-1
    
    def dip(self,params,batchStates):
        
        counts = {}
        nrep = 1000
        for ii in range(len(batchStates)):
            circuit = Dip_Pdip(params,batchStates[ii],1)
            circ = circuit.getFinalCircuitDIP()
            #self.printCircuit(circ)
            #circuit.compute_purity(1)
            
            count = circuit.obj_dip_counts(circ,nrep)
            for state, value in count.items():
                if state in counts:
                    counts[state] += value  # Se lo stato esiste, somma i valori
                else:
                    counts[state] = value  # Se lo stato non esiste, crea una nuova chiave
   
        return self.overlap_from_count(counts,nrep)
    
    def getRho(self,rho):#Il circuito
        
        quantum_state = Statevector(rho)
        qc = QuantumCircuit(quantum_state.num_qubits)
        qc.initialize(quantum_state, range(quantum_state.num_qubits))
        return qc
    
    def overlap_from_count(self, counts, repetitions):
        #print("Risultati per il dip: ", counts)
        zero_state = '0' * 2
        zero_state_count = 0
        
        # Se counts non è una lista, lo mettiamo in una lista
        if not isinstance(counts, list):
            counts = [counts]
        
        for i, count_item in enumerate(counts):
            #print(f"Esaminando elemento {i + 1}: {count_item}")
            
            if isinstance(count_item, dict):
                # Se è già un dizionario, lo usiamo direttamente
                count_dict = count_item
            elif isinstance(count_item, str):
                # Se è una stringa, proviamo a convertirla in un dizionario
                try:
                    count_dict = eval(count_item)
                except:
                    print(f"Impossibile convertire la stringa in dizionario: {count_item}")
                    continue
            elif isinstance(count_item, int):
                # Se è un intero, lo trattiamo come il conteggio diretto dello zero_state
                zero_state_count += count_item
                continue
            else:
                print(f"Tipo di dato non supportato: {type(count_item)}")
                continue
            
            if isinstance(count_dict, dict) and zero_state in count_dict:
                #print(f"Trovato {zero_state} con valore: {count_dict[zero_state]}")
                zero_state_count += count_dict[zero_state]

        #print(f"Numero totale di occorrenze di {zero_state}: {zero_state_count}")
        #print("Ho calcolato 0 * n_quibit un numero di volte = ", zero_state_count)
        return zero_state_count / repetitions
    
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
        print("parametri trovati matematicamente:" ,params)
        circ = Dip_Pdip(params,rho,num_layers)
        a = circ.layer
        self.res.printCircuit(a)
        print("PARAMETRI TROVATI PER CUI C1 = 0 NEL CIRCUITO" , a.parameters)
        #self.res.printCircuit(a)
        a.save_density_matrix()
        simulator = Aer.get_backend('aer_simulator')
        transpiled_circuit = transpile(a, simulator)
        result = simulator.run(transpiled_circuit, shots=1000).result()
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