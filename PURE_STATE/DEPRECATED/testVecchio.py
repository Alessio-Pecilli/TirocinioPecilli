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

class Test:

    def __init__(self):
        self.res = Result(1)
        a = np.array([1/np.sqrt(2), 0 ,0,1/np.sqrt(2)])
        b = np.array([0,1/np.sqrt(2), 1/np.sqrt(2),0])
        layerTot = 2
        batch = [a,b]
        # Calcolo degli autovalori e degli autovettori
        self.rho = self.getRhoMath(batch)

        self.c1Array = []
        self.DMaggioreArray = []
        self.OverlapArray = []
        self.num_layer = 1
        
        for self.num_layer in range(1,layerTot + 1):
            self.toFind = True
            print("Ora lavoro con: ", self.num_layer)
            self.a = self.res.get_params(2, self.num_layer)
            while(self.toFind is True):
                self.c1_optimization(a)
        self.plot()
        

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
            purity = self.res.load_purity(params, b,100)
            #print("C1: ", purity, " - ", dip, " = ", purity - dip)
            if purity - dip == 0:
                self.toFind = False
                print("SONO ENTRATO NEL GIUSTO BLOCCO")
                self.c1Array.append(purity-dip)
                
                optimized_params = parametri.reshape(self.a.shape)
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
    
    def c1_optimization(self, vector):
        flat_params = np.ravel(self.a)
        result = minimize(self.testCirc, flat_params, args=(vector,), method="cobyla")   
        optimized_params = result.x.reshape(self.a.shape)
        return result, optimized_params

    
    def c1(self,params,batchStates):
        parametri = params.reshape(self.a.shape)
        dip = self.dip(parametri,batchStates)
        purity = self.load_purity(parametri,batchStates,100)
        c1 = purity - dip
        #print("C1: ", purity, " - ", dip, " = ", c1)
        if c1 == 0:
            optimized_params = params.reshape(self.a.shape)
            x = Dip_Pdip(optimized_params,batchStates[0],self.num_layer)
            self.unitaria = self.getUnitaria(x.unitaria2)
            self.work()
            exit()
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
        countRep = 0
        nrep = 1000
        for ii in range(len(batchStates)):
            circuit = Dip_Pdip(params,batchStates[ii],1)
            circ = circuit.getFinalCircuitDIP()
            #self.printCircuit(circ)
            #circuit.compute_purity(1)
            countRep += nrep
            count = circuit.obj_dip_counts(circ,nrep)
            for state, value in count.items():
                if state in counts:
                    counts[state] += value  # Se lo stato esiste, somma i valori
                else:
                    counts[state] = value  # Se lo stato non esiste, crea una nuova chiave
   
        return self.overlap_from_count(counts,countRep)
    
    def getQc(self,qs):#Il circuito
        quantum_state = Statevector(qs)
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
        val = 0.0
        for a in batch:
            val += np.outer(a, a.conj())
        return val/len(batch)    
    
    def work(self):
        L = self.unitaria.data
        rho = self.rho
        #print("Matrice unitaria:", unitary.shape)
        #print("Rho: ", rho)
        print()
        U = L.T.conj()
        U_conj_T = L
        D = np.matmul(np.matmul(U_conj_T,rho),U)
        rho = np.matmul(np.matmul(U,D),U_conj_T)
        self.DMaggioreArray.append(np.max(np.diag(D)))
        print("D:",D)
        print()
        print("L:")
        print(np.round(L, 2))
        print("------------------------------------------------------------------")
        #print("L_dagger:")
        #print(L_conj_T)

        #print("Rho :",rho_calcolato)
        
        #Approssimo D
        #DApprossimata = np.where(D < 0.09, 0, 1)
        print()
        #print("D approssimata: ",DApprossimata)
        #autovalori, autovettori = np.linalg.eig(D)
        print()
        
        #print("Autovalori D:", autovalori)
        

    def density_matrix(self, rho, params, num_layers, n_rep):
        #print("parametri trovati matematicamente:" ,params)
        circ = Dip_Pdip(params,rho,num_layers)
        a = circ.layer
        #self.res.printCircuit(a)
        #print("PARAMETRI TROVATI PER CUI C1 = 0 NEL CIRCUITO" , a.parameters)
        #self.res.printCircuit(a)
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
        print(np.round(self.matrice_dens.data, 2))
        


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

    def plot(self):
        layers = np.arange(1, 3)  # Numero di layer da 1 a 5
        # Creazione del plot
        plt.figure(figsize=(10, 6))
        # Plot per C1 finale
        plt.plot(layers, self.c1Array, marker='o', linestyle='-', color='b', label='C1 finale')
        plt.plot(layers, self.DMaggioreArray, marker='o', linestyle='-', color='r', label='C1 finale')
        # Dettagli del grafico
        plt.xlabel('Numero di Layer')
        plt.ylabel('Valore')
        plt.title('Quantità in funzione del numero di layer')
        plt.grid(True)
        plt.legend()

        # Mostra il grafico
        plt.show()
        
test = Test()