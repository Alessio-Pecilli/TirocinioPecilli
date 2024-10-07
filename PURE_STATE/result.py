

# =============================================================================
# Imports
# =============================================================================

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

class Result:

    def __init__(self,n):
        
        batchStates, batchBinary  = self.load_img(n)
        self.params = self.get_params()
        #self.work_for_one(self.params)
        #self.c1(batchStates)
        #self.c2(batchStates)
        print(self.c1_optimization(batchStates))
        #print(batchBinary)
        #self.c1(batchStates)
        #self.c1_calc_tot(batchBinary)
        
        return
    
    def evaluete_c1(self,purity,dip):
        return purity - dip
    
    def work_for_one(self,params):#Prova a calcolarmi la purezza per uno stato
        state, state_bin = self.load_img(1)
        circ = Dip_Pdip(params,state[0],1)
        a = circ.getFinalCircuitDS()
        print("Purezza finale ottenuta:", 2*circ.obj_ds(a) -1)
    
    def get_params(self):
        #Per ora uso n_qubit e n_layer fissi
        self._num_qubits = 8
        self.num_layers = 1
        x = self.get_param_resolver(self._num_qubits, self.num_layers)
        params = self.min_to_vqsd(x,self._num_qubits, self.num_layers)
        return params
    
    def load_img(self,n):
        state_prep = StatePreparation(n)
        batchStates, batchBinary = state_prep.LoadVectorMultiplo()
        return batchStates, batchBinary
    
    def c1(self,batchStates):
         
        dip = self.dip(self.params,batchStates)
        purity = self.load_purity(self.params,batchStates)
        c1 = purity - dip
        #print("C1: ", purity, " - ", dip, " = ", c1)
        return c1
    
    def c2(self,batchStates):
         
        pdip = self.pdip(self.params,batchStates)
        purity = self.load_purity(self.params,batchStates)
        c2 = purity - pdip
        print("C2: ", purity, " - ", pdip, " = ", c2)
        return c2
    
    def c1_calc_tot(self, batchBinary):
        for ii in range(len(batchBinary)):
            self.c1_calc(batchBinary[ii])
        return
    
    def c1_calc(self,arr):
        p = self.calc_purity(arr)
        d = self.calc_dip(arr)
        print("Risultati atteso n:  = ",  p - d, " Purezza: ",p, " Dip: ",d)

    def dip(self,params,batchStates):
        
        counts = {}
        nrep = 1
        for ii in range(len(batchStates)):
            circuit = Dip_Pdip(params,batchStates[ii],1)
            circ = circuit.getFinalCircuitDIP()
            #circuit.compute_purity(1)
            
            count = circuit.obj_dip_counts(circ,nrep)
            for state, value in count.items():
                if state in counts:
                    counts[state] += value  # Se lo stato esiste, somma i valori
                else:
                    counts[state] = value  # Se lo stato non esiste, crea una nuova chiave
   
        return self.overlap_from_count(counts,len(batchStates))
    
    def pdip(self,params,batchStates):
        
        ov = 0.0
        for ii in range(len(batchStates)):
            circuit = Dip_Pdip(params,batchStates[ii],1)
            ov += circuit.pdip_test()
        return ov/len(batchStates)
    
    def overlap_from_count(self, counts, repetitions):
        zero_state = '0' * self._num_qubits
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
        return zero_state_count / repetitions
    
    

    def load_purity(self,params,batch):
        ov = 0.0
        for ii in range(len(batch)):
            circuit = Dip_Pdip(params,batch[ii],1)
            ov += circuit.obj_ds(circuit.getFinalCircuitDS())    
            #print("OV: ",ov)     
        return ov/len(batch)
    
    def min_to_vqsd(self, param_list, num_qubits, num_layer):
        # Verifica il numero totale di elementi
        #assert len(param_list) % 6 == 0, "invalid number of parameters"
        
        param_values = np.array(list(param_list.values()))#ho tolto .values per una migliore visualizzazione
        x = param_values.reshape(num_layer,2 ,num_qubits//2 ,12)
        x_reshaped = x.reshape(num_layer, 2, num_qubits // 2, 4, 3)
        #print(x_reshaped)
        return x_reshaped

    def get_param_resolver(self,num_qubits, num_layers):
        """Returns a ParamResolver for the parameterized unitary."""
        num_angles = 12 * num_qubits * num_layers
        angs = np.pi * (2 * np.random.rand(num_angles) - 1)
        params = ParameterVector('θ', num_angles)
        #print(params)
    
        # Creiamo un dizionario che mappa i parametri ai loro valori
        param_dict = dict(zip(params, angs))
        
        return param_dict
    
    def printCircuit(self, circuit):
        current_dir = os.path.dirname(os.path.realpath(__file__))        
        # Salva il circuito come immagine
        image_path = os.path.join(current_dir, 'PrepStatePassato.png')
        circuit_drawer(circuit, output='mpl', filename=image_path)
        
        # Apri automaticamente l'immagine
        img = Image.open(image_path)
        img.show()

    def calc_purity(self,rho):
        # Calcola rho^2
        rho_squared = np.matmul(rho, rho)
        rho_squared = np.atleast_2d(rho_squared) 
        # Calcola la traccia di rho^2
        trace = np.trace(rho_squared)
        return trace
    
    def calc_dip(self,rho):
        
        return self.trace_Z_rho_squared(rho)
    

    def global_dephasing_channel(self,rho):
        """
    Implementa il canale quantistico Z che defasa nella base standard globale.
    
    :param rho: Matrice densità dell'stato quantistico
    :return: Matrice densità dopo l'applicazione del canale di defasamento globale
    """
        return np.diag(np.diag(rho))
    
        

    def local_dephasing_channel(self,rho, j, n_qubits):
        """
        Implementa il canale quantistico Z_j che defasa nella base standard locale sul qubit j.
        
        :param rho: Matrice densità dell'stato quantistico
        :param j: Indice del qubit su cui applicare il defasamento locale (0-based)
        :param n_qubits: Numero totale di qubit nel sistema
        :return: Matrice densità dopo l'applicazione del canale di defasamento locale
        """
        dim = 2**n_qubits
        Z_j = np.eye(dim, dtype=complex)
        
        for i in range(dim):
            for k in range(dim):
                if i != k and (i ^ k) & (1 << j):
                    Z_j[i, k] = 0
        
        return np.matmul(Z_j, np.matmul(rho, Z_j.conj().T))

    def trace_Z_rho_squared(self,rho, channel_type='global', j=None, n_qubits=None):
        #print(rho)
        #print("bb:" , type(rho))
        #print("RHO TYPE: ", print(type(rho)))
        
        """
        Calcola Tr(Z(ρ)²) per il canale di defasamento specificato.
        
        :param rho: Matrice densità dell'stato quantistico
        :param channel_type: 'global' per Z, 'local' per Z_j
        :param j: Indice del qubit per il defasamento locale (richiesto se channel_type='local')
        :param n_qubits: Numero totale di qubit (richiesto se channel_type='local')
        :return: Il valore di Tr(Z(ρ)²)
        """
        if channel_type == 'global':
            Z_rho = self.global_dephasing_channel(rho)
        elif channel_type == 'local':
            if j is None or n_qubits is None:
                raise ValueError("Per il canale locale, specificare j e n_qubits")
            Z_rho = self.local_dephasing_channel(rho, j, n_qubits)
        else:
            raise ValueError("channel_type deve essere 'global' o 'local'")
        #print("Vecchio:", type(Z_rho))
        Z_rho_squared = np.matmul(Z_rho, Z_rho)
        Z_rho_squared = np.atleast_2d(Z_rho_squared) 
        #print("Nuovo:", type(Z_rho_squared))
        #print(len(Z_rho_squared))
        
        return np.trace(Z_rho_squared)

        """OTTIMIZZAZIONE FUNZIONE DI COSTO"""
    def c1_wrapper(self, parametri, batchStates):
        params = parametri.reshape(self.params.shape)
        dip = self.dip(params, batchStates)  # Passa i parametri corretti
        purity = self.load_purity(params, batchStates)
        c1 = purity - dip
        #print("C1: ", purity, " - ", dip, " = ", c1)
        return c1

    def c1_optimization(self, batchStates):
        # Appiattisci i parametri solo per la fase di ottimizzazione
        flat_params = np.ravel(self.params)
        
        # Definisci i vincoli per i parametri: ciascun parametro deve essere tra 0 e 1
        #bounds = [(0, 1) for _ in range(len(flat_params))]
        
        # Passa i parametri appiattiti a minimize con i vincoli
        result = minimize(self.c1_wrapper, flat_params, args=(batchStates,), method="cobyla")       
        # Puoi riconvertire result.x alla forma originale, se necessario
        optimized_params = result.x.reshape(self.params.shape)
        
        return result, optimized_params
    
    def c2_wrapper(self, parametri, batchStates):
        params = parametri.reshape(self.params.shape)
        pdip = self.pdip(params, batchStates)  # Passa i parametri corretti
        purity = self.load_purity(params, batchStates)
        c2 = purity - pdip
        #print("C1: ", purity, " - ", dip, " = ", c1)
        return c2

    def c2_optimization(self, batchStates):
        # Appiattisci i parametri solo per la fase di ottimizzazione
        flat_params = np.ravel(self.params)
        
        # Definisci i vincoli per i parametri: ciascun parametro deve essere tra 0 e 1
        #bounds = [(0, 1) for _ in range(len(flat_params))]
        
        # Passa i parametri appiattiti a minimize con i vincoli
        result = minimize(self.c2_wrapper, flat_params, args=(batchStates,), method="cobyla")
        
        # Puoi riconvertire result.x alla forma originale, se necessario
        optimized_params = result.x.reshape(self.params.shape)
        
        return result, optimized_params

#for j in range(0,10):
res = Result(1)
