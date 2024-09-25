

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

class Result:

    def __init__(self,n):
        # Get a VQSD instance
        n = 4
        batch = self.load_img(n)
        self.c1(batch)
        return
    
    def evaluete_c1(self,purity,dip):
        return purity - dip
    
    def load_img(self,n):
        state_prep = StatePreparation(n)
        batch = state_prep.LoadVectorMultiplo()
        return batch
    
    def c1(self,batch):
        #Per ora uso n_qubit e n_layer fissi
        self._num_qubits = 8
        self.num_layers = 1
        x = self.get_param_resolver(self._num_qubits, self.num_layers)
        params = self.min_to_vqsd(x,self._num_qubits, self.num_layers) 
        print(self.dip(params,batch))
        #self.load_purity(params,batch) - self.dip(params,batch)

        #scelgo n immagini(1 a N)
        #Sommo i count
        #Passo il ris nei vari calcoli
        return
    
    def dip(self,params,batch):
        counts = []
        for ii in range(len(batch)):
            circuit = Dip_Pdip(params,batch[ii],1)
            circ = circuit.getFinalCircuitDIP()
            count = circuit.obj_dip_counts(circ,1)
            counts.append(count)
            #devo convertire list in counts forse
            #<class 'qiskit.result.counts.Counts'> <class 'list'>
        return self.overlap_from_count(counts,len(batch))
    
    def overlap_from_count(self,counts,repetitions):
        zero_state = '0' * self._num_qubits
        print(counts, repetitions)
        overlap = counts[zero_state] / repetitions if zero_state in counts else 0
        #print("Overlap: ", overlap)
        return overlap

    def load_purity(self,params,batch):
        counts = []
        for ii in range(len(batch)):
            circuit = Dip_Pdip(params,batch[ii],1)
            circ = circuit.purity()
            count = circuit.obj_dip_counts(circ,1)
            counts.append(count)
            
        return 
    
    def min_to_vqsd(self, param_list, num_qubits, num_layer):
        # Verifica il numero totale di elementi
        #assert len(param_list) % 6 == 0, "invalid number of parameters"
        
        param_values = np.array(list(param_list.values()))#ho tolto .values per una migliore visualizzazione
        x = param_values.reshape(num_layer,2 ,num_qubits//2 ,12)
        x_reshaped = x.reshape(num_layer, 2, num_qubits // 2, 4, 3)
       # print(x_reshaped)
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
    
res = Result(4)