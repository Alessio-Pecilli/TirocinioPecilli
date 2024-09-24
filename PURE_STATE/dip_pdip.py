import cirq
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from layer import LayerPreparation
from prepState import StatePreparation
import readMINST
from qiskit.circuit import ParameterVector
import random
from qutip import Qobj, ket2dm
from qiskit.quantum_info import Statevector
import os
from qiskit.circuit import Parameter
import math
from qiskit import ClassicalRegister, QuantumRegister, transpile
from qiskit.visualization import circuit_drawer
import qiskit_aer
from qiskit_aer import Aer
from qiskit_aer import AerSimulator
from qiskit import QuantumCircuit
from qiskit.visualization import plot_histogram

class Dip_Pdip:

    def __init__(self):
        self.unitary = LayerPreparation()
        self.layer = self.unitary.mergePrepareUnitary()
        self.double = self.unitary.get_double()
        
        # key for measurements and statistics
        self._measure_key = "z"
        self._pdip_key = "p"

        # initialize the circuits into logical components
        self._num_qubits = len(self.layer.qubits)
        self.qubits = QuantumRegister(self._num_qubits*2)
        self.purity = 1#Per il momento sempre puro
        return
    
    def ds_test(self):
        self.clear_dip_pdip_circ()
        self.qubits = self.dip_pdip_circ.qubits  # Aggiorna self.qubits
        for ii in range(self._num_qubits):
            i1 = ii + int(self._num_qubits)
            self.dip_pdip_circ.cx(self.qubits[i1],self.qubits[ii])
            self.dip_pdip_circ.h(self.qubits[i1])
                
        cr = ClassicalRegister(self._num_qubits*2, name=self._measure_key)
        self.dip_pdip_circ.add_register(cr)
        self.dip_pdip_circ.measure(self.qubits, cr)
    
    def dip_test(self):
        # Assicurati che ci siano almeno 16 qubit nel circuito
        self.clear_dip_pdip_circ()
        self.qubits = self.dip_pdip_circ.qubits  # Aggiorna self.qubits

        for ii in range(self._num_qubits):
            i1 = ii + int(self._num_qubits)
            self.dip_pdip_circ.cx(self.qubits[i1], self.qubits[ii])  # Usa i qubit corretti
        
        qubits_to_measure = self.qubits[:self._num_qubits]
        cr = ClassicalRegister(len(qubits_to_measure), name=self._measure_key)
        self.dip_pdip_circ.add_register(cr)
        self.dip_pdip_circ.measure(qubits_to_measure, cr)

    def pdip_test(self):
        #self.clear_dip_pdip_circ()
        self.clear_dip_pdip_circ()
        self.qubits = self.dip_pdip_circ.qubits  # Aggiorna self.qubits
        for ii in range(self._num_qubits):
            i1 = ii + int(self._num_qubits)
            self.dip_pdip_circ.cx(self.qubits[i1],self.qubits[ii])

        ov = 0.0
        for j in range(self._num_qubits, self._num_qubits * 2):
            clone = self.dip_pdip_circ.copy()
            list = []
            for h in range(j, self._num_qubits * 2):
                clone.h(self.qubits[h])
                list.append(self.qubits[h])

            for j in range (0, self._num_qubits):
                list.append(self.qubits[j])

            cr = ClassicalRegister(len(list), name=self._measure_key)
            clone.add_register(cr)
            clone.measure(list, cr)
            self.pdip_work(clone)
  
        return ov / self._num_qubits

    def clear_dip_pdip_circ(self):
        """Sets the dip test circuit to be a new, empty circuit."""
        self.dip_pdip_circ = QuantumCircuit(self._num_qubits*2) 

    def printCircuit(self, circuit):
        current_dir = os.path.dirname(os.path.realpath(__file__))        
        # Salva il circuito come immagine
        image_path = os.path.join(current_dir, 'PrepStatePassato.png')
        circuit_drawer(circuit, output='mpl', filename=image_path)
        
        # Apri automaticamente l'immagine
        img = Image.open(image_path)
        img.show()

    def getFinalCircuitDIP(self):
        combined_circuit = QuantumCircuit(self._num_qubits*2)
        combined_circuit.compose(self.double, inplace=True)
        self.dip_test()
        combined_circuit.compose(self.dip_pdip_circ, inplace=True)
        #self.printCircuit(combined_circuit)
        return combined_circuit
    
    def getFinalCircuitDS(self):
        combined_circuit = QuantumCircuit(self._num_qubits*2)
        combined_circuit.compose(self.double, inplace=True)
        self.ds_test()
        combined_circuit.compose(self.dip_pdip_circ, inplace=True)
        #self.printCircuit(combined_circuit)
        return combined_circuit
    


    """ FUNZIONI PER I LAVORI"""
    
    def pdip_work(self,clone,simulator=Aer.get_backend('aer_simulator'), repetitions=1000):
        transpiled_circuit = transpile(clone, simulator)
        result = simulator.run(transpiled_circuit, shots=1000).result()
        # Ottieni i risultati
        dipcounts = result.measurements[self._measure_key]
        pdipcount = result.measurements[self._pdip_key]
        
        mask = self._get_mask_for_all_zero_outcome(dipcounts)
        toprocess = pdipcount[mask]

        
        return
    
    def obj_pdip(self, simulator=Aer.get_backend('aer_simulator'), repetitions=1000):
        return self.purity - self.overlap_pdip(simulator, repetitions)
    

    def obj_dip(self, circuit, simulator=Aer.get_backend('aer_simulator'), repetitions=1000):
        
        transpiled_circuit = transpile(circuit, simulator)
        result = simulator.run(transpiled_circuit, shots=1000).result()
        # Ottieni i risultati
        
        counts = result.get_counts(transpiled_circuit)

        #print("Risultati della misura:", counts)
        
        # Step 5: Calculate overlap, where '0'*num_qubits is the all-zeros state
        zero_state = '0' * self._num_qubits
        overlap = counts[zero_state] / repetitions if zero_state in counts else 0
        #print("Overlap: ", overlap)
        return self.purity - overlap
    
    def obj_ds(self, circuit, simulator=Aer.get_backend('aer_simulator'), repetitions=1000):
        
        transpiled_circuit = transpile(circuit, simulator)
        result = simulator.run(transpiled_circuit, shots=1000).result()
        # Ottieni i risultati
        
        counts = result.get_counts(transpiled_circuit)

        #print("Risultati della misura:", counts)
        
        # Step 5: Calculate overlap, where '0'*num_qubits is the all-zeros state
        zero_state = '0' * self._num_qubits
        overlap = counts[zero_state] / repetitions if zero_state in counts else 0
        #print("Overlap: ", overlap)
        return self.purity - overlap
    
    def _get_mask_for_all_zero_outcome(self, outcome):
        """Returns a mask corresponding to indices ii from 0 to len(outcome) 
        such that np.all(outcome[ii]) == True.
        
        Args:
            outcome : numpy.ndarray 
                The output of the state overlap algorithm.
                
                The format of output should be as follows:
                    outcome.size = (number of circuit repetitions, 
                                 number of qubits being measured)

                    the ith column of outcome is all the measurements on the
                    ith qubit. The length of this column is the number
                    of times the circuit has been run.
        """
        mask = []
        for meas in outcome:
            if not np.any(meas):
                mask.append(True)
            else:
                mask.append(False)
        return np.array(mask)
    
    def run_resolved(self, angles, simulator=Aer.get_backend('aer_simulator'), repetitions=1000):
        return
    
    def resolved_algorithm(self,angles):
        return

#dip = Dip_Pdip()
