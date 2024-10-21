import cirq
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
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

class LayerPreparation:

    def __init__(self):
        self.prep_state = StatePreparation(1)
        
        # Prepara il circuito di stato e salva il numero di qubit
        self.state_prep_circ = self.prep_state.PrepareONECircuit()
        self._num_qubits = int(self.state_prep_circ.num_qubits)
        
        self.qubits = QuantumRegister(self._num_qubits)
        self.unitary_circ = QuantumCircuit(self.qubits)
        #print("NUMERO DI QUBIT: ", num_qubits)
        self.num_layers = 2

        #print(f"num_qubits: {num_qubits}")
        #print(f"num_layers: {num_layers}")
        #print(f"Calcolo: 12 * (num_qubits // 2) * num_layers = {12 * (num_qubits // 2) * num_layers}")
        x = self.get_param_resolver(self._num_qubits, self.num_layers)
        params = self.min_to_vqsd(x,self._num_qubits, self.num_layers)
        #params = np.random.rand(num_layers, vqsd._num_qubits // 2, 12)
        for l in range(self.num_layers):
            self.layer(params[l][0], params[l][1],)
            #self.printCircuit(self.unitary_circ)
            #self.layer(params[l][0], params[l][1], copy=1)
        #self.printCircuit(self.unitary_circ)
        return
    
    def __init__(self,params,state_prep_circ,num_layers):

        self._num_qubits = int(state_prep_circ.num_qubits)
        self.state_prep_circ = state_prep_circ
        self.qubits = QuantumRegister(self._num_qubits)
        self.unitary_circ = QuantumCircuit(self.qubits)
        
        #print("NUMERO LAYER: ", num_layers)
        #print("Devo applicare questi parametri: ", params)
           
        for l in range(num_layers):
                #print("Applico correttamente i parametri")
                #self.printCircuit(self.unitary_circ)
            self.layer(params[l][0], params[l][1],)

        #self.printCircuit(self.unitary_circ)
        return
    
    def layer(self, params, shifted_params):
        """Implements a single layer of the diagonalizing unitary.

        input:
            params [type: numpy.ndarray]
                parameters for the layer of gates.
                shape should be (n // 2, 12) where n is the number of qubits

            copy [type: int (0 or 1), default=0]
                the copy of the state to apply the layer to
            PER ORA SEMPRE = 0
        modifies:
            self.unitary_circ
                appends the layer of operations to self.unitary_circ
                
        """
        n = self._num_qubits
        if params.size != self.num_angles_required_for_layer()/2:
            raise ValueError("incorrect number of parameters for layer, params.size: ", params.size, " angoli richiesti: ", self.num_angles_required_for_layer()/2)

        # helper function for indexing loops
        stop = lambda n: n - 1 if n % 2 == 1 else n

        #shift = 2 * n * copy
        shift = 0 #non ho copie, lavoro con un singolo stato
        for ii in range(0, n - 1, 2):
            qubits = [self.qubits[ii + shift], self.qubits[ii + 1 + shift]]
            gate_params = params[ii // 2]
            self._apply_gate(qubits, gate_params)
            
        #self.printCircuit(self.unitary_circ)

        stop = lambda n: n - 1 if n % 2 == 1 else n

        shift =  self._num_qubits * 0#0 = COPY
        
        """for ii in range(0, stop(n), 2):
            iiq = ii + shift
            self.printCircuit(self.unitary_circ)
            self._apply_gate(self.qubits[iiq : iiq + 2], params[ii // 2])"""
        
        if n >= 2:
            for ii in range(1, n, 2):
                self._apply_gate([self.qubits[ii],
                      self.qubits[(ii + 1) % n]],
                     shifted_params[ii // 2])             
        
         

    def _apply_gate(self, qubits, params):
            """Helper function to append the two qubit gate."""
            ##print("len q", len(qubits), "len par: ", len(params))
            q = qubits[0]
            p = params[0]
            self._rot(q,p)
            q = qubits[1]
            p = params[1]
            self._rot(q,p)
            self.unitary_circ.cx(qubits[0], qubits[1])
            q = qubits[0]
            p = params[2]
            self._rot(q,p)
            q = qubits[1]
            p = params[3]
            self._rot(q,p)
            self.unitary_circ.cx(qubits[0], qubits[1])
            


    def _rot(self, qubit, params):
        """Helper function that returns an arbitrary rotation of the form
        R = Rz(params[2]) * Ry(params[1]) * Rx(params[0])
        on the qubit, e.g. R |qubit>.

        Note that order is reversed when put into the circuit. The circuit is:
        |qubit>---Rx(params[0])---Ry(params[1])---Rz(params[2])---
        """
        #print("parames 0", params)
        self.unitary_circ.rx(params[0], qubit)
        self.unitary_circ.ry(params[1], qubit)
        self.unitary_circ.rz(params[2], qubit)
    
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

    def num_angles_required_for_layer(self):
        """Returns the number of angles need in a single layer of the
        diagonalizing unitary.
        """
        return 12 * (self._num_qubits)
        "12 * 8 = 96"

    def printCircuit(self, circuit):
        current_dir = os.path.dirname(os.path.realpath(__file__))        
        # Salva il circuito come immagine
        image_path = os.path.join(current_dir, 'PrepStatePassato.png')
        circuit_drawer(circuit, output='mpl', filename=image_path)
        
        # Apri automaticamente l'immagine
        img = Image.open(image_path)
        img.show()

    def get_unitary2(self):

        combined_circuit = QuantumCircuit(self._num_qubits)
        combined_circuit.compose(self.unitary_circ,inplace=True)
        return combined_circuit

    def mergePrepareUnitary(self):
        combined_circuit = QuantumCircuit(self._num_qubits)
        combined_circuit.compose(self.state_prep_circ,inplace=True)
        combined_circuit.compose(self.unitary_circ,inplace=True)
        #self.printCircuit(combined_circuit)
        return combined_circuit

    def get_double(self):

        combined_circuit = QuantumCircuit(self._num_qubits)
        combined_circuit.compose(self.state_prep_circ,inplace=True)
        combined_circuit.compose(self.unitary_circ,inplace=True)
        qc = QuantumCircuit(combined_circuit.num_qubits*2)

        qc.compose(combined_circuit, range(combined_circuit.num_qubits),inplace=True)
        qc.compose(combined_circuit, range(combined_circuit.num_qubits, combined_circuit.num_qubits * 2),inplace=True)
        #self.printCircuit(qc)
        return qc
    
    def get_prepare(self):
        circ = self.state_prep_circ
        qc = QuantumCircuit(circ.num_qubits*2)
        qc.compose(circ, range(circ.num_qubits),inplace=True)
        qc.compose(circ, range(circ.num_qubits, circ.num_qubits * 2),inplace=True)

        return qc
    
    def get_binary(self):
        #self.printCircuit(self.mergePrepareUnitary())
        return self.prep_state.getBinary()

