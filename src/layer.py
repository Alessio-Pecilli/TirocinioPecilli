
import numpy as np
from PIL import Image
from .prepState import StatePreparation
from qiskit.circuit import ParameterVector
import os
from qiskit import QuantumRegister
from qiskit.visualization import circuit_drawer
from qiskit import QuantumCircuit

class LayerPreparation:
    
    def __init__(self,params,state_prep_circ,num_layers):

        self._num_qubits = int(state_prep_circ.num_qubits)
        self.state_prep_circ = state_prep_circ
        self.qubits = QuantumRegister(self._num_qubits)
        self.unitary_circ = QuantumCircuit(self.qubits)
           
        for l in range(num_layers):
            self.layer(params[l][0], params[l][1],)

        #self.printCircuit(self.unitary_circ)
        return
    
    def layer(self, params, shifted_params):
        n = self._num_qubits
        if params.size != self.num_angles_required_for_layer()/2:
            raise ValueError("Params.size: ", params.size, " angoli richiesti: ", self.num_angles_required_for_layer()/2)

        #shift = 2 * n * copy
        shift = 0 #non ho copie, lavoro con un singolo stato
        for ii in range(0, n - 1, 2):
            qubits = [self.qubits[ii + shift], self.qubits[ii + 1 + shift]]
            gate_params = params[ii // 2]
            self._apply_gate(qubits, gate_params)

        shift =  self._num_qubits * 0#0 = COPY
        
        if n >= 2:
            for ii in range(1, n, 2):
                self._apply_gate([self.qubits[ii],
                      self.qubits[(ii + 1) % n]],
                     shifted_params[ii // 2])             
        
         

    def _apply_gate(self, qubits, params):
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
        num_angles = 12 * num_qubits * num_layers
        angs = np.pi * (2 * np.random.rand(num_angles) - 1)
        params = ParameterVector('θ', num_angles)
        param_dict = dict(zip(params, angs))
        
        return param_dict

    def num_angles_required_for_layer(self):
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

