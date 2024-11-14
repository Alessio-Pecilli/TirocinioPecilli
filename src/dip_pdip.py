from typing import Counter
import numpy as np
from PIL import Image
from .layer import LayerPreparation
import os
from qiskit import ClassicalRegister, QuantumRegister, transpile
from qiskit.visualization import circuit_drawer
from qiskit_aer import Aer
from qiskit import QuantumCircuit

class Dip_Pdip:

    def __init__(self,params,state_prep_circ,num_layers):
        self.unitary = LayerPreparation(params,state_prep_circ,num_layers)
        self.layer = self.unitary.mergePrepareUnitary()
        self.double = self.unitary.get_double()
        self.single = self.unitary.get_prepare()
        self.unitaria2 = self.unitary.get_unitary2()
        
        # key for measurements and statistics
        self._measure_key = "z"
        self._pdip_key = "p"

        # initialize the circuits into logical components
        self._num_qubits = len(self.layer.qubits)
        self.qubits = QuantumRegister(self._num_qubits*2)
        self.purity = self.compute_purity(2)#Per il momento sempre puro
        #self.purity = 1
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

    def compute_purity(self,repetitions,
                       simulator=Aer.get_backend('aer_simulator'),
                       ):
        # Computes and returns the (approximate) purity of the state.
        # get the circuit without the diagonalizing unitary
        state_prep_circ = self.unitary.get_prepare()
        state_prep_circ.compose(self.state_overlap(),inplace=True)
        # DEBUG
        #self.printCircuit(state_prep_circ)
        #print("I'm computing the purity as per the circuit:")
        transpiled_circuit = transpile(state_prep_circ, simulator)
        vals = simulator.run(transpiled_circuit, shots=repetitions).result().get_counts(transpiled_circuit)
        #print(vals)
        #print("Fino qui tutto ok")
        self.purity = self.state_overlap_postprocessing(vals,1,transpiled_circuit.num_qubits)   
        return self.purity

    def state_overlap(self):
        """Returns the state overlap circuit as a QuantumCircuit."""
        # Determine the number of qubits to measure
        num_measure = self._num_qubits

        # Create a new quantum circuit with the existing quantum register and the new classical register
        circuit = QuantumCircuit(self.qubits)

        def bell_basis_gates(circuit, qubits, index, num_qubits):
            circuit.cx(qubits[int(index)], qubits[int(index + num_qubits)])  # Gate CNOT
            circuit.h(qubits[int(index)])                                   # Gate Hadamard

        # Add the bell basis gates to the circuit
        for ii in range(int(self._num_qubits)):
            bell_basis_gates(circuit, self.qubits, ii, self._num_qubits)

        # Determine qubits to measure
        qubits_to_measure = self.qubits[ : self._num_qubits] + \
            self.qubits[2 * self._num_qubits : 3 * self._num_qubits]
               
        cr = ClassicalRegister(len(qubits_to_measure), name=self._measure_key)
        circuit.add_register(cr)
        circuit.measure(qubits_to_measure, cr)
        return circuit

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
            list_values = []
            list_index = []

            for h in range(self._num_qubits, self._num_qubits*2):
                if(h != j):
                    clone.h(self.qubits[h])
                    list_values.append(self.qubits[h])
                    list_index.append(h)

            for j in range (0, self._num_qubits):
                list_values.append(self.qubits[j])
                list_index.append(j)

            # add the measurements for the destructive swap test on the pdip qubits
            list_pdip = []

            for ii in range(self._num_qubits):
                qubit_index = ii + self._num_qubits
                
                if qubit_index < len(self.qubits):
                    qubit = self.qubits[qubit_index]
                    
                    if qubit not in list_values:
                        list_pdip.append(qubit)

            # edge case: no qubits in pdip set
            if len(list_pdip) > 0:
                cr = ClassicalRegister(len(list_pdip), name=self._pdip_key)
                clone.add_register(cr)
                clone.measure(list_pdip, cr)
            else:
                print("Inizialmente non metto niente")
            
            cr = ClassicalRegister(len(list_values), name=self._measure_key)
            clone.add_register(cr)
            clone.measure(list_values, cr)
            x = self.getFinalCircuitPDIP(clone)
            #self.printCircuit(x)
            ov += self.pdip_work(x)
            
            #print()
            #print()
        #ov+= self.pdip_work(self.getFinalCircuitDIP())#devo fare un dip diverso
  
        return ov / (self._num_qubits/2)

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

    def separate_p_z_measurements(self,result, circuit):
        # Ottieni i nomi e le dimensioni dei registri classici
        cregs = circuit.cregs
        #print(cregs)
        z_size = next(creg.size for creg in cregs if creg.name == self._measure_key)
        try:
            p_size = next(creg.size for creg in cregs if creg.name == self._pdip_key)
        except:
            p_size = 0
        
        counts = result.get_counts(circuit)
        p_counts = Counter()
        z_counts = Counter()
        
        for bitstring, count in counts.items():
            # Dividi la stringa di bit in base alle dimensioni dei registri
            p_value = bitstring[:p_size]
            z_value = bitstring[:z_size]
            
            p_counts[p_value] += count
            z_counts[z_value] += count
        
        return p_counts, z_counts

    def getFinalCircuitDIP(self):
        combined_circuit = QuantumCircuit(self._num_qubits*2)
        combined_circuit.compose(self.double, inplace=True)
        self.dip_test()
        combined_circuit.compose(self.dip_pdip_circ, inplace=True)
        #self.printCircuit(combined_circuit)
        return combined_circuit
    
    def getFinalCircuitDS(self):
        combined_circuit = QuantumCircuit(self._num_qubits*2)
        #combined_circuit.compose(self.single, inplace=True)
        self.ds_test()
        combined_circuit.compose(self.dip_pdip_circ, inplace=True)
        #self.printCircuit(combined_circuit)
        return combined_circuit
    
    def getFinalCircuitPDIP(self,clone):
        combined_circuit = QuantumCircuit(self._num_qubits*2)
        combined_circuit.compose(self.double, inplace=True)
        combined_circuit.compose(clone, inplace=True)
        #self.printCircuit(combined_circuit)
        return combined_circuit
    
    def purity_calc(self):
        return self.purity


    """ FUNZIONI PER I LAVORI"""
    
    def pdip_work(self,clone,simulator=Aer.get_backend('aer_simulator'), repetitions=5):
        transpiled_circuit = transpile(clone, simulator)
        result = Aer.get_backend('aer_simulator').run(transpiled_circuit, shots=repetitions).result()
            # Ottieni i risultati

        p_counts, z_counts = self.separate_p_z_measurements(result, transpiled_circuit)
        #print(self._measure_key, " : ", z_counts, self._pdip_key," : ",  p_counts)
        
        #Per p_counts devo eseguire il dip...credo

        #print(self.overlap_from_count(p_counts,repetitions))

        mask = self._get_mask_for_all_zero_outcome(p_counts)
        #print("Mask: ",mask)#Ok ma l'ordine dei risultati?
        #keys = list(z_counts.keys())

        # Crea una lista di array numpy da ciascuna chiave binaria
        #outcome = np.array([[int(bit) for bit in key] for key in keys])
        #print("Outcome: ", z_counts)
        result = Counter()
        for binary_string, count in z_counts.items():
            if all(bit == '1' for bit, mask_val in zip(binary_string[::-1], mask) if mask_val):
                result[binary_string] = count
        #toprocess = outcome[mask]
        #print("FINE: ",result)
        #overlap = self.state_overlap_postprocessing(z_counts,1,16)
        overlap = self.MIO_state_overlap(z_counts,repetitions)   
            # DEBUG
        #print("Overlap = ", overlap)
            
            # divide by the probability of getting the all zero outcome
        prob = len(np.where(mask == True)) / len(mask)
        #print("Prob: ",prob)
        #counts = result.histogram(key=self._measure_key)
        prob = result[0] / repetitions if 0 in result.keys() else 0.0
                
        assert 0 <= prob <= 1
        #print("prob =", prob)
                
                
        overlap *= prob
        #print("Scaled overlap =", overlap)
        
        return overlap

    def state_overlap_postprocessing(self, output, nreps, nqubits):
        """Does the classical post-processing for the state overlap algorithm.
        
        Args:
            output [type: np.array]
                The output of the state overlap algorithm.
                
                The format of output should be as follows:
                    vals.size = (number of circuit repetitions, 
                                 number of qubits being measured)

                    the ith column of vals is all the measurements on the
                    ith qubit. The length of this column is the number
                    of times the circuit has been run.
                    
        Returns:
            Estimate of the state overlap as a float
        """
        # =====================================================================
        # constants and error checking
        # =====================================================================

        # number of qubits and number of repetitions of the circuit
        #(nreps, nqubits) = output.shape

        # check that the number of qubits is even
        assert nqubits % 2 == 0, "Input is not a valid shape."
        # initialize variable to hold the state overlap estimate
        overlap = 0.0
        #print("Output ottenuto per la misurazione della purezza: ",output)
        # loop over all the bitstrings produced by running the circuit
        shift = int(nqubits // 4)
        for z in output:
            parity = 1
            #print(shift)
            #for ii in range(shift):
                #print(f"z: [{ii}], z[{ii+shift}]: {z[ii]},{z[ii+shift]}")
            pairs = [z[ii] and z[ii + shift] for ii in range(shift)]
            # DEBUG
            #print(pairs)
            for pair in pairs:
                #print(pair)
                parity *= (-1)**int(pair)

            overlap += parity
            #overlap += (-1)**(all(pairs))
        #print("Purezza: ",overlap)
        return overlap / nreps
    
    def obj_pdip(self, simulator=Aer.get_backend('aer_simulator'), repetitions=1000):
        return self.overlap_pdip(simulator, repetitions)
    

    def obj_dip(self, circuit,repetitions, simulator=Aer.get_backend('aer_simulator')):
        
        transpiled_circuit = transpile(circuit, simulator)
        result = simulator.run(transpiled_circuit, shots=repetitions).result()
        # Ottieni i risultati
        
        counts = result.get_counts(transpiled_circuit)

        print("Risultati della misura:", counts)
        #Costruisco a mano i counts
        #Parametri fissati
        
        return self.overlap_from_count(counts,repetitions)
    
    def obj_dip_counts(self, circuit,repetitions, simulator=Aer.get_backend('aer_simulator')):
        
        transpiled_circuit = transpile(circuit, simulator)
        result = simulator.run(transpiled_circuit, shots=repetitions).result()
        # Ottieni i risultati
        
        counts = result.get_counts(transpiled_circuit)

        return counts
    
    def overlap_from_count(self,counts,repetitions):
        zero_state = '0' * self._num_qubits
        overlap = counts[zero_state] / repetitions if zero_state in counts else 0
        #print("Overlap: ", overlap)
        return overlap

    def obj_ds(self, circuit, repetitions = 100, simulator=Aer.get_backend('aer_simulator')):
        
        transpiled_circuit = transpile(circuit, simulator)
        result = simulator.run(transpiled_circuit, shots=repetitions).result()
        # Ottieni i risultati
        
        counts = result.get_counts(transpiled_circuit)
        #print("counts: ", counts)
        result = []
        for key, value in counts.items():
            result.extend([key] * value)

        #print(len(result))
        lista = list(counts.keys())
        #print("Lista: ", lista)

        return self.MIO_state_overlap(result,repetitions)

        
    def MIO_state_overlap(self,lista,repetitions):
        count = 0
        for elem in lista:
            l = len(elem)
            is_valid = True  # Presupponiamo che la stringa sia valida
            #print("Stringa: ", elem)
            for i in range(l // 2):
                #print("Verifico: ",i,i+l//2, " : ", elem[i], elem[i + l//2])
                if elem[i] == '1' and elem[i + l // 2] == '1':
                    is_valid = False  # Se trovi una coppia "11", la stringa è invalida
                    #print("Smetto di analizzare questo elemento")
                    break  # Non serve continuare a controllare le altre coppie

            if is_valid:
                count += 1
                #print("Count aumentato")

            
        #print("Ho trovato n coppie = ", count)

            #Parte della visione
        #print("Risultati della misura:", counts)
        
        # Step 5: Calculate overlap, where '0'*num_qubits is the all-zeros state
        
        #print("Overlap: ", overlap)
        return count
    
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
        result = [int(key) == 0 for key in outcome.elements()]

        #print("res:",result)
        return result
    
    def run_resolved(self, angles, simulator=Aer.get_backend('aer_simulator'), repetitions=1000):
        return
    
    def resolved_algorithm(self,angles):
        return
    
    def get_binary(self):
        return self.unitary.get_binary()
    

