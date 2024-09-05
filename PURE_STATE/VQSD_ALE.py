"""VQSD.py

Contains a class for VQSD circuits utilizing Qiskit.
"""

# =============================================================================
# imports
# =============================================================================

import cirq
from qiskit.circuit import Parameter
from qiskit.circuit.library import RXGate, RYGate, RZGate
from qiskit import transpile, assemble, QuantumRegister, QuantumCircuit, ClassicalRegister
import numpy as np
from qiskit.exceptions import QiskitError
import qiskit_aer
from qiskit_aer import Aer
from qiskit_aer import AerSimulator
from prepState import StatePreparation

# =============================================================================
# VQSD class
# =============================================================================

class VQSD:
    # =========================================================================
    # init method
    # =========================================================================

    def __init__(self, measure_key='z'):
        """Initializes a VQSD circuit.

        input:
            num_qubits [type: int]
                the number of qubits in the VQSD circuit

            measure_key [type: str]
                used to easily access measurement results with cirq

        data:
            qubits [type: cirq.QubitId]
                qubits in the circuit

            state_prep_circ [type: cirq.Circuit]
                state preperation part of the VQSD circuit

            unitray_circ [type: cirq.Circuit]
                diagonalizing unitary part of the VQSD circuit

            dip_test_circ [type: cirq.Circuit]
                dip test part of the VQSD circuit

            purity [type: float, initialized to None]
                the purity of the state being diagonalized
                once the circuit is formed, purity can be computed using the
                method self.compute_purity
        """
        # set the number of qubits and get some qubits
        # TODO: add option for mixed/pure state
        # for pure, only need 2 * num_qubits
        # Crea un'istanza di PrepState
        self.prep_state = StatePreparation(1)
        
        # Prepara il circuito di stato e salva il numero di qubit
        self.state_prep_circ = self.prep_state.PrepareONECircuit()
        self._num_qubits = int(self.state_prep_circ.num_qubits/2)
        
        self._total_num_qubits = self._num_qubits * 2
        self.qubits = QuantumRegister(self._total_num_qubits)

        # key for measurements and statistics
        self._measure_key = measure_key
        self._pdip_key = "p"

        # initialize the circuits into logical components
        
        self.unitary_circ = QuantumCircuit(self.qubits)
        self.dip_test_circ = QuantumCircuit(self.qubits)
        #self.prep_state.printCircuit(self.unitary_circ)
        #self.prep_state.printCircuit(self.dip_test_circ)

        # initialize the purity of the state
        self.purity = None

    # =========================================================================
    # getter methods
    # =========================================================================

    def get_num_qubits(self):
        """Returns the number of qubits in the circuit."""
        return self._num_qubits

    # =========================================================================
    # methods to clear/reset circuits
    # =========================================================================
    
    def clear_unitary_circ(self):
        """Sets the unitary circuit to be a new, empty circuit."""
        self.unitary_circ = QuantumCircuit(self.qubits)
    
    def clear_dip_test_circ(self):
        """Sets the dip test circuit to be a new, empty circuit."""
        self.dip_test_circ = QuantumCircuit(self.qubits)
        #print("A DIP DO CL: ", self.dip_test_circ.num_clbits)

    # =========================================================================
    # circuit methods
    # =========================================================================
    
    # =========================================================================
    # ansatz methods
    # =========================================================================

    def product_ansatz(self, params, gate):
        """Modifies self.unitary_circ by appending a product ansatz with a
        gate on each qubit.
        """
        # make sure the number of parameters is correct
        if len(params) != self._num_qubits:
            raise ValueError("Incorrect number of parameters.")
        ##print("len param: ", len(params))
    
        n = self._num_qubits

        ##print("num q: ", n)
        
        for ii in range(len(params)):
            g = gate(params[ii])
            # Calcola l'indice del secondo qubit, assicurandosi che sia valido
            ##print("Guarda" , ii+n)
            self.unitary_circ.append(g, [self.qubits[ii]])
            self.unitary_circ.append(g, [self.qubits[ii+n]])
            
    
    def unitary(self, num_layers, params, shifted_params, copy):
        """Adds the diagonalizing unitary to self.unitary_circ.

        input:
            num_layers [type: int]
                number of layers to implement in the diagonalizing unitary.

            params [type: list<list<list<float>>>]
                parameters for every layer of gates
                the format of params is as follows:

                params = [[rotations for first layer],
                          [rotations for second layer],
                          ...,
                          [rotations for last layer]]

            shifted_params [type: list<list<list<float>>>]
                parameters for the shifted layers of gates
                format is the same as the format for params above

            copy [type: int, 0 or 1, default value = 0]
                the copy of the state to perform the unitary on
        """
        # TODO: implement
        pass

    def layer(self, params, shifted_params, copy):
        """Implements a single layer of the diagonalizing unitary.

        input:
            params [type: numpy.ndarray]
                parameters for the layer of gates.
                shape should be (n // 2, 12) where n is the number of qubits

            copy [type: int (0 or 1), default=0]
                the copy of the state to apply the layer to

        modifies:
            self.unitary_circ
                appends the layer of operations to self.unitary_circ
        """
        n = self._num_qubits
        if params.size != self.num_angles_required_for_layer():
            raise ValueError("incorrect number of parameters for layer")

        shift = n * copy

        for ii in range(0, n - 1, 2):
            qubits = [self.qubits[ii + shift], self.qubits[ii + 1 + shift]]
            gate_params = params[ii // 2]
            self._apply_gate(qubits, gate_params)

        stop = lambda n: n - 1 if n % 2 == 1 else n

        shift = 2 * self._num_qubits * copy

        for ii in range(0, stop(n), 2):
            iiq = ii + shift
            if iiq + 1 >= len(self.qubits):
                continue
            self._apply_gate(self.qubits[iiq : iiq + 2], params[ii // 2])

        if n > 2:
            for ii in range(1, n, 2):
                iiq = ii + shift
                if iiq + 1 >= len(self.qubits):
                    continue
                self._apply_gate([self.qubits[iiq], self.qubits[(iiq + 1) % (2 * n)]], shifted_params[ii // 2])
        #self.prep_state.printCircuit(self.unitary_circ)

    def _apply_gate(self, qubits, params):
            """Helper function to append the two qubit gate."""
            #print("len q", len(qubits), "len par: ", len(params))
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
        self.unitary_circ.rx(params[0], qubit)
        self.unitary_circ.ry(params[1], qubit)
        self.unitary_circ.rz(params[2], qubit)

    def dip_test(self, pdip=False):
        """Implements the dip test circuit.

        modifies:
            self.dip_test_circ
                appends the dip test circuit with measurements
                on the top state.
        """
        # TODO: implement option for partial dip test circuit
        # or make another method (e.g., pdip_test(self, qbit_to_measure))

        # do the cnots
        for ii in range(self._num_qubits):
            self.dip_test_circ.cx( self.qubits[ii + self._num_qubits],self.qubits[ii])
            
        
        qubits_to_measure = self.qubits[:self._num_qubits]
        cr = ClassicalRegister(len(qubits_to_measure), 'cr')  # Aggiungi un nome al registro
        self.dip_test_circ.add_register(cr)  # Aggiungi questa linea
        self.dip_test_circ.measure(qubits_to_measure, cr)


        ##print(f"Misurazione aggiunta per i qubit: {qubits_to_measure}")
    
    def pdip_test(self, pdip_qubit_indices): 
        #print("pdip_qubit_indices ", pdip_qubit_indices)
        
        # Fai i CNOT
        for ii in range(self._num_qubits):
            self.dip_test_circ.cx(self.qubits[ii + self._num_qubits], self.qubits[ii])
        
        # Aggiungi Hadamard sui qubit non in PDIP test
        all_qubit_indices = set(range(self._num_qubits))
        qubit_indices_to_hadamard = list(all_qubit_indices - set(pdip_qubit_indices))
        qubits_to_hadamard = [self.qubits[ii + self._num_qubits] for ii in qubit_indices_to_hadamard]
        for qubit in qubits_to_hadamard:
            self.dip_test_circ.h(qubit)
        
        # Crea due registri classici separati per DIP e PDIP
        cr_dip = ClassicalRegister(self._num_qubits, 'cr_dip')
        cr_pdip = ClassicalRegister(self._num_qubits, 'cr_pdip')
        self.dip_test_circ.add_register(cr_dip)
        self.dip_test_circ.add_register(cr_pdip)
        
        # Misura i qubit DIP
        self.dip_test_circ.measure(self.qubits[:self._num_qubits], cr_dip)
        
        # Misura i qubit PDIP
        pdip_qubits = [self.qubits[ii + self._num_qubits] for ii in pdip_qubit_indices]
        self.dip_test_circ.measure(pdip_qubits, cr_pdip[:len(pdip_qubit_indices)])
            
            

    def state_overlap(self):
        """Returns the state overlap circuit as a QuantumCircuit."""
        # Determine the number of qubits to measure
        num_measure = self._num_qubits

        # Create a new classical register
        cr = ClassicalRegister(num_measure)

        # Create a new quantum circuit with the existing quantum register and the new classical register
        circuit = QuantumCircuit(self.qubits, cr)

        def bell_basis_gates(circuit, qubits, index, num_qubits):
            circuit.cx(qubits[int(index)], qubits[int(index + num_qubits)])  # Gate CNOT
            circuit.h(qubits[int(index)])                                   # Gate Hadamard

        # Add the bell basis gates to the circuit
        for ii in range(int(self._num_qubits)):
            bell_basis_gates(circuit, self.qubits, ii, self._num_qubits)

        # Determine qubits to measure
        qubits_to_measure = self.qubits[ : self._num_qubits] + \
            self.qubits[2 * self._num_qubits : 3 * self._num_qubits]
        
        
        circuit.measure(qubits_to_measure,cr)

        return circuit

    # =========================================================================
    # helper circuit methods
    # =========================================================================

    def _get_unitary_symbols(self):
        """Returns a list of symbols required for the unitary ansatz."""
        # TODO: take into account the number of layers in the unitary
        # this should change how num_angles_required_for_unitary() is called
        # and the implementation of this method should change
        # the input arguments to this method should also include the number
        # of layers, as should num_angles_required_for_unitary()
        num_symbols_required = self.num_angles_required_for_unitary()
        return np.array(
            [cirq.Symbol(ii) for ii in range(num_symbols_required)]
            )

    def _reshape_sym_list_for_unitary(self):
        """Reshapes a one-dimensional list into the shape required by
        VQSD.layer.
        """
        pass

    def num_angles_required_for_unitary(self):
        """Returns the number of angles needed in the diagonalizing unitary."""
        # TODO: take into account the number of layers.
        # probably need to add a member variable to the class keeping track of
        # the number of layers.
        # should be 12 * num_qubits * num_layers
        return 12 * (self._num_qubits // 2)

    def num_angles_required_for_layer(self):
        """Returns the number of angles need in a single layer of the
        diagonalizing unitary.
        """
        return 12 * (self._num_qubits // 2)

    
        # =====================================================================
        # constants and error checking
        # =====================================================================

        # number of qubits and number of repetitions of the circuit
    def state_overlap_postprocessing(self, output):
        
        #print(f"Input ricevuto: {output}")
        
        # Converti l'input in un array numpy se non lo è già
        output = np.array(output)
        
        # Se l'input è una stringa, convertila in un array di interi
        if output.dtype.kind == 'U':
            output = np.array([[int(bit) for bit in row] for row in output])
        
        if output.ndim == 1:
            output = output.reshape(1, -1)
        
        #print(f"Output dopo il reshape: {output}")
        
        (nreps, nqubits) = output.shape
        #print(f"nreps: {nreps}, nqubits: {nqubits}")

        # Se il numero di qubits è dispari, aggiungi una colonna di zeri
        if nqubits % 2 != 0:
            output = np.pad(output, ((0, 0), (0, 1)), mode='constant')
            nqubits += 1
            #print(f"Numero dispari di qubits. Aggiunta una colonna di zeri. Nuovo nqubits: {nqubits}")
        
        print("GLI OUTPUT CON CUI LAVORO SONO: ", output)
        overlap = 0.0

        shift = nqubits // 2
        for z in output:
            parity = 1
            pairs = [(z[ii], z[ii + shift]) for ii in range(shift)]
            #print(f"Coppie: {pairs}")
            for pair in pairs:
                parity *= (-1)**(pair[0] and pair[1])
            overlap += parity
            #print(f"Parity: {parity}, Current overlap: {overlap}")
        if(overlap < 0):
            print("STATE PROCESSING, L'OVERLAP DEL PDIP CON CUI LAVORO VALE: ", overlap)
        final_overlap = overlap / nreps
        #print(f"Overlap finale: {final_overlap}")
        #print("--- Fine state_overlap_postprocessing ---\n")
        return final_overlap



    # =========================================================================
    # methods for running the circuit and getting the objective function
    # =========================================================================

    def algorithm(self):
        #print("ENTRO IN algorithm")
        """Returns the total algorithm of the VQSD circuit, which consists of
        state preperation, diagonalizing unitary, and dip test.

        rtype: QuantumCircuit
        """
        #print(f"state_prep_circ - num_qubits: {self.state_prep_circ.num_qubits}, num_clbits: {self.state_prep_circ.num_clbits}")
        #print(f"state_unitary_circ - num_qubits: {self.unitary_circ.num_qubits}, num_clbits: {self.unitary_circ.num_clbits}")
        #print(f"dip_test_circ - num_qubits: {self.dip_test_circ.num_qubits}, num_clbits: {self.dip_test_circ.num_clbits}")
        
        #nel caso le caratteristiche siano
        #state_prep_circ - num_qubits: 10, num_clbits: 10
        #state_unitary_circ - num_qubits: 20, num_clbits: 0
        #dip_test_circ - num_qubits: 20, num_clbits: 19
        maxQ = max(self.state_prep_circ.num_qubits,self.unitary_circ.num_qubits,self.dip_test_circ.num_qubits)
        maxC = max(self.state_prep_circ.num_clbits,self.unitary_circ.num_clbits,self.dip_test_circ.num_clbits)
        combined_circuit = QuantumCircuit(self._total_num_qubits)

        # Componi state_prep_circ
        combined_circuit.compose(self.state_prep_circ, qubits=range(self.state_prep_circ.num_qubits), clbits=range(self.state_prep_circ.num_clbits), inplace=True)

        # Componi state_unitary_circ
        combined_circuit.compose(self.unitary_circ, qubits=range(self.unitary_circ.num_qubits),clbits=range(self.unitary_circ.num_clbits), inplace=True)

        # Componi dip_test_circ
        combined_circuit.compose(self.dip_test_circ, qubits=range(self.dip_test_circ.num_qubits),clbits=range(self.dip_test_circ.num_clbits), inplace=True)
        self.prep_state.printCircuit(combined_circuit)
        #print(f"Parametri nel circuito combinato: {combined_circuit.parameters}")
        #print(f"combined_circuit - num_qubits: {combined_circuit.num_qubits}, num_clbits: {combined_circuit.num_clbits}")
        
        return combined_circuit

    def resolved_algorithm(self, angles):
        ##print("ENTRO IN resolved_algorithm")
        circuit = self.algorithm()
        #print("QUESTO normale HA CLBITS: ",circuit.num_clbits)
        params = list(circuit.parameters)
        
        if len(angles) < len(params):
            ##print(f"Attenzione: forniti {len(angles)} angoli, ma il circuito ha {len(params)} parametri.")
            # Estendi angles con valori casuali se necessario
            angles = list(angles) + [np.random.uniform(0, 2*np.pi) for _ in range(len(params) - len(angles))]
        elif len(angles) > len(params):
            ##print(f"Attenzione: forniti {len(angles)} angoli, ma il circuito ha solo {len(params)} parametri.")
            angles = angles[:len(params)]
        
        param_dict = dict(zip(params, angles))
        resolved_circuit = circuit.assign_parameters(param_dict)
        #print("QUESTO RISOLTO HA CLBITS: ",resolved_circuit.num_clbits)
        
        #if resolved_circuit.parameters:
            #print("Attenzione: alcuni parametri non sono stati risolti:", resolved_circuit.parameters)
        
        return resolved_circuit

    def run(self,
            simulator=Aer.get_backend('aer_simulator'),
            repetitions=10):
        """Runs the algorithm and returns the result.

        rtype: cirq.TrialResult
        """
        return simulator.run(self.algorithm(), repetitions=repetitions, memory = True)

    def run_resolved(self, angles, simulator=Aer.get_backend('aer_simulator'), repetitions=1000):
        #print("\n--- Inizio run_resolved ---")
        #print(f"Angoli ricevuti: {angles}")
        #print(f"Numero di ripetizioni: {repetitions}")
        
        circuit = self.resolved_algorithm(angles)
        
        if circuit.parameters:
            #print("Parametri non risolti:", circuit.parameters)
            param_binds = [{param: np.random.uniform(0, 2*np.pi) for param in circuit.parameters}]
            #print(f"Binding dei parametri: {param_binds}")
            job = simulator.run(circuit, shots=repetitions, parameter_binds=param_binds, memory = True)
        else:
            job = simulator.run(circuit, shots=repetitions, memory = True)
        
        try:
            result = job.result()
            #print("Esecuzione del job completata con successo")
            return result
        except Exception as e:
            #print(f"Errore durante l'esecuzione del job: {str(e)}")
            #print("Circuito:")
            #print(circuit)
            #print("Parametri del circuito:", circuit.parameters)
            return None
        #finally:
            #print("--- Fine run_resolved ---\n")

    def obj_dip_resolved(self, angles, simulator=Aer.get_backend('aer_simulator'), repetitions=1000):
        if not self.purity:
            self.compute_purity()
        
        circuit = self.resolved_algorithm(angles)
        job = simulator.run(circuit, shots=repetitions)
        result = job.result()
        counts = result.get_counts()
        
        # Calcola la probabilità dello stato |0...0>
        all_zero_state = '0' * self._num_qubits
        overlap = counts.get(all_zero_state, 0) / repetitions
        
        print(f"DIP test counts: {counts}")
        print(f"DIP test overlap: {overlap}")
        if overlap < 0:
            print("--------------------------------------------------------------------L'OVERLAP DEL DIP CON CUI LAVORO VALE: ", overlap)
        return self.purity - overlap


    
    def overlap_pdip(self,
                     simulator=Aer.get_backend('aer_simulator'),
                     repetitions=1000):
        """Returns the objective function as computed by the PDIP Test."""
        # make sure the purity is computed
        if not self.purity:
            self.compute_purity()
        
        # store the overlap
        ov = 0.0
        
        for j in range(self._num_qubits):
            # do the appropriate pdip test circuit
            self.clear_dip_test_circ()
            self.pdip_test([j])
            
            # DEBUG
            print("j =", j)
            print("PDIP Test Circuit:")
            print(self.dip_test_circ)
            
            # run the circuit
            outcome = self.run(simulator, repetitions)
            
            # get the measurement counts
            dipcounts = outcome.measurements[self._measure_key]
            pdipcount = outcome.measurements[self._pdip_key]
            
            # postselect on the all zeros outcome for the dip test measuremnt
            mask = self._get_mask_for_all_zero_outcome(dipcounts)
            toprocess = pdipcount[mask]
            print("TOPPROCESS  ",toprocess)
            # do the state overlap (destructive swap test) postprocessing
            overlap = self.state_overlap_postprocessing(toprocess)
            
            # DEBUG
            #print("Overlap = ", overlap)
            
            # divide by the probability of getting the all zero outcome
            prob = len(np.where(mask == True)) / len(mask)
            counts = outcome.get_counts()
            prob = counts[0] / repetitions if 0 in counts.keys() else 0.0
            
            assert 0 <= prob <= 1
            #print("prob =", prob)
            
            
            overlap *= prob
            #print("Scaled overlap =", overlap)
            #print()
            ov += overlap

        return ov / self._num_qubits
    
    
    def overlap_pdip_resolved(self, angles, simulator=Aer.get_backend('aer_simulator'), repetitions=1000):
        #print("\n--- Inizio overlap_pdip_resolved ---")
        print("MI stai passando un numero di angoli corrispondenti a: ", len(angles))
        if not self.purity:
            self.compute_purity()
        
        total_overlap = 0.0
        
        for j in range(self._num_qubits):
            ##print(f"\nProcessing qubit {j}")
            self.clear_dip_test_circ()
            self.pdip_test([j])
            result = self.run_resolved(angles, simulator, repetitions)
            
            if result is None:
                ##print(f"Impossibile calcolare l'overlap per j={j}")
                continue
            
            try:
                memory = result.get_memory()
            except QiskitError:
                ##print("get_memory() non disponibile, usando get_counts()")
                counts = result.get_counts()
                memory = []
                for bitstring, count in counts.items():
                    memory.extend([bitstring] * count)
            
            # Separa i risultati DIP e PDIP
            dipcounts = {}
            pdipcount = {}
            for shot in memory:
                dip_result = shot[:self._num_qubits]
                pdip_result = shot[self._num_qubits:]
                dipcounts[dip_result] = dipcounts.get(dip_result, 0) + 1
                pdipcount[pdip_result] = pdipcount.get(pdip_result, 0) + 1
            
            ##print(f"DIP counts: {dipcounts}")
            ##print(f"PDIP counts: {pdipcount}")

            # Crea la maschera per i risultati all-zero DIP
            all_zero_dip = '0' * self._num_qubits
            mask = [result[:self._num_qubits] == all_zero_dip for result in memory]
            
            # Filtra i risultati PDIP basandosi sulla maschera
            toprocess = [int(shot[self._num_qubits:]) for shot, m in zip(memory, mask) if m]
            
            if not toprocess:
                ##print(f"Nessun dato da processare per j={j}")
                continue
            print("Top process:" ,toprocess)
            
            overlap = self.state_overlap_postprocessing(np.array(toprocess))
            print("CHiamo state overlap post quando lavoro in pdip res")
            if overlap < 0:
                print("L'OVERLAP DEL PDIP CON CUI LAVORO VALE: ", overlap)
            
            prob = dipcounts.get(all_zero_dip, 0) / repetitions            
            assert 0 <= prob <= 1, f"Probabilità non valida: {prob}"
            #print("prob vale: ", prob)
            ##print(f"La prob vale {prob}")

            scaled_overlap = overlap * prob
            ##print(f"j={j}, prob={prob}, overlap={overlap}, scaled_overlap={scaled_overlap}")
            
            total_overlap += scaled_overlap
        
        average_overlap = total_overlap / self._num_qubits
        ##print(f"Overlap medio calcolato: {average_overlap}")
        ##print("--- Fine overlap_pdip_resolved ---\n")
        return average_overlap

    def obj_pdip(self,
                 simulator=Aer.get_backend('aer_simulator'),
                 repetitions=10):
        """Returns the purity of the state - the overlap as computed by the
        PDIP Test.
        """
        # make sure the purity is computed
        if not self.purity:
            self.compute_purity()
        a = self.overlap_pdip(simulator, repetitions)
        #print("Purezza: ",self.purity, " meno ", a )
        return self.purity - a
    
    def obj_pdip_resolved(self, angles, simulator=Aer.get_backend('aer_simulator'), repetitions=1000):
        ##print("\n--- Inizio obj_pdip_resolved ---")
        ##print(f"Angoli ricevuti: {angles}")
        ##print(f"Numero di ripetizioni: {repetitions}")
        
        if not self.purity:
            self.compute_purity()
        
        circuit = self.algorithm()
        ##print(f"Numero di parametri nel circuito: {len(circuit.parameters)}")
        ##print(f"Numero di angoli forniti: {len(angles)}")
        
        if len(angles) != len(circuit.parameters):
            print("Attenzione: il numero di angoli non corrisponde al numero di parametri del circuito")
            print("len param ", len(circuit.parameters), " len angles ", len(angles))
        
        ##print(f"Purity calcolata: {self.purity}")

        overlap = self.overlap_pdip_resolved(angles, simulator, repetitions)
        ##print(f"Overlap calcolato: {overlap}")
        #print("-------------------------purezza", self.purity, "overlap: ", overlap)
        obj = self.purity - overlap
        ##print(f"PDIP Test obj = {obj}")
        
        ##print("--- Fine obj_pdip_resolved ---\n")
        return obj
    
    
    def _get_mask_for_all_zero_outcome(self, outcome):
        mask = []
        all_zeros = '0' * self._total_num_qubits
        ##print(f"Cercando stati che iniziano con: {all_zeros}")
        
        for state, count in outcome.items():
            ###print(f"Checking state: {state}, All zeros: {all_zeros}")
            if state[:10] == all_zeros:
                mask.extend([True] * count)
                ##print(f"Trovato stato corrispondente: {state}")
            else:
                mask.extend([False] * count)
        
        ##print(f"Generated mask: {mask}")
        return np.array(mask)


    def compute_purity(self):
        ##print("\n--- Inizio compute_purity ---")
        nShot = 10#Per prestazioni deboli
        
        state_prep_circ = self.state_prep_circ
        ##print("Generazione del circuito di sovrapposizione dello stato...")
        state_overlap_circ = self.state_overlap()
        
        ##print(f"state_prep_circ - num_qubits: {state_prep_circ.num_qubits}, num_clbits: {state_prep_circ.num_clbits}")
        ##print(f"state_overlap_circ - num_qubits: {state_overlap_circ.num_qubits}, num_clbits: {state_overlap_circ.num_clbits}")

        # Creare un nuovo circuito con il numero massimo di qubit e bit classici
        max_qubits = max(state_prep_circ.num_qubits, state_overlap_circ.num_qubits)
        max_clbits = max(state_prep_circ.num_clbits, state_overlap_circ.num_clbits)
        a = max(max_qubits,max_clbits)
        
        ##print(f"Creazione di un nuovo circuito con {a} qubit e {a} bit classici...")
        combined_circuit = QuantumCircuit(a, a)
        
        ##print("Composizione del circuito di preparazione dello stato...")
        combined_circuit.compose(state_prep_circ, qubits=range(state_prep_circ.num_qubits), 
                                clbits=range(state_prep_circ.num_clbits), inplace=True)
        
        ##print("Composizione del circuito di sovrapposizione dello stato...")
        combined_circuit.compose(state_overlap_circ, qubits=range(state_overlap_circ.num_qubits), 
                                clbits=range(state_overlap_circ.num_clbits), inplace=True)

        ##print("\nCircuito combinato creato.")
        ##print(f"Numero di qubit nel circuito combinato: {combined_circuit.num_qubits}")
        ##print(f"Numero di bit classici nel circuito combinato: {combined_circuit.num_clbits}")
        ##print(f"Numero di gate nel circuito combinato: {len(combined_circuit)}")
        #self.prep_state.##printCircuit(combined_circuit)
        ##print("\nPreparazione della simulazione...")
        # Usa il simulatore Aer
        simulator = Aer.get_backend('aer_simulator')
        # Esegui il circuito sul simulatore
        ##print("\nPronto a transpile")
        transpiled_qc = transpile(combined_circuit, simulator)
        ##print("\nPronto al job")
        job = simulator.run(transpiled_qc, shots=nShot, memory = True)
        
        ##print("Esecuzione della simulazione...")
        try:
            outcome = job.result()
            ##print("Simulazione completata.")
        except Exception as e:
            ##print(f"Errore durante la simulazione: {str(e)}")
            return None

        ##print("\nElaborazione dei risultati...")
        try:
            counts = outcome.get_counts(combined_circuit)
            ##print(f"Numero di stati misurati: {len(counts)}")
        except Exception as e:
            ##print(f"Errore nell'ottenere i conteggi: {str(e)}")
            return None

        all_zero_state = '0' * combined_circuit.num_qubits
        vals = np.array([[counts.get(state, 0) / 10 for state in ['0'*combined_circuit.num_qubits, '1'*combined_circuit.num_qubits]]])
        #print(f"Valore calcolato per vals: {vals}")

        try:
            self.purity = self.state_overlap_postprocessing(vals)
            print("CHiamo state overlap post quando lavoro con la purity")
            #print(f"Purezza calcolata: {self.purity}")
        except Exception as e:
            #print(f"Errore nel post-processing: {str(e)}---------------------------------")
            return None

        #print("\n--- Fine compute_purity ---")

    
        

        

        

    def init_state_to_matrix(self):
        """Returns the initial state defined by the state preperation
        circuit in matrix form. This corresponds to \rho in the notation
        of the VQSD paper.
        """
        # TODO: implement

    def diag_state_to_matrix(self):
        """Returns the state in matrix form after the diagonalizing unitary
        has been applied to the input state. This corresponds to \rho' in the
        notation of the VQSD paper.
        """
        # TODO: implement

    # =========================================================================
    # overrides
    # =========================================================================

    def __str__(self):
        """Returns the VQSD circuit's algorithm."""
        return self.algorithm().to_text_diagram()


    def min_to_vqsd(self, param_list, num_qubits):
        # Verifica il numero totale di elementi
        assert len(param_list) % 6 == 0, "invalid number of parameters"
        return param_list.reshape(num_qubits // 2, 4, 3)

    def vqsd_to_min(param_array):
        """Helper function that converts the array of angles in the format
        required by VQSD.layer into a linear array of angles (used to call the
        optimize function).
        """
        # TODO: add this as a member function of VQSD class
        return param_array.flatten()

    def symbol_list_for_product(num_qubits):
        """Returns a list of qiskit.circuit.Parameter's for a product state ansatz."""
        return np.array(
            [Parameter(f'theta_{ii}') for ii in range(num_qubits)]
        )
    
    def symbol_list(self,num_qubits, num_layers):
        """Returns a list of qiskit.circuit.Parameter's for a product state ansatz."""
        num_qubits = int(num_qubits) 
        return np.array(
            [Parameter(f'theta_{ii}') for ii in range(12 * (num_qubits // 2) * num_layers)]
        )

