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
        self._num_qubits = 10 #Lavoro sempre con 10 Qubit
        
        self._total_num_qubits = 2 * self._num_qubits
        self.qubits = QuantumRegister(self._total_num_qubits, name='q')

        # key for measurements and statistics
        self._measure_key = measure_key
        self._pdip_key = "p"

        # initialize the circuits into logical components
        
        self.unitary_circ = QuantumCircuit(self.qubits)
        self.dip_test_circ = QuantumCircuit(self.qubits)

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
        #print("len param: ", len(params))
    
        n = self._num_qubits

        #print("num q: ", n)
        
        for ii in range(len(params)):
            g = gate(params[ii])
            # Calcola l'indice del secondo qubit, assicurandosi che sia valido
            #print("Guarda" , ii+n)
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
            params [type: list<list<float>>]
                parameters for the first layer of gates.
                len(params) must be n // 2 where n is the number of qubits
                in the state and // indicates floor division.

                the format of params is as follows:

                params = [rotations for gates in layer]

                where the rotations for the gates in the layer have the form

                rotations for gates in layer =
                    [x1, y1, z1],
                    [x2, y2, z2],
                    [x3, y3, z3],
                    [x4, y4, z4].

                Note that each gate consists of 12 parameters. 3 parameters
                for each rotation and 4 total rotations.

                The general form for a gate, which acts on two qubits,
                is shown below:

                    ----------------------------------------------------------
                    | --Rx(x1)--Ry(y1)--Rz(z1)--@--Rx(x3)--Ry(y3)--Rz(z3)--@ |
                G = |                           |                          | |
                    | --Rx(x2)--Ry(y2)--Rz(z2)--X--Rx(x4)--Ry(y4)--Rz(z4)--X |
                    ----------------------------------------------------------

            shifted_params [type: ]
                TODO: figure this out
                parameters for the second shifted layer of gates

            copy [type: int (0 or 1)]
                the copy of the state to apply the layer to

        modifies:
            self.unitary_circ
                appends the layer of operations to self.unitary_circ
        """
        # for brevity
        n = self._num_qubits

        if params.size != self.num_angles_required_for_layer():
            raise ValueError("incorrect number of parameters for layer")

        # =====================================================================
        # helper functions for layer
        # =====================================================================

        def gate(qubits, params):
            """Helper function to append the two qubit gate
            ("G" in the VQSD paper figure).

            input:
                qubits [type: list<Qubits>]
                    qubits to be acted on. must have length 2.

                params [type: list<list<angles>>]
                    the parameters of the rotations in the gate.
                    len(params) must be equal to 12: 4 arbitrary rotations x
                    3 angles per arbitrary rotation.

                    the format of params must be

                    [[x1, y1, z1],
                     [x2, y2, z2],
                     [x3, y3, z3],
                     [x4, y4, z4]].

                    the general form of a gate, which acts on two qubits,
                    is shown below:

                    ----------------------------------------------------------
                    | --Rx(x1)--Ry(y1)--Rz(z1)--@--Rx(x3)--Ry(y3)--Rz(z3)--@ |
                G = |                           |                          | |
                    | --Rx(x2)--Ry(y2)--Rz(z2)--X--Rx(x4)--Ry(y4)--Rz(z4)--X |
                    ----------------------------------------------------------

            modifies:
                self.unitary_circ
                    appends a gate acting on the qubits to the unitary circ.
            """
            # rotation on 'top' qubit
            self.unitary_circ.append(
                self._rot(qubits[0], params[0]),
                )

            # rotation on 'bottom' qubit
            self.unitary_circ.append(
                self._rot(qubits[1], params[1]),
                )
                

            # cnot from 'top' to 'bottom' qubit
            self.unitary_circ.cx(qubits[0], qubits[1])

            # second rotation on 'top' qubit
            self.unitary_circ.append(
                self._rot(qubits[0], params[2]),
                )

            # second rotation on 'bottom' qubit
            self.unitary_circ.append(
                self._rot(qubits[1], params[3]),
                )

            # second cnot from 'top' to 'bottom' qubit
            self.unitary_circ.cx(qubits[0], qubits[1])

        # helper function for indexing loops
        stop = lambda n: n - 1 if n % 2 == 1 else n

        # shift in qubit indexing for different copies
        shift = 2 * self._num_qubits * copy

        # =====================================================================
        # implement the layer
        # =====================================================================

        # TODO: speedup. combine two loops into one

        # unshifted gates on adjacent qubit pairs
        for ii in range(0, stop(n), 2):
            iiq = ii + shift
            gate(self.qubits[iiq : iiq + 2], params[ii // 2])

        # shifted gates on adjacent qubits
        if n > 2:
            for ii in range(1, n, 2):
                iiq = ii + shift
                gate([self.qubits[iiq],
                      self.qubits[(iiq + 1) % n + shift]],
                     shifted_params[ii // 2])

    def _rot(self, qubit, params):
        """Helper function that returns an arbitrary rotation of the form
        R = Rz(params[2]) * Ry(params[1]) * Rx(params[0])
        on the qubit, e.g. R |qubit>.

        Note that order is reversed when put into the circuit. The circuit is:
        |qubit>---Rx(params[0])---Ry(params[1])---Rz(params[2])---
        """
        rx = RXGate(params[0])  # Rotazione attorno all'asse X
        ry = RYGate(params[1])  # Rotazione attorno all'asse Y
        rz = RZGate(params[2]) 

        yield (rx(qubit), ry(qubit), rz(qubit))

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
                

        # do the measurements
        qubits_to_measure = self.qubits[:self._num_qubits]
        cr = ClassicalRegister(len(qubits_to_measure), 'cr')  # Aggiungi un nome al registro
        self.dip_test_circ.add_register(cr)  # Aggiungi questa linea
        self.dip_test_circ.measure(qubits_to_measure, cr)
    
    def pdip_test(self, pdip_qubit_indices):
        """Implements the partial dip test circuit.
        
        Args:
            pdip_qubit_indices : list
                List of qubit indices (j in the paper) to do the pdip test on.
        
        Modifies:
            self.dip_test_circ
        """
        # do the cnots
        for ii in range(self._num_qubits):
            self.dip_test_circ.cx(self.qubits[ii + self._num_qubits],self.qubits[ii])
        
        # add a Hadamard on each qubit not in the PDIP test
        all_qubit_indices = set(range(self._num_qubits))
        qubit_indices_to_hadamard = list(
            all_qubit_indices - set(pdip_qubit_indices)
        )
        qubits_to_hadamard = [self.qubits[ii + self._num_qubits]
                              for ii in qubit_indices_to_hadamard]
        for qubit in qubits_to_hadamard:
            self.dip_test_circ.h(qubit)
        
        # add the measurements for the dip test
        num_measure = self._num_qubits

        # Create a new classical register

        qubits_to_measure = [self.qubits[ii] for ii in pdip_qubit_indices]
        cr = ClassicalRegister(len(qubits_to_measure), 'cr')  # Aggiungi un nome al registro
        self.dip_test_circ.add_register(cr)  # Aggiungi questa linea
        self.dip_test_circ.measure(qubits_to_measure, cr)
        
        # add the measurements for the destructive swap test on the pdip qubits
        pdip_qubits = [self.qubits[ii] for ii in qubit_indices_to_hadamard] \
                      + qubits_to_hadamard
        # edge case: no qubits in pdip set
        if len(pdip_qubits) > 0:
            cr_pdip = ClassicalRegister(len(pdip_qubits), name=self._pdip_key)
            self.dip_test_circ.add_register(cr_pdip)
            self.dip_test_circ.measure(pdip_qubits, cr_pdip)
            

    def state_overlap(self):
        """Returns the state overlap circuit as a QuantumCircuit."""
        # Determine the number of qubits to measure
        num_measure = self._num_qubits

        # Create a new classical register
        cr = ClassicalRegister(num_measure)

        # Create a new quantum circuit with the existing quantum register and the new classical register
        circuit = QuantumCircuit(self.qubits, cr)

        def bell_basis_gates(circuit, qubits, index, num_qubits):
            circuit.cx(qubits[index], qubits[index + num_qubits])  # Gate CNOT
            circuit.h(qubits[index])                                   # Gate Hadamard

        # Add the bell basis gates to the circuit
        for ii in range(self._num_qubits):
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
        print("\n--- Inizio state_overlap_postprocessing ---")
        print(f"Input ricevuto: {output}")
        output = np.array(output)
        if output.ndim == 1:
            output = output.reshape(1, -1)
        if output.shape[1] == 1:
            output = np.repeat(output, 2, axis=1)
        
        print(f"Output dopo il reshape: {output}")
        
        (nreps, nqubits) = output.shape
        print(f"nreps: {nreps}, nqubits: {nqubits}")

        assert nqubits % 2 == 0, "Input is not a valid shape."

        overlap = 0.0

        shift = nqubits // 2
        for z in output:
            parity = 1
            pairs = [z[ii] and z[ii + shift] for ii in range(shift)]
            print(f"Coppie: {pairs}")
            for pair in pairs:
                parity *= (-1)**pair
            overlap += parity
            print(f"Parity: {parity}, Current overlap: {overlap}")

        final_overlap = overlap / nreps
        print(f"Overlap finale: {final_overlap}")
        print("--- Fine state_overlap_postprocessing ---\n")
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
        combined_circuit = QuantumCircuit(maxQ, maxC)

        # Componi state_prep_circ
        combined_circuit.compose(self.state_prep_circ, qubits=range(self.state_prep_circ.num_qubits), clbits=range(self.state_prep_circ.num_clbits), inplace=True)

        # Componi state_unitary_circ
        combined_circuit.compose(self.unitary_circ, qubits=range(self.unitary_circ.num_qubits),clbits=range(self.unitary_circ.num_clbits), inplace=True)

        # Componi dip_test_circ
        combined_circuit.compose(self.dip_test_circ, qubits=range(self.dip_test_circ.num_qubits),clbits=range(self.dip_test_circ.num_clbits), inplace=True)
        #self.prep_state.printCircuit(combined_circuit)
        print(f"Parametri nel circuito combinato: {combined_circuit.parameters}")
        
        return combined_circuit

    def resolved_algorithm(self, angles):
        #print("ENTRO IN resolved_algorithm")
        circuit = self.algorithm()
        params = list(circuit.parameters)
        
        if len(angles) < len(params):
            #print(f"Attenzione: forniti {len(angles)} angoli, ma il circuito ha {len(params)} parametri.")
            # Estendi angles con valori casuali se necessario
            angles = list(angles) + [np.random.uniform(0, 2*np.pi) for _ in range(len(params) - len(angles))]
        elif len(angles) > len(params):
            #print(f"Attenzione: forniti {len(angles)} angoli, ma il circuito ha solo {len(params)} parametri.")
            angles = angles[:len(params)]
        
        param_dict = dict(zip(params, angles))
        resolved_circuit = circuit.assign_parameters(param_dict)
        
        if resolved_circuit.parameters:
            print("Attenzione: alcuni parametri non sono stati risolti:", resolved_circuit.parameters)
        
        return resolved_circuit

    def run(self,
            simulator=Aer.get_backend('aer_simulator'),
            repetitions=10):
        """Runs the algorithm and returns the result.

        rtype: cirq.TrialResult
        """
        return simulator.run(self.algorithm(), repetitions=repetitions)

    def run_resolved(self, angles, simulator=Aer.get_backend('aer_simulator'), repetitions=1000):
        print("\n--- Inizio run_resolved ---")
        print(f"Angoli ricevuti: {angles}")
        print(f"Numero di ripetizioni: {repetitions}")
        
        circuit = self.resolved_algorithm(angles)
        
        if circuit.parameters:
            print("Parametri non risolti:", circuit.parameters)
            param_binds = [{param: np.random.uniform(0, 2*np.pi) for param in circuit.parameters}]
            print(f"Binding dei parametri: {param_binds}")
            job = simulator.run(circuit, shots=repetitions, parameter_binds=param_binds)
        else:
            job = simulator.run(circuit, shots=repetitions)
        
        try:
            result = job.result()
            print("Esecuzione del job completata con successo")
            return result
        except Exception as e:
            print(f"Errore durante l'esecuzione del job: {str(e)}")
            print("Circuito:")
            print(circuit)
            print("Parametri del circuito:", circuit.parameters)
            return None
        finally:
            print("--- Fine run_resolved ---\n")

    def obj_dip(self,
                simulator=Aer.get_backend('aer_simulator'),
                repetitions=10):
        """Returns the objective function as computed by the DIP Test."""
        # make sure the purity is computed
        if not self.purity:
            self.compute_purity()

        # run the circuit
        outcome = self.run(simulator, repetitions)
        counts = outcome.get_counts()
        
        # compute the overlap and return the objective function
        overlap = counts[0] / repetitions if 0 in counts.keys() else 0
        return self.purity - overlap

    def obj_dip_resolved(self, angles, simulator=Aer.get_backend('aer_simulator'), repetitions=10):
        print("ENTRO IN obj_dip_resolved")
        if not self.purity:
            self.compute_purity()
        
        # run the circuit
        outcome = self.run_resolved(angles, simulator, repetitions)

        counts = outcome.get_counts()
        # compute the overlap and return the objective
        overlap = counts[0] / repetitions if 0 in counts.keys() else 0
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
            
            # do the state overlap (destructive swap test) postprocessing
            overlap = self.state_overlap_postprocessing(toprocess)
            
            # DEBUG
            print("Overlap = ", overlap)
            
            # divide by the probability of getting the all zero outcome
            prob = len(np.where(mask == True)) / len(mask)
            counts = outcome.get_counts()
            prob = counts[0] / repetitions if 0 in counts.keys() else 0.0
            
            assert 0 <= prob <= 1
            print("prob =", prob)
            
            
            overlap *= prob
            print("Scaled overlap =", overlap)
            print()
            ov += overlap

        return ov / self._num_qubits
    
    def overlap_pdip_resolved(self, angles, simulator=Aer.get_backend('aer_simulator'), repetitions=1000):
        print("\n--- Inizio overlap_pdip_resolved ---")
        print(f"Angoli ricevuti: {angles}")
        print(f"Numero di ripetizioni: {repetitions}")
        
        if not self.purity:
            self.compute_purity()
        
        total_overlap = 0.0
        
        for j in range(self._num_qubits):
            print(f"\nProcessing qubit {j}")
            self.clear_dip_test_circ()
            self.pdip_test([j])
            result = self.run_resolved(angles, simulator, repetitions)
            if result is None:
                print(f"Impossibile calcolare l'overlap per j={j}")
                continue
            
            counts = result.get_counts()
            print(f"Conteggi per qubit {j}: {counts}")
            
            dipcounts = {k.split()[0]: v for k, v in counts.items()}
            pdipcount = {k.split()[1]: v for k, v in counts.items() if len(k.split()) > 1}
            
            print(f"DIP counts: {dipcounts}")
            print(f"PDIP counts: {pdipcount}")

            mask = self._get_mask_for_all_zero_outcome(dipcounts)
            toprocess = [pdipcount[k] for k in pdipcount.keys() if mask[k]]
            
            print(f"Mask: {mask}")
            print(f"To process: {toprocess}")
            
            if not toprocess:
                print(f"Nessun dato da processare per j={j}")
                continue
            
            overlap = self.state_overlap_postprocessing(np.array(toprocess))
            
            prob = dipcounts.get('0' * self._num_qubits, 0) / repetitions            
            assert 0 <= prob <= 1, f"Probabilità non valida: {prob}"
            
            scaled_overlap = overlap * prob
            print(f"j={j}, prob={prob}, overlap={overlap}, scaled_overlap={scaled_overlap}")
            
            total_overlap += scaled_overlap
        
        average_overlap = total_overlap / self._num_qubits
        print(f"Overlap medio calcolato: {average_overlap}")
        print("--- Fine overlap_pdip_resolved ---\n")
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

        return self.purity - self.overlap_pdip(simulator, repetitions)
    
    def obj_pdip_resolved(self, angles, simulator=Aer.get_backend('aer_simulator'), repetitions=1000):
        print("\n--- Inizio obj_pdip_resolved ---")
        print(f"Angoli ricevuti: {angles}")
        print(f"Numero di ripetizioni: {repetitions}")
        
        if not self.purity:
            self.compute_purity()
        
        circuit = self.algorithm()
        print(f"Numero di parametri nel circuito: {len(circuit.parameters)}")
        print(f"Numero di angoli forniti: {len(angles)}")
        
        if len(angles) != len(circuit.parameters):
            print("Attenzione: il numero di angoli non corrisponde al numero di parametri del circuito")
        
        print(f"Purity calcolata: {self.purity}")

        overlap = self.overlap_pdip_resolved(angles, simulator, repetitions)
        print(f"Overlap calcolato: {overlap}")
        
        obj = self.purity - overlap
        print(f"PDIP Test obj = {obj}")
        
        print("--- Fine obj_pdip_resolved ---\n")
        return obj
    
    
    def _get_mask_for_all_zero_outcome(self, outcome):
        mask = []
        all_zeros = '0' * 10
        print(f"Cercando stati che iniziano con: {all_zeros}")
        
        for state, count in outcome.items():
            print(f"Checking state: {state}, All zeros: {all_zeros}")
            if state[:10] == all_zeros:
                mask.extend([True] * count)
                print(f"Trovato stato corrispondente: {state}")
            else:
                mask.extend([False] * count)
        
        print(f"Generated mask: {mask}")
        return np.array(mask)


    def compute_purity(self):
        print("\n--- Inizio compute_purity ---")
        nShot = 10#Per prestazioni deboli
        
        state_prep_circ = self.state_prep_circ
        print("Generazione del circuito di sovrapposizione dello stato...")
        state_overlap_circ = self.state_overlap()
        
        print(f"state_prep_circ - num_qubits: {state_prep_circ.num_qubits}, num_clbits: {state_prep_circ.num_clbits}")
        print(f"state_overlap_circ - num_qubits: {state_overlap_circ.num_qubits}, num_clbits: {state_overlap_circ.num_clbits}")

        # Creare un nuovo circuito con il numero massimo di qubit e bit classici
        max_qubits = max(state_prep_circ.num_qubits, state_overlap_circ.num_qubits)
        max_clbits = max(state_prep_circ.num_clbits, state_overlap_circ.num_clbits)
        
        print(f"Creazione di un nuovo circuito con {max_qubits} qubit e {max_clbits} bit classici...")
        combined_circuit = QuantumCircuit(max_qubits, max_clbits)
        
        print("Composizione del circuito di preparazione dello stato...")
        combined_circuit.compose(state_prep_circ, qubits=range(state_prep_circ.num_qubits), 
                                clbits=range(state_prep_circ.num_clbits), inplace=True)
        
        print("Composizione del circuito di sovrapposizione dello stato...")
        combined_circuit.compose(state_overlap_circ, qubits=range(state_overlap_circ.num_qubits), 
                                clbits=range(state_overlap_circ.num_clbits), inplace=True)

        print("\nCircuito combinato creato.")
        print(f"Numero di qubit nel circuito combinato: {combined_circuit.num_qubits}")
        print(f"Numero di bit classici nel circuito combinato: {combined_circuit.num_clbits}")
        print(f"Numero di gate nel circuito combinato: {len(combined_circuit)}")
        #self.prep_state.printCircuit(combined_circuit)
        print("\nPreparazione della simulazione...")
        # Usa il simulatore Aer
        simulator = Aer.get_backend('aer_simulator')
        # Esegui il circuito sul simulatore
        print("\nPronto a transpile")
        transpiled_qc = transpile(combined_circuit, simulator)
        print("\nPronto al job")
        job = simulator.run(transpiled_qc, shots=nShot)
        
        print("Esecuzione della simulazione...")
        try:
            outcome = job.result()
            print("Simulazione completata.")
        except Exception as e:
            print(f"Errore durante la simulazione: {str(e)}")
            return None

        print("\nElaborazione dei risultati...")
        try:
            counts = outcome.get_counts(combined_circuit)
            print(f"Numero di stati misurati: {len(counts)}")
        except Exception as e:
            print(f"Errore nell'ottenere i conteggi: {str(e)}")
            return None

        all_zero_state = '0' * combined_circuit.num_qubits
        vals = np.array([[counts.get(state, 0) / 10 for state in ['0'*combined_circuit.num_qubits, '1'*combined_circuit.num_qubits]]])
        print(f"Valore calcolato per vals: {vals}")

        try:
            self.purity = self.state_overlap_postprocessing(vals)
            print(f"Purezza calcolata: {self.purity}")
        except Exception as e:
            print(f"Errore nel post-processing: {str(e)}---------------------------------")
            return None

        print("\n--- Fine compute_purity ---")

    
        

        

        

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


def min_to_vqsd(param_list, num_qubits=2):
    """Helper function that converts a linear array of angles (used to call
    the optimize function) into the format required by VQSD.layer.
    """
    # TODO: add this as a member function of VQSD class
    # error check on input
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

