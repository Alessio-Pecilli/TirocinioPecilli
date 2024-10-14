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
from qiskit import QuantumCircuit
from qiskit.circuit import Instruction, Parameter
from qiskit.circuit.quantumcircuitdata import QuantumCircuitData
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
        #self.prep_state.#printCircuit(self.unitary_circ)
        #self.prep_state.#printCircuit(self.dip_test_circ)

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
        ##print("A DIP DO CL: ", self.dip_test_circ.num_clbits)

    # =========================================================================
    # circuit methods
    # =========================================================================      

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

        shift =  self._num_qubits * copy

        for ii in range(0, stop(n), 2):
            iiq = ii + shift
            self._apply_gate(self.qubits[iiq : iiq + 2], params[ii // 2])

        if n > 2:
            for ii in range(1, n, 2):
                iiq = ii + shift
                self._apply_gate([self.qubits[iiq],
                      self.qubits[(iiq + 1) % n + shift]],
                     shifted_params[ii // 2])        #self.prep_state.#printCircuit(self.unitary_circ)

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
        cr = ClassicalRegister(len(qubits_to_measure), name=self._measure_key)
        self.dip_test_circ.add_register(cr)
        self.dip_test_circ.measure(qubits_to_measure, cr)


        ###print(f"Misurazione aggiunta per i qubit: {qubits_to_measure}")
    
    def pdip_test(self, pdip_qubit_indices): 
        ##print("pdip_qubit_indices ", pdip_qubit_indices)
         
        # Aggiungi i CNOT (cx in Qiskit) per ciascun qubit
        for ii in range(self._num_qubits):
            control_qubit = ii +  self._num_qubits
            target_qubit = ii
            self.dip_test_circ.cx(control_qubit, target_qubit)
        
        # Aggiungi Hadamard ai qubit non coinvolti nel test PDIP
        all_qubit_indices = set(range(self._num_qubits))
        qubit_indices_to_hadamard = list(
            all_qubit_indices - set(pdip_qubit_indices)
        )
        qubits_to_hadamard = [self.qubits[ii + self._num_qubits]
                              for ii in qubit_indices_to_hadamard]
        self.dip_test_circ.h(qubits_to_hadamard)
                
                # Misura i qubit coinvolti nel test PDIP
        qubits_to_measure = [ii for ii in pdip_qubit_indices]
        cr = ClassicalRegister(len(qubits_to_measure), name=self._measure_key)
        self.dip_test_circ.add_register(cr)

        # Effettuiamo la misura utilizzando il ClassicalRegister appena creato
        self.dip_test_circ.measure(qubits_to_measure, cr)
        print("EEEEEEEEEEEE")

        # add the measurements for the destructive swap test on the pdip qubits
        pdip_qubits = [self.qubits[ii] for ii in qubit_indices_to_hadamard] \
                      + qubits_to_hadamard

        # Aggiungi le misure per il test di swap distruttivo sui qubit PDIP
        pdip_qubits = [ii for ii in qubit_indices_to_hadamard] + qubits_to_measure
        if len(pdip_qubits) > 0:
            cr = ClassicalRegister(len(pdip_qubits), name=self._pdip_key)
            self.dip_test_circ.add_register(cr)
            # Effettuiamo la misura utilizzando il ClassicalRegister appena creato
            self.dip_test_circ.measure(qubits_to_measure, cr) 
            
            

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
        ## Stampa l'input iniziale
        #print(f"Input ricevuto: {output}")
        
        # Converti l'input in un array numpy se non lo è già
        output = np.array(output)
        
        # Se l'input è una stringa, convertila in un array di interi
        if output.dtype.kind == 'U':
            output = np.array([[int(bit) for bit in row] for row in output])
        
        # Se l'input è un array monodimensionale, ridimensionalo a (1, -1)
        if output.ndim == 1:
            output = output.reshape(1, -1)
        
        ## Stampa l'output dopo il reshape
        #print(f"Output dopo il reshape: {output}")
        
        (nreps, nqubits) = output.shape
        ## Stampa il numero di repliche e qubits
        #print(f"nreps: {nreps}, nqubits: {nqubits}")

        # Se il numero di qubits è dispari, aggiungi una colonna di zeri
        if nqubits % 2 != 0:
            output = np.pad(output, ((0, 0), (0, 1)), mode='constant')
            nqubits += 1
            ## Stampa che è stato aggiunto uno zero e il nuovo numero di qubits
            print(f"Numero dispari di qubits. Aggiunta una colonna di zeri. Nuovo nqubits: {nqubits}")
        
        ## Stampa l'output finale prima del calcolo dell'overlap
        #print("GLI OUTPUT CON CUI LAVORO SONO: ", output)
        
        overlap = 0.0
        shift = nqubits // 2
        
        # Calcola l'overlap
        for z in output:
            parity = 1
            pairs = [(1 if z[ii] and z[ii + shift] else 0) for ii in range(shift)]

            #print(f"Coppie per la riga {z}: {pairs}")
            for pair in pairs:
                #print(f"PAIR: {pair}, Parity prima del calcolo: {parity}")
                parity *= (-1)**pair
                #print(f"Parity dopo il calcolo: {parity}")
            overlap += parity
            #print(f"Parity per la riga {z}: {parity}, Current overlap: {overlap}")
        
        # Se l'overlap è negativo, stampalo
        #if overlap < 0:
            #print("STATE PROCESSING, L'OVERLAP DEL PDIP CON CUI LAVORO VALE: ", overlap)
        
        final_overlap = overlap / nreps
        ## Stampa l'overlap finale
        #print("STATE PROCESSING, L'OVERLAP FINALE DEL PDIP: ", final_overlap)
        
        ## Stampa la fine della funzione
        #print("--- Fine state_overlap_postprocessing ---\n")
        
        return final_overlap




    # =========================================================================
    # methods for running the circuit and getting the objective function
    # =========================================================================

    def algorithm(self):
        ##print("ENTRO IN algorithm")
        """Returns the total algorithm of the VQSD circuit, which consists of
        state preperation, diagonalizing unitary, and dip test.

        rtype: QuantumCircuit
        """
        combined_circuit = QuantumCircuit(self._total_num_qubits)

        # Componi state_prep_circ
        combined_circuit.compose(self.state_prep_circ, qubits=range(self.state_prep_circ.num_qubits), clbits=range(self.state_prep_circ.num_clbits), inplace=True)

        # Componi state_unitary_circ
        combined_circuit.compose(self.unitary_circ, qubits=range(self.unitary_circ.num_qubits),clbits=range(self.unitary_circ.num_clbits), inplace=True)
        #print("STAMPO IL CIRCUITO UNITARIO")
        #self.examine_circuit(self.unitary_circ)
        # Componi dip_test_circ
        combined_circuit.compose(self.dip_test_circ, qubits=range(self.dip_test_circ.num_qubits),clbits=range(self.dip_test_circ.num_clbits), inplace=True)
        print("STAMPO IL DIP TEST CIRC")
        #self.examine_circuit(self.dip_test_circ)
        #self.prep_state.printCircuit(combined_circuit)
        ##print(f"Parametri nel circuito combinato: {combined_circuit.parameters}")
        ##print(f"combined_circuit - num_qubits: {combined_circuit.num_qubits}, num_clbits: {combined_circuit.num_clbits}")
        
        return combined_circuit

    def resolved_algorithm(self, angles):
        print("ENTRO IN resolved_algorithm")
        circuit = self.algorithm()
        
        num_params = 6 * self._num_qubits
        print("QUI PARAM: invece gli angoli sono: ",num_params, " - ",len(angles))

        if angles is None:
            # Genera angoli casuali
            angles = 2 * np.random.rand(num_params)

        # Verifica che il numero di angoli corrisponda al numero di parametri
        if len(angles) != num_params:
            raise ValueError(f"Expected {num_params} angles, but got {len(angles)}")

        # Crea una lista di Parameter objects
        params = [Parameter(f'θ_{i}') for i in range(num_params)]
        for i, param in enumerate(params):
            circuit.rx(param, i % self._num_qubits)

        
        # Crea un dizionario che mappa i Parameter objects ai valori degli angoli
        param_dict = self.create_aer_parameters(params, angles)
        #print("Parametri presenti nel circuito prima dell'assegnazione:", circuit.parameters)


        resolved_circuit = circuit.assign_parameters(param_dict)
        #print("Circuit after assigning parameters:", resolved_circuit)

        
        return resolved_circuit

    def create_aer_parameters(self, params, angles):
        # Crea un dizionario che mappa ogni parametro a un angolo corrispondente
        param_dict = {param: float(angle) for param, angle in zip(params, angles)}
        return param_dict


    
    
    def examine_circuit(self,circuit):
        """
        Esamina in dettaglio ogni aspetto di un circuito quantistico Qiskit.
        
        Parametri:
            circuit (QuantumCircuit): Il circuito quantistico da esaminare.
            
        Stampa:
            - Numero di qubit e bit classici.
            - Parametri liberi e il loro stato.
            - Ogni istruzione del circuito con i dettagli su operazioni e parametri.
            - Informazioni su eventuali porte non parametrizzate.
            - Stato finale del circuito.
        """
        print("\n--- DETTAGLI GENERALI DEL CIRCUITO ---")
        print(f"Numero di qubit: {circuit.num_qubits}")
        print(f"Numero di bit classici: {circuit.num_clbits}")
        print(f"Numero totale di porte: {len(circuit.data)}")
        
        print("\n--- PARAMETRI DEL CIRCUITO ---")
        if circuit.parameters:
            print(f"Parametri liberi: {circuit.parameters}")
            for param in circuit.parameters:
                print(f"  - {param}: nome = {param.name}, valore attuale = {param._symbol_expr}")
        else:
            print("Nessun parametro libero.")
        
        print("\n--- ISTRUZIONI DEL CIRCUITO ---")
        """
        for i, instruction in enumerate(circuit.data):
            gate, qubits, clbits = instruction
            print(f"Istruzione {i+1}:")
            print(f"  Porta: {gate.name}")
            print(f"  Qubits coinvolti: {[qubit._index for qubit in qubits]}")
            if clbits:
                print(f"  Bit classici coinvolti: {[clbit._index for clbit in clbits]}")
            if isinstance(gate, Instruction) and hasattr(gate, 'params'):
                if gate.params:
                    print(f"  Parametri della porta: {gate.params}")
                else:
                    print("  Questa porta non ha parametri.")
        
        print("\n--- CONDIZIONI CLASSICHE ---")
        if circuit.cregs:
            #for creg in circuit.cregs:
                #print(f"Registri classici: {creg.name}, dimensione: {creg.size}")
        else:
            print("Nessun registro classico.")
        """
        print("\n--- STATO DEL CIRCUITO ---")
        print(f"Numero di parametri liberi nel circuito: {len(circuit.parameters)}")
        print(f"Profondità del circuito: {circuit.depth()}")
        print(f"Numero di operazioni nel circuito: {circuit.size()}")
        print(f"Fattore di larghezza del circuito: {circuit.width()}")
        print(f"Numero di parametri non risolti: {len(circuit.parameters)}")

        print("\n--- CIRCUITO COMPLETO ---")
        #print(circuit.draw(output='text'))
        self.prep_state.printCircuit(circuit)

    def run(self,
            simulator=Aer.get_backend('aer_simulator'),
            repetitions=10):
        """Runs the algorithm and returns the result.

        rtype: cirq.TrialResult
        """
        return simulator.run(self.algorithm(), repetitions=repetitions, memory = True)

    def run_resolved(self, angles, simulator=Aer.get_backend('aer_simulator'), repetitions=1000):
        ##print("\n--- Inizio run_resolved ---")
        ##print(f"Angoli ricevuti: {angles}")
        ##print(f"Numero di ripetizioni: {repetitions}")
        
        circuit = self.resolved_algorithm(angles)
        print(circuit.parameters)
        if circuit.parameters:
            ##print("Parametri non risolti:", circuit.parameters)
            param_binds = [{param: np.random.uniform(0, 2*np.pi) for param in circuit.parameters}]
            ##print(f"Binding dei parametri: {param_binds}")
            job = simulator.run(circuit, shots=repetitions, parameter_binds=param_binds, memory = True)
        else:
            job = simulator.run(circuit, shots=repetitions, memory = True)
        print("result")
        try:
            result = job.result()
            ##print("Esecuzione del job completata con successo")
            return result
        except Exception as e:
            print("Exception")
            return None
        #finally:
            ##print("--- Fine run_resolved ---\n")

    def obj_dip_resolved(self, angles, simulator=Aer.get_backend('aer_simulator'), repetitions=1000):
        print("BBBBBBBBBBBBBBBBBBBBBB")
        if not self.purity:
            self.compute_purity()

        outcome = self.run_resolved(angles, simulator, repetitions)
        print("OUTCOME: ", outcome)
        counts = outcome.histogram(key=self._measure_key)

        overlap = counts[0] / repetitions if 0 in counts.keys() else 0
        return self.purity - overlap


    
    def overlap_pdip(self,
                     simulator=Aer.get_backend('aer_simulator'),
                     repetitions=1000):
        """Returns the objective function as computed by the PDIP Test."""
        print("AAAAAAAAAAAAAAAAAAAAA")
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
            #print("j =", j)
            #print("PDIP Test Circuit:")
            #print(self.dip_test_circ)
            
            # run the circuit
            outcome = self.run(simulator, repetitions)
            
            # get the measurement counts
            dipcounts = outcome.measurements[self._measure_key]
            pdipcount = outcome.measurements[self._pdip_key]
            
            # postselect on the all zeros outcome for the dip test measuremnt
            mask = self._get_mask_for_all_zero_outcome(dipcounts)
            toprocess = pdipcount[mask]
            #print("TOPPROCESS DEVO LAVORARE QUI SOPRA",toprocess)
            #print("TIPOOOOOOOOOOOOOOOOOO" ,type(toprocess))  # Verifica se è una lista o altro tipo
            #print(type(toprocess[0])) 
            # do the state overlap (destructive swap test) postprocessing
            overlap = self.state_overlap_postprocessing(toprocess)
            
            # DEBUG
            ##print("Overlap = ", overlap)
            
            # divide by the probability of getting the all zero outcome
            prob = len(np.where(mask == True)) / len(mask)
            counts = outcome.get_counts()
            prob = counts[0] / repetitions if 0 in counts.keys() else 0.0
            
            assert 0 <= prob <= 1
            ##print("prob =", prob)
            
            
            overlap *= prob
            ##print("Scaled overlap =", overlap)
            ##print()
            ov += overlap

        return ov / self._num_qubits
    
    
    def overlap_pdip_resolved(self, angles, simulator=Aer.get_backend('aer_simulator'), repetitions=1000):
        ##print("\n--- Inizio overlap_pdip_resolved ---")
        #print("MI stai passando un numero di angoli corrispondenti a: ", len(angles))
        print("DDDDDDDDDDDDDDDDDDDDDD")
        if not self.purity:
            self.compute_purity()

        total_overlap = 0.0
        
        for j in range(self._num_qubits):
            ###print(f"\nProcessing qubit {j}")
            self.clear_dip_test_circ()
            self.pdip_test([j])
            print("qui pdip test in overPdipRes")
            result = self.run_resolved(angles, simulator, repetitions)
            
            if result is None:
                ###print(f"Impossibile calcolare l'overlap per j={j}")
                continue
            
            try:
                memory = result.get_memory()
            except QiskitError:
                ###print("get_memory() non disponibile, usando get_counts()")
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
            
            ###print(f"DIP counts: {dipcounts}")
            ###print(f"PDIP counts: {pdipcount}")

            # Crea la maschera per i risultati all-zero DIP
            all_zero_dip = '0' * self._num_qubits
            mask = [result[:self._num_qubits] == all_zero_dip for result in memory]
            
            # Filtra i risultati PDIP basandosi sulla maschera
            toprocess = [int(shot[self._num_qubits:]) for shot, m in zip(memory, mask) if m]
            
            if not toprocess:
                ###print(f"Nessun dato da processare per j={j}")
                continue
            #print("Top process: PDIP" ,toprocess)
            #print("TIPOOOOOOOOOOOOOOOOOO" ,type(toprocess))  # Verifica se è una lista o altro tipo
            #print(type(toprocess[0])) 
            max_length = max(len(bin(x)[2:]) for x in toprocess)
    
            # Converti ogni numero in una stringa binaria e riempi con zeri a sinistra
            output = [bin(x)[2:].zfill(max_length) for x in toprocess]
    
            # Converti le stringhe binarie in un array NumPy di interi
            output_array = np.array([[int(bit) for bit in row] for row in output])
            overlap = self.state_overlap_postprocessing(output_array)
            #print("CHiamo state overlap post quando lavoro in pdip res")
            #if overlap < 0:
                #print("L'OVERLAP DEL PDIP CON CUI LAVORO VALE: ", overlap)
            
            prob = dipcounts.get(all_zero_dip, 0) / repetitions            
            assert 0 <= prob <= 1, f"Probabilità non valida: {prob}"
            ##print("prob vale: ", prob)
            ###print(f"La prob vale {prob}")

            scaled_overlap = overlap * prob
            ###print(f"j={j}, prob={prob}, overlap={overlap}, scaled_overlap={scaled_overlap}")
            
            total_overlap += scaled_overlap
        
        average_overlap = total_overlap / self._num_qubits
        ###print(f"Overlap medio calcolato: {average_overlap}")
        ###print("--- Fine overlap_pdip_resolved ---\n")
        return average_overlap
    
    def obj_pdip_resolved(self, angles, simulator=Aer.get_backend('aer_simulator'), repetitions=1000):
        
        if not self.purity:
            self.compute_purity()

        overlap = self.overlap_pdip_resolved(angles, simulator, repetitions)

        return self.purity - overlap
    
    
    def _get_mask_for_all_zero_outcome(self, outcome):
        mask = []
        all_zeros = '0' * self._total_num_qubits
        ###print(f"Cercando stati che iniziano con: {all_zeros}")
        
        for state, count in outcome.items():
            ####print(f"Checking state: {state}, All zeros: {all_zeros}")
            if state[:10] == all_zeros:
                mask.extend([True] * count)
                ###print(f"Trovato stato corrispondente: {state}")
            else:
                mask.extend([False] * count)
        
        ###print(f"Generated mask: {mask}")
        return np.array(mask)


    def compute_purity(self):
        ###print("\n--- Inizio compute_purity ---")
        nShot = 10#Per prestazioni deboli
        
        state_prep_circ = self.state_prep_circ
        ###print("Generazione del circuito di sovrapposizione dello stato...")
        state_overlap_circ = self.state_overlap()
        
        ###print(f"state_prep_circ - num_qubits: {state_prep_circ.num_qubits}, num_clbits: {state_prep_circ.num_clbits}")
        ###print(f"state_overlap_circ - num_qubits: {state_overlap_circ.num_qubits}, num_clbits: {state_overlap_circ.num_clbits}")

        # Creare un nuovo circuito con il numero massimo di qubit e bit classici
        max_qubits = max(state_prep_circ.num_qubits, state_overlap_circ.num_qubits)
        max_clbits = max(state_prep_circ.num_clbits, state_overlap_circ.num_clbits)
        a = max(max_qubits,max_clbits)
        
        ###print(f"Creazione di un nuovo circuito con {a} qubit e {a} bit classici...")
        combined_circuit = QuantumCircuit(a, a)
        
        ###print("Composizione del circuito di preparazione dello stato...")
        combined_circuit.compose(state_prep_circ, qubits=range(state_prep_circ.num_qubits), 
                                clbits=range(state_prep_circ.num_clbits), inplace=True)
        
        ###print("Composizione del circuito di sovrapposizione dello stato...")
        combined_circuit.compose(state_overlap_circ, qubits=range(state_overlap_circ.num_qubits), 
                                clbits=range(state_overlap_circ.num_clbits), inplace=True)

        ###print("\nCircuito combinato creato.")
        ###print(f"Numero di qubit nel circuito combinato: {combined_circuit.num_qubits}")
        ###print(f"Numero di bit classici nel circuito combinato: {combined_circuit.num_clbits}")
        ###print(f"Numero di gate nel circuito combinato: {len(combined_circuit)}")
        #self.prep_state.###printCircuit(combined_circuit)
        ###print("\nPreparazione della simulazione...")
        # Usa il simulatore Aer
        simulator = Aer.get_backend('aer_simulator')
        # Esegui il circuito sul simulatore
        ###print("\nPronto a transpile")
        transpiled_qc = transpile(combined_circuit, simulator)
        ###print("\nPronto al job")
        job = simulator.run(transpiled_qc, shots=nShot, memory = True)
        
        ###print("Esecuzione della simulazione...")
        try:
            outcome = job.result()
            ###print("Simulazione completata.")
        except Exception as e:
            ###print(f"Errore durante la simulazione: {str(e)}")
            return None

        ###print("\nElaborazione dei risultati...")
        try:
            counts = outcome.get_counts(combined_circuit)
            ###print(f"Numero di stati misurati: {len(counts)}")
        except Exception as e:
            ###print(f"Errore nell'ottenere i conteggi: {str(e)}")
            return None

        all_zero_state = '0' * combined_circuit.num_qubits
        vals = np.array([[counts.get(state, 0) / 10 for state in ['0'*combined_circuit.num_qubits, '1'*combined_circuit.num_qubits]]])
        ##print(f"Valore calcolato per vals: {vals}")

        try:
            self.purity = self.state_overlap_postprocessing(vals)
            #print("CHiamo state overlap post quando lavoro con la purity")
            ##print(f"Purezza calcolata: {self.purity}")
        except Exception as e:
            ##print(f"Errore nel post-processing: {str(e)}---------------------------------")
            return None

        ##print("\n--- Fine compute_purity ---")

    
        

        

        

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
