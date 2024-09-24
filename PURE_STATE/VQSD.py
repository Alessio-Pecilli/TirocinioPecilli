"""VQSD.py

Contains a class for VQSD circuits utilizing Qiskit.
"""

# =============================================================================
# imports
# =============================================================================

import os
from tkinter import Image
import cirq
from qiskit.visualization import circuit_drawer
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

class PID_PDIP:
    # =========================================================================
    # init method
    # =========================================================================

    def __init__(self, measure_key='z'):

        # key for measurements and statistics
        self._measure_key = measure_key
        self._pdip_key = "p"

        # initialize the circuits into logical components
        
        self.dip_test_circ = QuantumCircuit(self.qubits)
        #self.prep_state.#printCircuit(self.unitary_circ)
        #self.prep_state.#printCircuit(self.dip_test_circ)

        # initialize the purity of the state
        self.purity = None

    
    def clear_dip_test_circ(self):
        """Sets the dip test circuit to be a new, empty circuit."""
        self.dip_test_circ = QuantumCircuit(self.qubits)
        ##print("A DIP DO CL: ", self.dip_test_circ.num_clbits)

    # =========================================================================
    # circuit methods
    # =========================================================================      

    def dip_test(self):

        for ii in range(self._num_qubits):
            self.dip_test_circ.cx( self.qubits[ii + self._num_qubits],self.qubits[ii])
            
        
        qubits_to_measure = self.qubits[:self._num_qubits]
        cr = ClassicalRegister(len(qubits_to_measure), name=self._measure_key)
        self.dip_test_circ.add_register(cr)
        self.dip_test_circ.measure(qubits_to_measure, cr)


    
    
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
        print("STAMPO IL CIRCUITO UNITARIO")
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
        print("QUESTO RISOLTO HA param: ",resolved_circuit.parameters)
        
        #if resolved_circuit.parameters:
            #print("Attenzione: alcuni parametri non sono stati risolti:", resolved_circuit.parameters)
        
        return resolved_circuit
    
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
        
        if circuit.parameters:
            ##print("Parametri non risolti:", circuit.parameters)
            param_binds = [{param: np.random.uniform(0, 2*np.pi) for param in circuit.parameters}]
            ##print(f"Binding dei parametri: {param_binds}")
            job = simulator.run(circuit, shots=repetitions, parameter_binds=param_binds, memory = True)
        else:
            job = simulator.run(circuit, shots=repetitions, memory = True)
        
        try:
            result = job.result()
            ##print("Esecuzione del job completata con successo")
            return result
        except Exception as e:
            ##print(f"Errore durante l'esecuzione del job: {str(e)}")
            ##print("Circuito:")
            ##print(circuit)
            ##print("Parametri del circuito:", circuit.parameters)
            return None
        #finally:
            ##print("--- Fine run_resolved ---\n")

    def obj_dip_resolved(self, angles, simulator=Aer.get_backend('aer_simulator'), repetitions=1000):
        print("BBBBBBBBBBBBBBBBBBBBBB")
        if not self.purity:
            self.compute_purity()

        
        
        circuit = self.resolved_algorithm(angles)
        param_dict = dict(zip(circuit.parameters, angles))

        # Trasforma il circuito per l'ottimizzazione
        print("Parametri originali:", circuit.parameters)

        transpiled_circuit = transpile(circuit, simulator)

        job = simulator.run(transpiled_circuit, shots=repetitions, parameter_binds=[param_dict])
        result = job.result()
        counts = result.get_counts()
        
        # Calcola la probabilità dello stato |0...0>
        all_zero_state = '0' * self._num_qubits
        overlap = counts.get(all_zero_state, 0) / repetitions
        
        #print(f"DIP test counts: {counts}")
        #print(f"DIP test overlap: {overlap}")
        #if overlap < 0:
        print("--------------------------------------------------------------------L'OVERLAP DEL DIP CON CUI LAVORO VALE: ", overlap)
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
            print("Overlap = ", overlap)
            
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
    
    def printCircuit(self, circuit):
        current_dir = os.path.dirname(os.path.realpath(__file__))        
        # Salva il circuito come immagine
        image_path = os.path.join(current_dir, 'PrepStatePassato.png')
        circuit_drawer(circuit, output='mpl', filename=image_path)
        
        # Apri automaticamente l'immagine
        img = Image.open(image_path)
        img.show()


    