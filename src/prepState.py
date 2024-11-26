import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random
from sklearn import datasets
from qiskit.quantum_info import Statevector
import os
import math
from qiskit import transpile
from qiskit.visualization import circuit_drawer
import qiskit_aer
from qiskit_aer import Aer
from qiskit_aer import AerSimulator
from qiskit import QuantumCircuit
from qiskit.visualization import plot_histogram

class StatePreparation:
    def __init__(self, num_img):
        self.num_img = num_img
        self.index = 0

    @property
    def num_qubits(self):
        return 8

    def create_density_matrix(self, state_vectors):
        n = len(state_vectors[0].data)  # Dimensione del vettore di stato
        rho = np.zeros((n, n), dtype=np.complex128)  # Matrice densità inizialmente vuota

        for statevector in state_vectors:
            vector = statevector.data  # Ottieni i dati del vettore come array di numpy
            # Stato puro come matrice densità (prodotto esterno)
            state = np.outer(vector, np.conjugate(vector))
            rho += state

        # Normalizziamo la matrice densità
        rho /= len(state_vectors)

        return rho

    def FromFileToStateVector(self, file_path):
        #processor = readMINST.MINSTImageProcessor()
        #binary_array = processor.produceArray(file_path, random.randint(1, 10000))
        binary_array = None #Se voglio gli 8x8 del database
        return self.normalizzazione64(binary_array)

    def statePrep(self, quantum_state):
        
        qc = QuantumCircuit(quantum_state.num_qubits*2)
        # Inizializza il circuito con lo stato fornito
        qc.initialize(quantum_state, range(quantum_state.num_qubits))

        qc.initialize(quantum_state, range(quantum_state.num_qubits, quantum_state.num_qubits*2))
        # Aggiungi operazioni di misura
        #qc.measure(range(self.num_qubits), range(self.num_qubits))

        # Usa il simulatore Aer
        #simulator = Aer.get_backend('aer_simulator')
        # Trasponi il circuito per adattarlo al backend
        #transpiled_qc = transpile(qc, simulator)
        
        #return transpiled_qc
        return qc
    
    def statePrepSingle(self, quantum_state):
        
        qc = QuantumCircuit(quantum_state.num_qubits)
        # Inizializza il circuito con lo stato fornito
        qc.initialize(quantum_state, range(quantum_state.num_qubits))

        #self.printCircuit(qc)
        return qc

    def measure_statevector(self, quantum_state):
        # Crea un circuito quantistico con il numero di qubit corrispondente
        qc = QuantumCircuit(self.num_qubits, self.num_qubits)
        # Inizializza il circuito con lo stato fornito
        qc.initialize(quantum_state, range(self.num_qubits))
        # Aggiungi operazioni di misura
        qc.measure(range(self.num_qubits), range(self.num_qubits))

        # Usa il simulatore Aer
        simulator = Aer.get_backend('aer_simulator')
        # Trasponi il circuito per adattarlo al backend
        transpiled_qc = transpile(qc, simulator)
        # Esegui il circuito sul simulatore
        result = simulator.run(transpiled_qc, shots=1000).result()
        # Ottieni i risultati
        
        counts = result.get_counts(transpiled_qc)

        print("Risultati della misura:", counts)
        plot_histogram(counts)
        plt.show()

    def measure_qc(self, qc):
        #print(qc)
        # Usa il simulatore Aer
        simulator = Aer.get_backend('aer_simulator')
        # Trasponi il circuito per adattarlo al backend
        transpiled_qc = transpile(qc, simulator)
        # Esegui il circuito sul simulatore
        result = simulator.run(transpiled_qc, shots=1000).result()
        # Ottieni i risultati
        counts = result.get_counts(transpiled_qc)

        print("Risultati della misura:", counts)
        plot_histogram(counts)
        plt.show()    

    def printKet(self, vectorized_matrix, num_digits):
        # Flatten della matrice in un array monodimensionale
        flattened_matrice = vectorized_matrix.flatten()
        # Itera attraverso ogni riga della matrice
        c = 0
        for riga in flattened_matrice:
            if riga != 0:
                c = riga

        print("∣ψ> = ")
        for i, amplitude in enumerate(vectorized_matrix):
            if amplitude != 0:
                print(f"{c} * |{i:0{num_digits}b}> +")

    def randomMNSIT8x8(self):
        mnist = datasets.load_digits()
        # Sceglie un indice casuale
        
        image = mnist.images[self.index]
        self.index+=1
        
        # Visualizza l'immagine originale
        #plt.imshow(image, cmap='gray')
        #plt.axis('off')
        #plt.show()
        
        mg_array = np.array(image)

            # Definisci la soglia: i pixel con valore maggiore di 128 sono considerati bianchi, altrimenti neri
        threshold = 1
            # Crea un array binario: 0 per bianco, 1 per nero
        binary_array = np.where(mg_array > threshold, 0, 1)
        print(binary_array)
        return binary_array

    def normalizzazione64(self, binary_array):
        binary_array = self.randomMNSIT8x8()
        norm_squared = np.sum(np.abs(binary_array) ** 2)
        # Normalizza il vettore per la radice quadrata della norma dei quadrati degli amplitudi
        normalized_params = binary_array / np.sqrt(norm_squared)
        # Appiattisci la matrice in un vettore
        vectorized_matrix = normalized_params.flatten()
        # Crea uno stato quantistico Statevector dal vettore
        quantum_state = Statevector(vectorized_matrix)

        diagonal_elements = np.diag(vectorized_matrix)

        # Calcoliamo la somma dei quadrati di questi elementi
        sum_of_squares = np.sum(np.square(diagonal_elements))

        np.set_printoptions(threshold=np.inf, suppress=True, precision=4)
        """
        print("----------------",binary_array, "------------------------------")
        if np.log2(binary_array.shape[0]) % 1 != 0:
            next_power_of_2_rows = 2 ** int(np.ceil(np.log2(binary_array.shape[0])))
            num_rows_to_add = next_power_of_2_rows - binary_array.shape[0]
            binary_array = np.vstack((binary_array, np.ones((num_rows_to_add, binary_array.shape[1]), dtype=binary_array.dtype)))

    # Verifica se il numero di colonne non è una potenza di 2
        if np.log2(binary_array.shape[1]) % 1 != 0:
            next_power_of_2_cols = 2 ** int(np.ceil(np.log2(binary_array.shape[1])))
            num_cols_to_add = next_power_of_2_cols - binary_array.shape[1]
            binary_array = np.hstack((binary_array, np.ones((binary_array.shape[0], num_cols_to_add), dtype=binary_array.dtype)))
        
        np.set_printoptions(threshold=np.inf, suppress=True, precision=4)

        # Stampa l'array
        a = 8
        reduced_array = np.zeros((a, a), dtype=int)
        print("Ridotto shape",reduced_array.shape)
        
        
        for i in range(a):
            for j in range(a):
                # Prendiamo un blocco 2x2 dall'array originale
                block = binary_array[2*i:2*i+2, 2*j:2*j+2]
                
                # Contiamo gli 1 nel blocco
                ones_count = np.sum(block)
                
                # Applichiamo la logica descritta
                if ones_count > 2:  # Più 1 che 0
                    reduced_array[i, j] = 1
                elif ones_count == 2:  # Tanti 1 quanti 0
                    reduced_array[i, j] = 1
                else:  # Più 0 che 1
                    reduced_array[i, j] = 0

        #reduced_array = 1 - reduced_array
        np.set_printoptions(threshold=np.inf, suppress=True, precision=4)

        # Stampa l'array
        #print(reduced_array)
        binary_array = reduced_array
        print("Ridotto",binary_array)"""
        

        # Stampa l'array
        #print("ba: ",binary_array.shape)
        #print("vm:" ,vectorized_matrix.shape)

        #print(f"Risultato atteso del DIP TEST è: {sum_of_squares}")
        #print(f"[0][0] vale : {binary_array[0][0]}")

        num_qubits = int(np.log2(len(quantum_state)))
        num_digits = num_qubits if num_qubits > 0 else 1
        #self.num_qubits = num_qubits -- SONO SEMPRE 10

        # Itera attraverso ogni riga della matrice
        #self.printKet(vectorized_matrix, num_digits)
        #self.measure_statevector(quantum_state)

        return self.statePrepSingle(quantum_state), vectorized_matrix
    
    def getBinary(self):
        return self.binary


    def ChooseRandomIMG(self):
        rdn = random.randint(1, 2)
        if rdn == 1:
            file_path = ".\\MINST_DATA\\t10k-images.idx3-ubyte"
        elif rdn == 2:
            file_path = ".\\MINST_DATA\\train-images.idx3-ubyte"
        return file_path

    def LoadVectorMultiplo(self):
        n = self.num_img
        current_dir = os.path.dirname(os.path.realpath(__file__))
        risultatiStateVector = []
        risultatiBinary = []
        for _ in range(n):
            # Scegliere un file casuale
            file_path = os.path.join(current_dir, self.ChooseRandomIMG())
            # Eseguire FromFileToStateVector e salvare il risultato nella lista
            risultatoSV, risultatoB = self.FromFileToStateVector(file_path)
            risultatiStateVector.append(risultatoSV)
            risultatiBinary.append(risultatoB)
        return risultatiStateVector, risultatiBinary

    def PrepareONECircuit(self):
        current_dir = os.path.dirname(os.path.realpath(__file__))
        # Scegliere un file casuale
        file_path = os.path.join(current_dir, self.ChooseRandomIMG())
        # Eseguire FromFileToStateVector e salvare il risultato nella lista
        risultato = self.FromFileToStateVector(file_path)
        #self.printCircuit(risultato)
        #print("Si lavora con uno stato PURO, lavoro con n_qubits: ", risultato.num_qubits)
        #print("RISULTATO INIZIALE-------------")
        
        
        
        
        return risultato
    
    def printCircuit(self, circuit):
        current_dir = os.path.dirname(os.path.realpath(__file__))        
        # Salva il circuito come immagine
        image_path = os.path.join(current_dir, 'PrepStatePassato.png')
        circuit_drawer(circuit, output='mpl', filename=image_path)
        
        # Apri automaticamente l'immagine
        img = Image.open(image_path)
        img.show()