import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import readMINST
import VQSD_QISKIT as vqsd
import random
from qutip import Qobj, ket2dm
from qiskit.quantum_info import Statevector
import os
import math
from qiskit import transpile
import state_preparation
import qiskit_aer
from qiskit_aer import Aer
from qiskit_aer import AerSimulator
from qiskit import QuantumCircuit
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

def create_density_matrix(state_vectors):
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

def FromFileToStateVector(file_path):
    processor = readMINST.MINSTImageProcessor()
    binary_array = processor.produceArray(file_path,random.randint(1, 10000))
    return normalizzazione(binary_array)

def measure_statevector(quantum_state, num_qubits):
    
    # Crea un circuito quantistico con il numero di qubit corrispondente
    qc = QuantumCircuit(num_qubits, num_qubits)
    
    # Inizializza il circuito con lo stato fornito
    qc.initialize(quantum_state, range(num_qubits))
    
    # Aggiungi operazioni di misura
    qc.measure(range(num_qubits), range(num_qubits))

    # Usa il simulatore Aer
    simulator = Aer.get_backend('aer_simulator')

    # Trasponi il circuito per adattarlo al backend
    transpiled_qc = transpile(qc, simulator)

    # Esegui il circuito sul simulatore
    result = simulator.run(transpiled_qc, shots=1000).result()

    # Ottieni i risultati
    counts = result.get_counts(transpiled_qc)

    #print("Risultati della misura:", counts)
    #plot_histogram(counts)
    #plt.show()

def printKet(vectorized_matrix,num_digits):

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

def normalizzazione(binary_array):

    if np.log2(binary_array.shape[0]) % 1 != 0:
        next_power_of_2_rows = 2 ** int(np.ceil(np.log2(binary_array.shape[0])))
        num_rows_to_add = next_power_of_2_rows - binary_array.shape[0]
        binary_array = np.vstack((binary_array, np.zeros((num_rows_to_add, binary_array.shape[1]), dtype=binary_array.dtype)))

# Verifica se il numero di colonne non è una potenza di 2
    if np.log2(binary_array.shape[1]) % 1 != 0:
        next_power_of_2_cols = 2 ** int(np.ceil(np.log2(binary_array.shape[1])))
        num_cols_to_add = next_power_of_2_cols - binary_array.shape[1]
        binary_array = np.hstack((binary_array, np.zeros((binary_array.shape[0], num_cols_to_add), dtype=binary_array.dtype)))
    
    norm_squared = np.sum(np.abs(binary_array) ** 2)
        
    # Normalizza il vettore per la radice quadrata della norma dei quadrati degli amplitudi

    normalized_params = binary_array / np.sqrt(norm_squared)

    # Appiattisci la matrice in un vettore
    vectorized_matrix = normalized_params.flatten()

    # Crea uno stato quantistico Statevector dal vettore
    quantum_state = Statevector(vectorized_matrix)

    num_qubits = int(np.log2(len(quantum_state)))
    num_digits = num_qubits if num_qubits > 0 else 1
    # Itera attraverso ogni riga della matrice

    #printKet(vectorized_matrix,num_digits)

    measure_statevector(quantum_state,num_qubits)

    return quantum_state

def ChooseRandomIMG():
    rdn = random.randint(1, 2)
    if rdn == 1:
        file_path = ".\\MINST_DATA\\t10k-images.idx3-ubyte"
    elif rdn == 2:
        file_path = ".\\MINST_DATA\\train-images.idx3-ubyte"
    return file_path

def LoadVector(n):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    risultati = []
    for _ in range(n):
        # Scegliere un file casuale
        file_path = os.path.join(current_dir, ChooseRandomIMG())
        
        # Eseguire FromFileToStateVector e salvare il risultato nella lista
        risultato = FromFileToStateVector(file_path)
        risultati.append(risultato)
    return risultati
    
density_matrix = create_density_matrix(LoadVector(3))

#print("Matrice densità quantistica:")
#print(density_matrix)
