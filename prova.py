import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


import cmath
from typing import Union, Optional

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
import numpy as np
import math

from qiskit.exceptions import QiskitError
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.circuit.gate import Gate
from qiskit.circuit.library.standard_gates.x import CXGate, XGate
from qiskit.circuit.library.standard_gates.h import HGate
from qiskit.circuit.library.standard_gates.s import SGate, SdgGate
from qiskit.circuit.library.standard_gates.ry import RYGate
from qiskit.circuit.library.standard_gates.rz import RZGate
from qiskit.circuit.exceptions import CircuitError
from qiskit.quantum_info.states.statevector import Statevector  # pylint: disable=cyclic-import

# Funzione per visualizzare un'immagine
def display_image(image):
    plt.imshow(image, cmap='gray')
    plt.axis('off')  # Rimuove gli assi
    plt.show()

def read_mnist_images(file_path):
    with open(file_path, 'rb') as file:
        # Leggi l'intestazione
        magic_number = int.from_bytes(file.read(4), 'big')
        num_images = int.from_bytes(file.read(4), 'big')
        num_rows = int.from_bytes(file.read(4), 'big')
        num_cols = int.from_bytes(file.read(4), 'big')
        # Leggi i dati delle immagini
        data = np.frombuffer(file.read(), dtype=np.uint8)
        data = data.reshape((num_images, num_rows, num_cols))
    return data

# Esempio di utilizzo
file_path = "C:\\Users\\Ale\\Desktop\\TirocinioPecilli\\t10k-images.idx3-ubyte"


images = read_mnist_images(file_path)
print(images.shape)


# Apri l'immagine
img = images[15]

# Visualizza immagine del dataset
display_image(img)

# Converti l'immagine in un array NumPy
img_array = np.array(img)

# Definisci la soglia: i pixel con valore maggiore di 128 sono considerati bianchi, altrimenti neri
threshold = 128

# Crea un array binario: 0 per bianco, 1 per nero
binary_array = np.where(img_array > threshold, 0, 1)

print(binary_array.shape)  # Mostra le dimensioni dell'array


# Definire le ampiezze per gli stati |00> e |11>

# Creare un oggetto Statevector
state = Statevector(binary_array)

# Stampare lo stato
print(state)

# Stampa di a e b con la notazione di Dirac

N = len(binary_array)
c = 1 / np.sqrt(N)
# Determina il numero di cifre necessarie per rappresentare gli stati
num_qubits = int(np.log2(N))
num_digits = num_qubits if num_qubits > 0 else 1  # Almeno una cifra per il caso di 1 stato

print("∣ψ> = ")
# Stampa gli stati quantistici usando la notazione di Dirac
for i, amplitude in enumerate(binary_array):
    if amplitude != 0:  # Stampa solo gli stati con ampiezza non nulla
        print(f"{c} * |{i:0{num_digits}b}> +")