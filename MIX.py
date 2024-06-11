import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import QuantumStateConverter, readMINST
import os

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

current_dir = os.path.dirname(os.path.realpath(__file__))
file_path = os.path.join(current_dir, 't10k-images.idx3-ubyte')

processor = readMINST.MINSTImageProcessor()
converter = QuantumStateConverter.QuantumStateConverter()  # Crea un'istanza della classe QuantumStateConverter
binary_array = processor.produceArray(file_path, 24)
converter.to_quantum_state(binary_array)  # Chiama il metodo to_quantum_state per elaborare l'array binario


