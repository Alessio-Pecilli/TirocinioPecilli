import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


class QuantumStateConverter:
    def __init__(self):
        pass
        

    def extend_to_power_of_two(self, matrix):
        # Calcola la potenza di due successiva più vicina al massimo delle dimensioni della matrice dopo l'estensione
        new_size = max(matrix.shape)
        next_power_of_two = 1
        while next_power_of_two < new_size:
            next_power_of_two *= 2

        # Calcola quanto padding aggiungere per raggiungere la nuova dimensione
        pad_size = next_power_of_two - matrix.shape[0]
        pad_width = ((0, pad_size), (0, pad_size))  # Aggiunge lo stesso padding sia sopra che a sinistra

        # Aggiungi il padding alla matrice
        padded_matrix = np.pad(matrix, pad_width, mode='constant')
        print(binary_array.shape," -> ",padded_matrix.shape)  # Mostra le dimensioni dell'array
        return padded_matrix

    def to_quantum_state(self, binary_array):
        if(np.log2(len(binary_array) ) % 1 != 0):
            binary_array = self.extend_to_power_of_two(binary_array)

        # Trasformiamo la matrice in un vettore colonna
        binary_array = binary_array.flatten()

        num_ones = sum(binary_array)

        c = 1 / num_ones
        c = round(c, 5)

        print(len(binary_array) , "di cui 1:", num_ones)

        num_qubits = int(np.log2(len(binary_array)))
        num_digits = num_qubits if num_qubits > 0 else 1

        print("∣ψ> = ")
        #for i, amplitude in enumerate(binary_array):
            #if amplitude != 0:
                #print(f"{c} * |{i:0{num_digits}b}> +")


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
file_path = "C:\\Users\\aless\\Desktop\\TirocinioPecilli\\t10k-images.idx3-ubyte"


images = read_mnist_images(file_path)
#print(images.shape)


# Apri l'immagine
img = images[24]

# Visualizza immagine del dataset
#display_image(img)

# Converti l'immagine in un array NumPy
img_array = np.array(img)

# Definisci la soglia: i pixel con valore maggiore di 128 sono considerati bianchi, altrimenti neri
threshold = 128

# Crea un array binario: 0 per bianco, 1 per nero
binary_array = np.where(img_array > threshold, 0, 1)

print(binary_array)  # Mostra l'array binario
converter = QuantumStateConverter()  # Crea un'istanza della classe QuantumStateConverter
converter.to_quantum_state(binary_array)  # Chiama il metodo to_quantum_state per elaborare l'array binario


