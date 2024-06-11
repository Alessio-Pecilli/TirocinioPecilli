import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

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
print(images.shape)
# Visualizza la prima immagine del dataset
#display_image(images[0])

# Apri l'immagine
img = images[0]
# Converti l'immagine in scala di grigi GIA' E' BIANCO O NERO
# img_gray = img.convert('L')

# Converti l'immagine in un array NumPy
img_array = np.array(img)

# Definisci la soglia: i pixel con valore maggiore di 128 sono considerati bianchi, altrimenti neri
threshold = 128
# Crea un array binario: 0 per bianco, 1 per nero
binary_array = np.where(img_array > threshold, 0, 1)
inverted_array = np.logical_not(binary_array).astype(int)

print(inverted_array.shape)  # Mostra le dimensioni dell'array
print(inverted_array)  # Mostra l'array binario
