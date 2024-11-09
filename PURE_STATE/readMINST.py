import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import gaussian_filter
from skimage.measure import block_reduce

class MINSTImageProcessor:


    def display_image(self,  file_path, index):
        images = self.read_minst_images(file_path)
        img = images[index]
        plt.imshow(img, cmap='gray')
        plt.axis('off')  # Rimuove gli assi
        plt.show()

    def produceArray(self, file_path, index):
        #images = self.read_minst_images(file_path)
        immagine = Image.open(file_path)
        immagine = immagine.resize((6, 6))
        immagine_bianco_nero = immagine.convert('L')
        threshold = 128
        immagine_bn_puro = immagine_bianco_nero.point(lambda p: 255 if p > threshold else 0)

# Mostra l'immagine in bianco e nero puro
        #immagine_bn_puro.show()
        array = np.array(immagine_bn_puro)

        # Ora, sostituiamo i valori 255 (bianco) con 1 e 0 (nero) con 0
        array_binario = np.where(array == 255, 1, 0)

        # Mostra l'array binario
        """
        # Mostra l'immagine in bianco e nero
        img = images[index]
        self.display_image(file_path, index)
        # Converti l'immagine in scala di grigi GIA' E' BIANCO O NERO
        #img_gray = img.convert('L')

        # Converti l'immagine in un array NumPy
        img_array = np.array(img)
        
        # Definisci la soglia: i pixel con valore maggiore di 128 sono considerati bianchi, altrimenti neri
        threshold = 128
        # Crea un array binario: 0 per bianco, 1 per nero
        binary_array = np.where(img_array > threshold, 0, 1)
        #print("Matrix shape: ",binary_array.shape)  # Mostra le dimensioni dell'array
        """
        np.set_printoptions(threshold=np.inf)
        print(array_binario)  # Mostra l'array binario
        
        return array_binario

    def read_minst_images(self, file_path):
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