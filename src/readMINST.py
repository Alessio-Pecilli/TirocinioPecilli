import numpy as np
import matplotlib.pyplot as plt

class MINSTImageProcessor:
    def display_image(self,  file_path, index):
        images = self.read_minst_images(file_path)
        img = images[index]
        plt.imshow(img, cmap='gray')
        plt.axis('off')  # Rimuove gli assi
        plt.show()

    def produceArray(self, file_path, index):
        images = self.read_minst_images(file_path)
        img = images[index]
        #self.display_image(file_path, index)
        # Converti l'immagine in scala di grigi GIA' E' BIANCO O NERO
        #img_gray = img.convert('L')

        # Converti l'immagine in un array NumPy
        img_array = np.array(img)

        # Definisci la soglia: i pixel con valore maggiore di 128 sono considerati bianchi, altrimenti neri
        threshold = 128
        # Crea un array binario: 0 per bianco, 1 per nero
        binary_array = np.where(img_array > threshold, 0, 1)

        #print("Matrix shape: ",binary_array.shape)  # Mostra le dimensioni dell'array
        #print(binary_array)  # Mostra l'array binario
        return binary_array

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