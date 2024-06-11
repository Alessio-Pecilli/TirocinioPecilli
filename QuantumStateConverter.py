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
        print(matrix.shape," -> ",padded_matrix.shape)  # Mostra le dimensioni dell'array
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
        for i, amplitude in enumerate(binary_array):
            if amplitude != 0:
                print(f"{c} * |{i:0{num_digits}b}> +")

