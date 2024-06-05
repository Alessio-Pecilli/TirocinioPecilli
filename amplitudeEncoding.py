import cirq
import numpy as np

"""Corrected function TRANSLATED calculate_bloch_angles and rotations_to_disentangle""" 

def calculate_bloch_angles(pair_of_complex):
    a_complex, b_complex = pair_of_complex
    a_complex = complex(a_complex)
    b_complex = complex(b_complex)
    mag_a = abs(a_complex)
    final_r = np.sqrt(mag_a**2 + abs(b_complex) ** 2)
    
    if final_r < 1e-10:
        theta = 0
        phi = 0
        final_r = 0
        final_t = 0
    else:
        theta = 2 * np.arccos(mag_a / final_r)
        a_arg = np.angle(a_complex)
        b_arg = np.angle(b_complex)
        final_t = a_arg + b_arg
        phi = b_arg - a_arg
    
    return final_r * np.exp(1j * final_t / 2), theta, phi

def rotations_to_disentangle(local_param):
    remaining_vector = []
    thetas = []
    phis = []
    param_len = len(local_param)
    
    for i in range(param_len // 2):
        remains, add_theta, add_phi = calculate_bloch_angles(local_param[2 * i : 2 * (i + 1)])
        remaining_vector.append(remains)
        thetas.append(-add_theta)
        phis.append(-add_phi)
    
    return thetas, phis



def PCNOT(T): #Are the index rights?
    for lprime in range (0,len(T) - 2):
        l = len(T) - lprime
        for j in range (0, (2**l)-2,2):
            cirq.CNOT(T[l-1][j],T[l-2][j/2])
        for j in range (1, (2**l)-1,2):
            cirq.CNOT(T[l][j],T[l-1][(j-1)/2])

    for lprime in range (0,len(T) - 3):
        l = len(T) - lprime
        for j in range (0, (2**l)-2,2):
            cirq.CNOT(T[l][j],T[l-1][j/2])
        for j in range (1, (2**l)-1,2):
            cirq.CNOT(T[l][j],T[l-1][(j-1)/2])

def Fanout(T):
    for l in range (0,len(T) - 1):
        for j in range(0,(2**l)-1):
            cirq.CNOT(T[l][j],T[l+1][2*j])
            cirq.CNOT(T[l][j],T[l+1][2*j + 1])
    for l in range(0,len(T) - 2):
            cirq.CNOT(T[l][j],T[l+1][2*j])
            cirq.CNOT(T[l][j],T[l+1][2*j + 1])

def QSP(circuit, n_levels, H, V, rotation, phase):
    for l in range(1, n_levels):
            for j in range(0,(2**(l-1))-1):
                circuit.append(cirq.CNOT(H[l-1][j], H[l][2*j]))
                circuit.append(cirq.ZPowGate(exponent=rotation[l][j]).on(H[l-1][j], H[l][2*j+1]))
                circuit.append(cirq.SWAP(H[l][2*j], H[l-1][j]))
    
    for j in range((2 * n_levels)-1):
        ph = phase[j-1]
        circuit.append(cirq.PhasedXPowGate(phase_exponent=ph).on(H[n_levels-1][j-1]))
        #TO FIX!

    for l in range(1, n_levels):
        for j in range(1, (2 ** l)-1, 2):
            circuit.append(cirq.CNOT(H[l][j], V[l][l][j]))

    for l in range(1, n_levels):
         PCNOT(V[l])

    for l in range(1, n_levels):
        for j in range(1, (2 ** l)-1, 2):
            circuit.append(cirq.CNOT(H[l][j], V[l][l][j]))

    for l in range(1, n_levels):
        Fanout(V[l])

    for lprime in range(1,n_levels-1):
        l = n_levels- lprime + 1
        for j in range(0,2**(l-1)-1):
            circuit.append(cirq.X(V[l][l][2*j]))
            circuit.append(cirq.CCNOT(V[l][l][2*j],H[l-1][j],H[l][2*j]))
            circuit.append(cirq.X(V[l][l][2*j]))
            circuit.append(cirq.CCNOT(V[l][l][2*j+1],H[l-1][j],H[l][2*j+1]))
    
    for l in range(1, n_levels):
        Fanout(V[l])
"""
def initValues(normalized_data, n_levels, rotation, phase): #TO FIX!
    # Inizializzare b come un array multidimensionale
    b = np.zeros((n_levels, 2**(n_levels-1)), dtype=float)
    b[n_levels-1, :] = np.abs(normalized_data)
    phase = np.zeros((n_levels, 2 * n_levels - 1), dtype=float)
        

    for j in range(1,n_levels):
        for l in range(0,n_levels-1):
                    b_real = b[l + 1, 2*j]
                    b_imag = b[l + 1, 2*j+1]
                    b[l][j] =np.sqrt(b_real**2 + b_imag**2)
                    
                    if b[l-1, j] != 0:
                        rotation[l, j] = np.arccos(b[l,2*j] / b[l-1, j])
                    else:
                        rotation[l, j] = 0
                    
                    phase[j] = b[l][j]

    print("b:\n", b)
    print("rotation:\n", rotation)
    print("phase:\n", phase)

    return rotation,phase
"""

D = np.array([0.20 + 0.3j, 0.70 - 0.1j, 0.05 + 0.2j, 0.05 - 0.1j])
norm = np.linalg.norm(D)
normalized_data = D / norm

circuit = cirq.Circuit()
# Numero di livelli nell'albero binario

n_levels = int(np.ceil(np.log2(len(normalized_data))))


# Creazione di H come lista di liste di qubit
H = [cirq.LineQubit.range(2**i) for i in range(n_levels)]

V = [cirq.LineQubit.range(2**i) for i in range(n_levels)]

# Inizializza H[0][0] allo stato quantistico |1>
circuit = cirq.Circuit(cirq.X(H[0][0]))
circuit.append(cirq.measure(H[0][0], key='result'))

# Creazione di variabili per gli angoli di rotazione e di fase
rotation = np.zeros((n_levels, 2**(n_levels-1)))
phase = np.random.uniform(0, 2*np.pi, size=(2 * n_levels - 1))

rotation, phase = rotations_to_disentangle(normalized_data)

QSP(circuit, n_levels, H, V, rotation, phase)

# Simulare il circuito
for i in range(n_levels):
    circuit.append(cirq.measure(H[i], key='result' + str(i)))

print("Circuito generato:")
print(circuit)
