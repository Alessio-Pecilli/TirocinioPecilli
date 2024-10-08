import pandas as pd
import cirq
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import qiskit
from dip_pdip import Dip_Pdip
from prepState import StatePreparation
import readMINST
from qiskit.circuit import ParameterVector
import random
from qutip import Qobj, ket2dm
from qiskit.quantum_info import Statevector
import os
from qiskit.circuit import Parameter
import math
from qiskit import QuantumRegister, transpile
from qiskit.visualization import circuit_drawer
import qiskit_aer
from qiskit_aer import Aer
from qiskit_aer import AerSimulator
from qiskit import QuantumCircuit
from qiskit.visualization import plot_histogram
from scipy.optimize import minimize
from result import Result


class Test:

    def __init__(self):
        
        self.res = Result(1)
        self.testDipPurity()

    def testDipPurity(self):
        vector = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)])
        vectorized_matrix = vector.flatten()
        # Crea uno stato quantistico Statevector dal vettore
        quantum_state = Statevector(vectorized_matrix)
        qc = QuantumCircuit(quantum_state.num_qubits)
        qc.initialize(quantum_state, range(quantum_state.num_qubits))
        circuit = Dip_Pdip(self.res.get_params(),qc,1)
        a = circuit.getFinalCircuitDS()
        print("Purezza finale ottenuta:", 2*circuit.obj_ds(a) -1)
        a = circuit.getFinalCircuitDIP()
        circuit.obj_dip_counts(a,1)
        print("Dip ottenuto:", circuit.obj_dip(a))


        
test = Test()
test.testDipPurity()