from VQSD import VQSD
import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import Aer
from qiskit import transpile

class TestVQSD(VQSD):
    def __init__(self):
        super().__init__()
        self.create_initial_state()

    def create_initial_state(self):
        self.state_prep_circ = QuantumCircuit(self._num_qubits, self._num_qubits)
        self.state_prep_circ.h(0)
        self.state_prep_circ.cx(0, 1)
        self.state_prep_circ.ry(np.pi/4, 2)
        # Aggiungiamo misurazioni
        self.state_prep_circ.measure_all()

# Creiamo un'istanza della nostra classe di test
test_vqsd = TestVQSD()

# Calcoliamo la purezza dello stato
test_vqsd.compute_purity()
print(f"Purezza calcolata: {test_vqsd.purity}")

# Test del DIP
angles = [0] * test_vqsd.num_angles_required_for_unitary()
try:
    dip_result = test_vqsd.obj_dip_resolved(angles)
    print(f"Risultato del DIP test: {dip_result}")
except Exception as e:
    print(f"Errore nel DIP test: {str(e)}")
    # Proviamo a eseguire il circuito manualmente per vedere cosa succede
    simulator = Aer.get_backend('aer_simulator')
    # Trasponi il circuito per adattarlo al backend
    transpiled_qc = transpile(test_vqsd.state_prep_circ, simulator)
        # Esegui il circuito sul simulatore
    result = simulator.run(transpiled_qc, shots=1000).result()
    print("Risultati grezzi:")
    print(result.get_memory())

# Test del PDIP
try:
    pdip_result = test_vqsd.obj_pdip_resolved(angles)
    print(f"Risultato del PDIP test: {pdip_result}")
except Exception as e:
    print(f"Errore nel PDIP test: {str(e)}")

# Confronto tra DIP e PDIP
if 'dip_result' in locals() and 'pdip_result' in locals():
    print(f"Differenza tra DIP e PDIP: {abs(dip_result - pdip_result)}")