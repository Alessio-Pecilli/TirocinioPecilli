"""Tests the local cost vs global cost in VQSD."""

# =============================================================================
# Imports
# =============================================================================

import time,re

from scipy.optimize import minimize
import matplotlib.pyplot as plt
import numpy as np
import cirq
from qiskit.circuit import Parameter
import os
import sympy
from qiskit.circuit.library import RXGate, RYGate, RZGate
import cirq_google

from dip_pdip import Dip_Pdip


# =============================================================================
# Constants
# =============================================================================


nreps = 2
method = "COBYLA"
q = 0.5

#maxiter = 1000
maxiter = 2

# =============================================================================
# Functions
# =============================================================================

def process(vals):
    new = [vals[0]]
    for ii in range(1, len(vals)):
        if vals[ii] < new[-1]:
            new.append(vals[ii])
        else:
            new.append(new[-1])
    return new

# =============================================================================
# Main script
# =============================================================================

if __name__ == "__main__":    
    # Arrays to store the cost values
    OBJDIPS = []        # global cost trained with local cost
    OBJPDIPS = []       # local cost trained with local cost
    OBJGLOBALDIPS = []  # global cost trained with global cost
    OBJQDIPS = []       # global cost trained with q cost
    QOBJS = []          # weighted sum of local and global cost
    OBJGLOBALDS = [] 

    # Get a VQSD instance
    vqsd = Dip_Pdip()
    

    def objdip(self):
        val = vqsd.obj_dip(vqsd.getFinalCircuitDIP(),1)
        OBJGLOBALDIPS.append(val)
        print("DIP Test obj =", val)
        return val
    
    def objds(self):
        val = vqsd.obj_ds(vqsd.getFinalCircuitDS())
        OBJGLOBALDS.append(val)
        print("DS Test obj =", val)
        return val

    """def objpdip_compare(angs):
        print("CCCCCCCCCCCCCCCCCCCCCCCCCCCCC")
        vqsd.clear_dip_test_circ()
        pval = vqsd.obj_pdip_resolved(angs, repetitions=nreps)
        OBJPDIPS.append(pval)
        if pval is None:
            print("Impossibile calcolare l'obiettivo. Restituisco un valore grande.")
            return 1e10
        
        vqsd.clear_dip_test_circ()
        vqsd.dip_test()
        val = vqsd.obj_dip_resolved(angs, repetitions=10)
        OBJDIPS.append(val)
        print("DIP Test obj =", val)
        
        return pval
    
    # Does the weighted sum of costs
    def qcost(angs):
        #print("PDIP cost: ")
        # PDIP cost
        vqsd.clear_dip_test_circ()
        pdip = vqsd.obj_pdip_resolved(angs, repetitions=nreps)
        
        # DIP cost
        print("\nDIP cost: ")
        vqsd.clear_dip_test_circ()
        vqsd.dip_test()
        dip = vqsd.obj_dip_resolved(angs, repetitions=nreps)
        
        # weighted sum
        print(dip,"\nweighted sum: ")
        obj = q * dip + (1 - q) * pdip
        
        QOBJS.append(obj)
        print("QCOST OBJ =", obj)
        
        # DIP Cost with greater shots
        print("DIP Cost with greater shots")
        vqsd.clear_dip_test_circ()
        vqsd.dip_test()
        val = vqsd.obj_dip_resolved(angs, repetitions=10)
        OBJQDIPS.append(val)
        
        return obj"""
    
    # =========================================================================
    # Do the optimization
    # =========================================================================
    
    # Start the timer
    start = time.time()
    init = np.zeros(vqsd._num_qubits)
    # Minimize using the local cost + evaluate the global cost at each iteration
    out = minimize(objdip,init, method=method, options={"maxiter": maxiter})
    #out2 = minimize(objds,init, method=method, options={"maxiter": maxiter})
    
    # Minimize using the global cost
    #glob = minimize(objdip,init, method=method, options={"maxiter": maxiter})
    
    # Minimize using the weighted cost
    #weight = minimize(qcost,init,  method=method, options={"maxiter": maxiter})
    
    #print("PDIP angles:", [x % 2 for x in out["x"]])
    #print("DIP angles:", [x % 2 for x in glob["x"]])
    #print("Actual angles:", sprep_angles)
    
    # Print the runtime
    wall = (time.time() - start) / 60
    print("Runtime {} minutes".format(wall))
    
    # =========================================================================
    # Do the plotting
    # =========================================================================
    
    plt.figure(figsize=(6, 7))
    title = "EXACT GLOBAL EVAL {} {} Qubit Product State, {} Shots, {} Iterations, Runtime = {} min.".format(method, vqsd._num_qubits, nreps, maxiter, round(wall, 2))
    #plt.title(title)
    #print(OBJGLOBALDIPS)
    #plt.plot(process(OBJGLOBALDS), "b-o", linewidth=2, label="$C(q=0.0)$ with $C(q=0.0)$ training")
    plt.plot(process(OBJGLOBALDIPS), "g-o", linewidth=2, label="$C(q=1.0)$ with $C(q=1.0)$ training")
    #plt.plot(process(QOBJS), "r-o", linewidth=2, label="$C(q=0.5)$ with $C(q=0.5)$ training")
    #plt.plot(process(OBJDIPS), "-o", color="orange", linewidth=2, label="$C(q=1.0)$ with $C(q=0.0)$ training")
    #plt.plot(process(OBJQDIPS), "-o", color="purple", linewidth=2, label="$C(q=1.0)$ with $C(q=0.5)$ training")
    #DSTEST - DIP TEST deve venire circa 0 
    plt.grid()
    plt.legend()
    
    plt.xlabel("Iteration", fontsize=15, fontweight="bold")
    plt.ylabel("Cost", fontsize=15, fontweight="bold")
    
    
    # Generazione di un timestamp per il nome del file
    t = time.strftime('%Y-%m-%d_%H-%M-%S')

    # Pulizia del titolo per essere un nome di file valido
    safe_title = re.sub(r'[^a-zA-Z0-9_\-]', '', title.replace(' ', '_'))
# Percorso di salvataggio
    save_path = os.path.join(os.path.expanduser('~'), 'Documents', 'PURE_STATE_OUTPUT')

    # Assicurati che la directory esista
    os.makedirs(save_path, exist_ok=True)

    # Salva il grafico in PDF
    pdf_filename = os.path.join(save_path, f"{safe_title}_{t}.pdf")
    plt.savefig(pdf_filename, format='pdf')

    # Salva i dati in un file di testo
    costs = [#process(OBJPDIPS),
            #process(OBJDIPS),
            process(OBJGLOBALDIPS),
            #process(QOBJS),
            #process(OBJQDIPS)
            ]

    # Pad the lengths
    maxlen = max(len(a) for a in costs)
    for a in costs:
        while len(a) < maxlen:
            a.append(a[-1])

    data = np.array(costs)
    txt_filename = os.path.join(save_path, f"{title}_{t}.txt")
    np.savetxt(txt_filename, data.T)

