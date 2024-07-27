"""Tests the local cost vs global cost in VQSD."""

# =============================================================================
# Imports
# =============================================================================

import time,re

from VQSD_ALE import VQSD, symbol_list_for_product
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import numpy as np
import cirq
import sympy
from qiskit.circuit.library import RXGate, RYGate, RZGate
import cirq_google


# =============================================================================
# Constants
# =============================================================================

n = 10
nreps = 10
method = "COBYLA"
q = 0.5
#maxiter = 1000
maxiter = 1

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

    # Get a VQSD instance
    vqsd = VQSD()
    
    # Get preparation angles
    #sprep_angles = np.random.rand(n)
    
    # Add the state prep circuit and compute the purity
    #Inizializzo con l'img, per noi gli angles sono le ampiezze dello stateVector
    #vqsd.product_state_prep(sprep_angles, cirq.RotXGate)
    #vqsd.compute_purity()
    
    # Add the ansatz
    num_qubits = vqsd._num_qubits

# Create rotation gates with these parameters

    vqsd.product_ansatz(symbol_list_for_product(num_qubits), RXGate)

    def get_param_resolver(num_qubits, num_layers):
        """Returns a cirq.ParamResolver for the parameterized unitary."""
        num_angles = 12 * num_qubits * num_layers
        angs = np.pi * (2 * np.random.rand(num_angles) - 1)
        return cirq.ParamResolver(
            {str(ii) : angs[ii] for ii in range(num_angles)}
        )

    # Objective function for Dip Test
    def objdip(angs):
        vqsd.clear_dip_test_circ()
        vqsd.dip_test()
        val = vqsd.obj_dip_resolved(angs, repetitions=nreps)
        OBJGLOBALDIPS.append(val)
        print("DIP Test obj =", val)
        return val

    # Objective function for PDIP Test
    def objpdip(angs):
        vqsd.clear_dip_test_circ()
        val = vqsd.obj_pdip_resolved(angs, repetitions=nreps)
        OBJPDIPS.append(val)
        print("PDIP Test obj =", val)
        return val
    
    # Does the PDIP and also appends the DIP
    def objpdip_compare(angs):
        # Do the PDIP first
        vqsd.clear_dip_test_circ()
        pval = vqsd.obj_pdip_resolved(angs, repetitions=nreps)
        OBJPDIPS.append(pval)
        if pval is None:
            print("Impossibile calcolare l'obiettivo. Restituisco un valore grande.")
            return 1e10  # o un altro valore grande
        print("\nPDIP Test obj =", pval)
        
        # Do the DIP Test next. Evaluate with many repetitions to get a
        # good estimate of the cost here. We're not training with this,
        # just evaluating the cost
        vqsd.clear_dip_test_circ()
        vqsd.dip_test()
        val = vqsd.obj_dip_resolved(angs, repetitions=10)
        OBJDIPS.append(val)
        print("DIP Test obj =", val)
        
        # return the PDIP Test obj val
        return pval
    
    # Does the weighted sum of costs
    def qcost(angs):
        print("PDIP cost: ")
        # PDIP cost
        vqsd.clear_dip_test_circ()
        pdip = vqsd.obj_pdip_resolved(angs, repetitions=nreps)
        
        # DIP cost
        print(pdip, "\nDIP cost: ")
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
        
        return obj
    
    # =========================================================================
    # Do the optimization
    # =========================================================================
    
    # Initial values
    init = np.random.random(10) * 0.1  # angoli piccoli
    # Start the timer
    start = time.time()
    
    # Minimize using the local cost + evaluate the global cost at each iteration
    out = minimize(objpdip_compare, init, method=method, options={"maxiter": maxiter})
    
    # Minimize using the global cost
    glob = minimize(objdip, init, method=method, options={"maxiter": maxiter})
    
    # Minimize using the weighted cost
    weight = minimize(qcost, init, method=method, options={"maxiter": maxiter})
    
    print("PDIP angles:", [x % 2 for x in out["x"]])
    print("DIP angles:", [x % 2 for x in glob["x"]])
    #print("Actual angles:", sprep_angles)
    
    # Print the runtime
    wall = (time.time() - start) / 60
    print("Runtime {} minutes".format(wall))
    
    # =========================================================================
    # Do the plotting
    # =========================================================================
    
    plt.figure(figsize=(6, 7))
    title = "EXACT GLOBAL EVAL {} {} Qubit Product State, {} Shots, {} Iterations, Runtime = {} min.".format(method, n, nreps, maxiter, round(wall, 2))
    #plt.title(title)
    
    plt.plot(process(OBJPDIPS), "b-o", linewidth=2, label="$C(q=0.0)$ with $C(q=0.0)$ training")
    plt.plot(process(OBJGLOBALDIPS), "g-o", linewidth=2, label="$C(q=1.0)$ with $C(q=1.0)$ training")
    plt.plot(process(QOBJS), "r-o", linewidth=2, label="$C(q=0.5)$ with $C(q=0.5)$ training")
    plt.plot(process(OBJDIPS), "-o", color="orange", linewidth=2, label="$C(q=1.0)$ with $C(q=0.0)$ training")
    plt.plot(process(OBJQDIPS), "-o", color="purple", linewidth=2, label="$C(q=1.0)$ with $C(q=0.5)$ training")
    
    plt.grid()
    plt.legend()
    
    plt.xlabel("Iteration", fontsize=15, fontweight="bold")
    plt.ylabel("Cost", fontsize=15, fontweight="bold")
    
    # Save the figure
    t = time.strftime('%Y-%m-%d_%H-%M-%S')
    safe_title = re.sub(r'[^a-zA-Z0-9_\-]', '', title.replace(' ', '_'))
    plt.savefig(safe_title + '_' + t + '.pdf', format='pdf')
    
    # =========================================================================
    # Write the data to a text file
    # =========================================================================

    costs = [process(OBJPDIPS),
             process(OBJDIPS),
             process(OBJGLOBALDIPS),
             process(QOBJS),
             process(OBJQDIPS)]
    
    # Pad the lengths
    maxlen = max([len(a) for a in costs])
    for a in costs:
        while len(a) < maxlen:
            a.append(a[-1])

    data = np.array(costs)
    
    fname = title + t + ".txt"
    
    np.savetxt(fname, data.T)
    

