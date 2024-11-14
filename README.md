**ENG:**
--------------------------------------


**Project Objective:**

The objective of this project is to achieve results comparable to those obtained through QPCA using a VQSD algorithm, with MNIST images as the starting input.

**QPCA: Algorithm Purpose**
Principal Component Analysis (PCA) is a statistical technique used to reduce a dataset’s dimensionality while preserving as much of the original variability as possible. Its main purpose is to identify the principal directions in which the data varies most, projecting the original data into a lower-dimensional space.
In practice, PCA finds a new coordinate basis where the axes (the principal components) are chosen to maximize data variance along each axis. This process is helpful when working with large datasets, as reducing dimensionality helps to lower noise and simplify the model. The main steps of PCA are:
1. **Data Standardization**: To ensure all variables carry equal weight, data are normalized by subtracting the mean and dividing by the standard deviation.
2. **Covariance Matrix Calculation**: This step helps to understand how variables are correlated.
3. **Eigenvectors and Eigenvalues Calculation**: Eigenvectors represent the primary directions (principal components), while eigenvalues indicate the significance of each direction.
4. **Selection of Principal Components**: The principal components that explain most of the variance are chosen, thereby reducing dimensionality.

**Quantum Principal Component Analysis (QPCA)**
QPCA is a quantum variant of PCA that leverages quantum computing principles to achieve dimensionality reduction more efficiently than classical approaches. While classical PCA requires operations like covariance matrix decomposition and eigenvector calculation, QPCA uses the quantum computer’s ability to leverage superposition and entanglement to perform these calculations more quickly.
In particular, QPCA utilizes quantum algorithms to compute the principal components, reducing computation time, especially for large datasets. The most notable algorithm for QPCA relies on Quantum Singular Value Decomposition (QSVD), which can be used to obtain eigenvectors and eigenvalues more efficiently than classical algorithms.
**Advantages of QPCA**
- **Speed**: Quantum algorithms can significantly reduce computation time for large datasets by utilizing superposition and entanglement.
- **Security**: Quantum computing offers a higher level of security and privacy, though this advantage is not always the primary focus of QPCA.
- **Data Handling Efficiency**: QPCA is particularly useful when dealing with massive datasets or tackling problems that require high computational complexity.

**VQSD: Algorithm Purpose**
The goal of the VQSD algorithm is to output the Eigenvalues and Eigenvectors of a given quantum state ρ. This state, along with a copy, is passed through a quantum circuit \(U(\theta)\), with parameters \(\theta\) initialized randomly.
By executing the circuit a sufficient number of times, the objective is to optimize the parameters in order to minimize the cost function \(C(U(\theta))\). Once the cost function is optimized, the optimal \(\theta\) parameters will allow us to obtain the desired Eigenvalues and Eigenvectors.
The quantum state represented by ρ may be either pure or mixed:
- **Pure State**:
  - Perfectly known state (maximum system knowledge)
  - Quantum uncertainty on measurement
- **Mixed State**:
  - The state is not fully known (introducing an ensemble of states)
  - Uncertainty in both classical and quantum measurement

**Cost Function C**
The goal is to find parameters \(\theta\) and a sequence of gates \(Up(\theta)\) such that \(\theta^* = \arg \min_{\theta} C(Up(\theta))\).
Thus, the aim is to find a cost function \(C\) that can be efficiently calculated on classical-quantum hardware. To this end, cost functions \(C\) that can be expressed in terms of the purity of a quantum state are considered.
Calculating the purity of a state ψ with a quantum computer exponentially reduces the number of calculations, leading to linear complexity.
The cost function is defined as follows:
\[ C(Up(\theta)) = \text{Tr}(\psi^2) - \text{Tr}(Z(\psi^2)) \]
It is important to note:
\[ C(Up(\psi)) = 0 \iff \psi = Z(\psi) \]
C can be interpreted in three equivalent ways in the context of a density matrix:
- The minimal distance between the state ψ and states that are diagonal only
- The distance between the state ψ and \(Z(\psi)\)
- The squared sum of the absolute values of the off-diagonal elements of ψ, representing quantum coherence
C represents the upper bound of eigenvalue error \(C > \Delta\lambda\) and corresponds exactly to the eigenvector error \(C = \Delta v\).

**IT:**
--------------------------------------
**Scopo del progetto**:
Lo scopo del progetto è quello di ottenere i risultati ottenibile tramite la QPCA attraverso un algoritmo VQSD, come input di partenza sono state usate delle immagini MNIST

**QPCA: Scopo dell’algoritmo**
La PCA (Principal Component Analysis) è una tecnica statistica utilizzata per ridurre la dimensionalità di un dataset, mantenendo il più possibile la variabilità originale. Il suo scopo principale è quello di identificare le direzioni principali in cui i dati variano
di più, proiettando i dati originali in uno spazio di dimensioni inferiori.
In termini pratici, la PCA trova una nuova base di coordinate in cui gli assi (le componenti principali) sono scelti in modo da massimizzare la varianza dei dati lungo ciascun asse. 
Questo processo è utile quando si lavora con grandi moli di dati, poiché ridurre la dimensionalità aiuta a ridurre il rumore e a semplificare il modello. Le
principali fasi della PCA sono:
1. **Standardizzazione dei dati**: Per garantire che tutte le variabili abbiano lo stesso peso, si normalizzano i dati, sottraendo la media e dividendo per la deviazione standard.
2. **Calcolo della matrice di covarianza**: Per comprendere come le variabili siano correlate tra loro.
3. **Calcolo degli autovettori e degli autovalori della matrice di covarianza**: Gli autovettori rappresentano le direzioni principali (componenti principali), mentre gli autovalori indicano l’importanza di ciascuna direzione.
4. **Selezione dei principali componenti**: Si scelgono le componenti principali che spiegano la maggior parte della varianza, riducendo la dimensionalità.

**QPCA (Quantum Principal Component Analysis)**
La QPCA è una variante quantistica della PCA, che sfrutta i principi della computazione quantistica per ottenere una riduzione della dimensionalità in modo più efficiente rispetto agli approcci classici. Mentre la PCA classica richiede operazioni come la de-
composizione della matrice di covarianza e il calcolo degli autovettori, la QPCA si basa sulla capacità dei computer quantistici di sfruttare la sovrapposizione e l’entanglement per eseguire questi calcoli in modo più veloce.
In particolare, la QPCA utilizza gli algoritmi quantistici per calcolare le compo nenti principali, riducendo così i tempi di calcolo, specialmente per dataset di grandi dimensioni. 
L’algoritmo più noto per la QPCA è quello basato sulla Quantum Singular Value Decomposition (QSVD), che può essere utilizzato per ottenere gli autovettori e gli autovalori in modo più efficiente rispetto agli algoritmi classici.
Vantaggi della QPCA
• **Velocità**: Gli algoritmi quantistici possono ridurre significativamente il tempo di calcolo per grandi dataset, grazie all’uso della sovrapposizione e dell’entanglement.
• **Sicurezza**: La computazione quantistica offre un livello di sicurezza e di privacy superiori, sebbene questo vantaggio non sia sempre il focus principale della QPCA.
• **Efficienza nella gestione dei dati**: La QPCA è particolarmente utile quando si lavora con enormi moli di dati o quando si vogliono risolvere problemi che richiedono una complessità computazionale elevata.

**VQSD: Scopo dell’algoritmo**
Lo scopo dell’algoritmo è quello di restituire Autovalori ed Autovettori di un determinato stato quantistico ρ.
Questo stato, e una copia di esso verranno passati attraverso un circuito quantistico U(θ), con paratri θ inizialmente casuali.
Andando ad eseguire il circuito un adeguato numero di volte, il mio obiettivo è quello di andare a ottimizzare i miei parametri, con l’obiettivo di minimizzare la mia funzione di costo C(U(θ)).
Una volta che la mia funzione di costo sarà ottimizzata, avrò dei parametri θ ottimali, che mi permetteranno di ottenere di ottenere gli Autovalori e gli Autovettori desiderati.
Lo stato quantistico che può essere rappresentato da ρ, può essere puro o misto:
Stato puro:
• Stato perfettamente noto (massima conoscenza del sistema)
• Indeterminazione quantistica sulla misura
Stato misto:
• Lo stato non è del tutto noto (si introduce un ensemble di stati)
• Indeterminazione sulla misura classica e quantistica

**Funzione C**
L’obiettivo è quello di trovare dei parametri θ e una sequenza di gate Up(θ) tali che **θ∗ = arg min θ C(Up(θ))**
Si vuole quindi cercare una funzione di costo C che sia efficacemente calcolabile su un hardware classico - quantistico. Per questo motivo, verranno considerate funzioni di costo C esprimibili come purezza per una stato quantistico. 
Calcolare la purezza di uno stato ψ con un computer quantistico riduce esponenzialmente i calcoli, portando ad avere una complessità lineare.
La funzione di costo con cui si andrà a lavorare è le seguente:
**C(Up(θ)) = Tr(ψ2) − Tr(Z(ψ2))**
È importante considerare che:
**C(Up(ψ)) = 0 ⇐⇒ ψ = Z(ψ)**
C può essere interpretata in 3 diversi modi, equivalenti nel contesto di una matrice densità:
• La distanza minima tra lo stato ψ e gli stati che sono solo diagonali
• La distanza tra lo stato ψ e Z(ψ)
• Somma quadratica dei valori assoluti degli elementi fuori la diagonale di ψ, rappresenta quindi la coerenza quantistica
C rappresenta il limite superiore dell’errore dell’autovalore, C > ∆λ, e corrisponde esattamente all’errore dell’autovettore C = ∆v.
