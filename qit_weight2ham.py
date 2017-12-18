import numpy as np
import scipy as sp
from scipy import linalg
#import Translation
#import ParTrace
import matplotlib.pyplot as plt
from scipy.linalg import norm, eigh, eigvalsh, sqrtm, svd, svdvals, det
from scipy import sparse
from scipy.sparse import csr_matrix
from qit import *
from qit.state import state, fidelity

X = np.matrix([[0,1],[1,0]])
Y = np.matrix([[0,-1j],[1j,0]])
Z = np.matrix([[1,0],[0,-1]])
I = np.matrix([[1,0],[0,1]])

e_0 = np.matrix([1,0]).T
e_1 = np.matrix([0,1]).T





def CountSpins(W):
    N = 0 #Initialize number of spins
    num_layers = 0
    spins_per_layer = []
    flat_W = np.array([])
    for layer in W:
        flat_W = np.append(flat_W,layer.flatten())
        if num_layers == 0:
            N += layer.shape[0]+layer.shape[1]
            spins_per_layer = np.append(np.append(spins_per_layer, layer.shape[0]), layer.shape[1])
        else:
            N += layer.shape[1]
            spins_per_layer = np.append(spins_per_layer, layer.shape[1])
        num_layers += 1
    num_layers += 1
    num_w = flat_W.shape[0]
    return N, num_layers, spins_per_layer, flat_W, num_w

def Weight2Matrix2(w, i, j, N):
    M = sp.sparse.kron(sp.sparse.eye(2**i),sp.sparse.kron(Z,sp.sparse.kron(sp.sparse.eye(2**(j-i-1)),sp.sparse.kron(Z,sp.sparse.eye(2**(N-j-1)))))) 
    return w*M

def BuildHamiltonian2(W, N, spins_per_layer):
    H = sp.sparse.csr_matrix(np.zeros([2**N, 2**N]))
    for k,layer in enumerate(W):
        for a,row in enumerate(layer):   
            i = a + int(spins_per_layer[:k].sum()) 
            for b,w in enumerate(row):
                j = b + int(spins_per_layer[:k+1].sum()) 
                M = Weight2Matrix2(w, i, j, N)
                #print(M.diagonal())
                H = np.add(H, M)
    return -H
'''
def Weight2Matrix(w, i, j, N):
    M = 1
    for p in range(N):
        if p == i or p == j:
            M = np.kron(M, Z)
        else:
            M = np.kron(M, I)
    return w*M


def BuildHamiltonian(W, N, spins_per_layer):
    H = np.zeros([2**N, 2**N])
    for k,layer in enumerate(W):
        for a,row in enumerate(layer):   
            i = a + int(spins_per_layer[:k].sum()) 
            for b,w in enumerate(row):
                j = b + int(spins_per_layer[:k+1].sum()) 
                M = Weight2Matrix(w, i, j, N)
                #print(M.diagonal())
                H = np.add(H, M)
    return -H
'''

def Ham2Density(H):
    # NOTE: What about the beta parameter?
    beta = 1
    expH = linalg.expm(-beta*H)
    #Z = np.trace(expH)
    Z = expH.diagonal().sum()
    rho = (expH)/Z
    return rho


def SpinsInLayer(spins_per_layer, layer):
    indices = []
    previous_spins = int(np.array(spins_per_layer[:layer]).sum())
    for i in range(previous_spins, previous_spins + int(spins_per_layer[layer])):
        indices.append(i)
    return indices


def ObtainPartialDens(DensityMatrix, N, spins_per_layer):
    Density_per_layer = []
    for i,e in enumerate(spins_per_layer):
         dont_trace = SpinsInLayer(spins_per_layer, i)
         Density_per_layer.append(DensityMatrix.ptrace([j for j in range(N) if j not in dont_trace]))        
    return Density_per_layer


def ObtainPartialJoint(DensityMatrix, spins_per_layer, N, layer_p, layer_q, Density_per_layer):
    dont_trace = SpinsInLayer(spins_per_layer, layer_p) + SpinsInLayer(spins_per_layer, layer_q)
    return DensityMatrix.ptrace([j for j in range(N) if j not in dont_trace]) 



def Entropy_qit(rho):
    entropy = rho.entropy()
    return entropy



def RelativeEntropy(rho, sigma):
    # Calculate S(rho || sigma)
    rho_np = rho.data
    sigma_np = sigma.data
    rho_e = eigvalsh(rho_np)
    sigma_e = eigvalsh(sigma_np)
    return -rho_e.dot(np.log2(sigma_e)) + rho_e.dot(np.log2(rho_e))


def MutualInf(rho, sigma, joint):
    return Entropy_qit(rho) + Entropy_qit(sigma) - Entropy_qit(joint)


def RelEnt_allLayers(layer, Density_per_layer):
    rel_entropies = []
    for i, e in enumerate(Density_per_layer):
        if i <= layer:
            rel_entropies.append(RelativeEntropy(Density_per_layer[layer], Density_per_layer[i]))
        if i > layer:
            rel_entropies.append(RelativeEntropy(Density_per_layer[i], Density_per_layer[layer]))
    return rel_entropies    


def RelEnt_firstANDlast(layer, Density_per_layer):
    output = []
    output.append(RelativeEntropy(Density_per_layer[0], Density_per_layer[layer]))
    #output.append(RelativeEntropy(Density_per_layer[-1], Density_per_layer[layer]))
    output.append(RelativeEntropy(Density_per_layer[layer], Density_per_layer[-1]))
    return output     


def MI_firstANDlast(layer, DensityMatrix, Density_per_layer, spins_per_layer, N):
    output = []
    last_layer = np.array(spins_per_layer).shape[0] - 1
    output.append(MutualInf(Density_per_layer[0], Density_per_layer[layer], ObtainPartialJoint(DensityMatrix, spins_per_layer, N, layer, 0, Density_per_layer)))
    #print(ObtainPartialJoint(DensityMatrix, spins_per_layer, N, layer, 0, Density_per_layer).shape)
    output.append(MutualInf(Density_per_layer[layer], Density_per_layer[-1], ObtainPartialJoint(DensityMatrix, spins_per_layer, N, last_layer, layer, Density_per_layer)))
    #print(ObtainPartialJoint(DensityMatrix, spins_per_layer, N, last_layer, layer, Density_per_layer).shape)
    return output   
        

def plot_RelEnt(D_1_i, D_N_i, label):
    fig2, ax2 = plt.subplots()
    ax2.scatter(D_1_i, D_N_i)
    plt.ylabel('D(Layer i || layer N)')
    plt.xlabel('D(layer 0 || layer i)')  
    for i, txt in enumerate(label):
        ax2.annotate(txt, (D_1_i[i], D_N_i[i]))
    plt.savefig('Info_D_plot_test_qit.pdf')
    plt.close()
    
    plt.figure(1)
    plt.title('D(Layer 1 || i) v/s Layers')
    plt.ylabel('D(Layer 1 || layer i)')
    plt.xlabel('Layers (first to last)')
    plt.scatter(np.arange(len(D_1_i)).tolist(), D_1_i) 
    plt.savefig('Rel_entropy_layer1_test_qit.pdf')
    plt.close()
    
    plt.figure(2)
    plt.title('D(Layer i || N) v/s Layers')
    plt.ylabel('D(Layer i || layer N)')
    plt.xlabel('Layers (first to last)')
    a = np.arange(len(D_N_i)).tolist()
    #a.reverse()
    plt.scatter(a, D_N_i) 
    plt.savefig('Rel_entropy_layerN_test_qit.pdf')
    plt.close()



def plot_MI(M_1_i, M_N_i, label):

    #print(label)
    #print(M_1_i)
    #print(M_N_i)
    fig1, ax1 = plt.subplots()
    ax1.scatter(M_1_i, M_N_i)
    plt.ylabel('Mi(Layer i , layer N)')
    plt.xlabel('Mi(layer 0 , layer i)')  
    for i, txt in enumerate(label):
        ax1.annotate(txt, (M_1_i[i], M_N_i[i]))
    plt.savefig('Info_MI_plot_test_qit.pdf')
    plt.close()
    
    plt.figure(3)
    plt.title('Mi(Layer 1 || i) v/s Layers')
    plt.ylabel('Mi(Layer 1 || layer i)')
    plt.xlabel('Layers (first to last)')
    plt.scatter(np.arange(len(M_1_i)).tolist(), M_1_i) 
    plt.savefig('MI_layer1_test_qit.pdf')
    plt.close()
    
    plt.figure(4)
    plt.title('Mi(Layer i , N) v/s Layers')
    plt.ylabel('Mi(Layer i , layer N)')
    plt.xlabel('Layers (first to last)')
    a = np.arange(len(M_N_i)).tolist()
    #a.reverse()
    plt.scatter(a, M_N_i) 
    plt.savefig('MI_layerN_test_qit.pdf')
    plt.close()
             
    
   
def main():
    W = np.load('rbm_weights.npy')
    #W = np.array([[[5,1], [0,1]], [[1,1], [1,1]]])
    #W = np.array([[[1,1], [1,1]]])
    equal_nodes = False ## True if all layers have same number of nodes. Important for relative entropy calculation
    N, num_layers, spins_per_layer, flat_W, num_w = CountSpins(W) # N: Total number of spins; num_layers: number of layers; spins_per_layer: array where ith element is the number of spins in the ith layer
    H = BuildHamiltonian2(W, N, spins_per_layer)
    #print(H.diagonal())

    rho = Ham2Density(H)
    rho = state(rho.A, [2]*N)
    Density_per_layer = ObtainPartialDens(rho, N, spins_per_layer)

    RelEnt_coord = []
    MI_coord = []
    label = []
    word = 'Layer '
    for i,e in enumerate(spins_per_layer):
        MI_coord.append(MI_firstANDlast(i, rho, Density_per_layer, spins_per_layer, N)) # Check first and last layer to see that all works
        label.append(word + str(i))
        if equal_nodes:
            RelEnt_coord.append(RelEnt_firstANDlast(i, Density_per_layer))
        else:
            RelEnt_coord.append('vacio') #This is for the iterator in the zip    
    #print(RelEnt_coord)
    #print(RelativeEntropy(Density_per_layer[1], Density_per_layer[-1]))}
    #print(MI_coord)
    #print(RelativeEntropy(Density_per_layer[1], Density_per_layer[-1]))
    M_1_i = []
    M_N_i = []
    D_1_i = []
    D_N_i = []
    for E, I in zip(RelEnt_coord, MI_coord):
        if equal_nodes:
            D_1_i.append(E[0])
            D_N_i.append(E[1])
        M_1_i.append(I[0])
        M_N_i.append(I[1])        
    #label = ['layer 0', 'layer 1', 'layer 2', 'layer 3']

    if equal_nodes:
        plot_RelEnt(D_1_i, D_N_i, label)
    plot_MI(M_1_i, M_N_i, label)



def test():
    print('hola, este es un test')
    a = spin2front(1, 10, 4)


if __name__ == '__main__':
    main()
    print('Fin')
    #test()

