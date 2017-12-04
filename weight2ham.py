import numpy as np
import scipy as sp
from scipy import linalg
#import Translation
#import ParTrace
import matplotlib.pyplot as plt
from scipy.linalg import norm, eigh, eigvalsh, sqrtm, svd, svdvals, det

Z = np.matrix([[1,0],[0,-1]])
I = np.matrix([[1,0],[0,1]])
e_0 = np.matrix([1,0]).T
e_1 = np.matrix([0,1]).T


def PartialTrace(FirstNSpinsToStay,NumberOfSpins,DensityMatrix):
    dimToTraceOut = 2**(NumberOfSpins-FirstNSpinsToStay)
    dimToStay = 2**FirstNSpinsToStay
    return np.einsum('ijkj->ik',DensityMatrix.reshape((dimToStay,dimToTraceOut,dimToStay,dimToTraceOut)))


def PartialTraceSparse(FirstNSpinsToStay,NumberOfSpins,DensityMatrix):
    dimToTraceOut = 2**(NumberOfSpins-FirstNSpinsToStay)
    dimToStay = 2**FirstNSpinsToStay
    result = np.einsum('ijkj->ik',DensityMatrix.toarray().reshape((dimToStay,dimToTraceOut,dimToStay,dimToTraceOut)))
    return sp.sparse.csr_matrix(result)


def PartialTraceVec(FirstNSpinsToStay,NumberOfSpins,StateKet):
    StateKetResh = StateKet.reshape(2**FirstNSpinsToStay,2**(NumberOfSpins-FirstNSpinsToStay))
    return np.dot(StateKetResh,StateKetResh.transpose())


# Translates 1st -> 2nd, 2nd –> 3rd, ... , Nth –> 1st (So translates to the right)
def Translation(NumberOfSpins):
    permu = sp.sparse.csr_matrix(np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]]))
    result = sp.sparse.kron(permu,sp.sparse.eye(2**(NumberOfSpins-2)))
    for i in range(1,NumberOfSpins-1):
        result = result.dot(sp.sparse.kron(sp.sparse.eye(2**i),sp.sparse.kron(permu,sp.sparse.eye(2**(NumberOfSpins-2-i)))))
    return result

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


def Weight2Matrix(w, i, j, N, spins_per_layer):
    M = 1
    for p in range(N):
        if p == i or p == j:
            M = np.kron(M, Z)
        else:
            M = np.kron(M, I)
    return w*M


def BuildHamiltonian(W, N, spins_per_layer):
    H = np.zeros(2**N)
    for k,layer in enumerate(W):
        for a,row in enumerate(layer):   
            i = a + int(spins_per_layer[:k].sum()) 
            for b,w in enumerate(row):
                j = b + int(spins_per_layer[:k+1].sum()) 
                M = Weight2Matrix(w, i, j, N, spins_per_layer)
                H = np.add(H, M)
    return H

'''
def BuildHamiltonian2(N, flat_W):
    H = np.zeros(2**N)
    for k,w in enumerate(flat_W):
        M = Weight2Matrix(w, k, N)
        H = np.add(H, M)
    return H
'''
def Ham2Density(H):
    # NOTE: What about the beta parameter?
    beta = 1
    expH = linalg.expm(-beta*H)
    Z = np.trace(expH)
    rho = (expH)/Z
    return rho
'''
def ObtainPartialDens(rho, spins_per_layer, N):
    Left_M0 = 1
    Left_M1 = 1
    Right_M0 = 1
    Left_M1 = 1
    current_spin = 0
    partial_per_layer = []
    for spins in spins_per_layer:
       for i in range(spins):
           a = i + current_spin
           for k in range(N):
               if k == i:
                   Left_M0 = np.kron(Left_M0, e_0.transpose() )
                   Left_M1 = np.kron(Left_M0, e_1.transpose() )
               else:
                   Left_M0 = np.kron(Left_M0, I)
                   Left_M1 = np.kron(Left_M1, I)
           Left = Left_M0 + Left_M1
           current_spin += 1
    current_spin += spins 
'''                   
def Apply_n_translations(M, n, num_spins):
    M = sp.sparse.csr_matrix(M)
    Trans = Translation(num_spins)
    #print(type(M))
    #print(type(Trans))
    if n > 0:
        for i in range(n):
            M = (Trans.dot(M)).dot(Trans.transpose())
            #M = Trans.transpose()
            #print(sp.sparse.csr_matrix.get_shape(M))
            #print(M.shape)
            #print(type(M))
            #print(M)
            #break
        return sp.sparse.csr_matrix.toarray(M)
    else:
        return sp.sparse.csr_matrix.toarray(M)

    

def ObtainPartialDens(rho, spins_per_layer, N):
    Density_per_layer = []
    for i, spins in enumerate(spins_per_layer):
        spins = int(spins)
        if i > 0:
            num_translations = int(np.sum(spins_per_layer[i:]))
        else:
            num_translations = 0
        DensityMatrix = Apply_n_translations(rho, num_translations, N)
        #print(DensityMatrix.shape)
        Density_per_layer.append(PartialTrace(spins, N, DensityMatrix))     
    return Density_per_layer

'''
def ObtainPartialJoint(rho, spins_per_layer, N, layer_p, layer_q)
    # NOTE: layer_p > layer_q
    num_translations = int(np.sum(spins_per_layer[i:]))
    DensityMatrix = Apply_n_translations(rho, num_translations, N)
'''    


def Entropy_qit(rho):
    rho_e = eigvalsh(rho)
    return -rho_e.dot(np.log2(rho_e))

def RelativeEntropy2(rho, sigma):
    # Calculate S(rho || sigma)
    return -np.trace(rho.dot(linalg.logm(sigma))) - Entropy(rho)

def RelativeEntropy(rho, sigma):
    # Calculate S(rho || sigma)
    rho_e = eigvalsh(rho)
    sigma_e = eigvalsh(sigma)
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
    
    
def main():
    W = np.load('rbm_weights.npy')
    #W = np.array([[[1,1],[1,1]]])
    N, num_layers, spins_per_layer, flat_W, num_w = CountSpins(W)
    #print(W)
    H = BuildHamiltonian(W, N, spins_per_layer)
    #print(H)
    #print(H.shape)
    rho = Ham2Density(H)
    #print(rho)
    Density_per_layer = ObtainPartialDens(rho, spins_per_layer, N)
    #print(np.sum(spins_per_layer))
    #print(Density_per_layer[0])

    
    RelEnt_coord = []
    label = []
    word = 'Layer '
    for i,e in enumerate(spins_per_layer):
        RelEnt_coord.append(RelEnt_firstANDlast(i, Density_per_layer))
        label.append(word + str(i))
    #print(RelEnt_coord)
    #print(RelativeEntropy(Density_per_layer[1], Density_per_layer[-1]))
    
    D_1_i = []
    D_N_i = []
    for e in RelEnt_coord:
        D_1_i.append(e[0])
        D_N_i.append(e[1])
    #label = ['layer 0', 'layer 1', 'layer 2', 'layer 3']
    fig, ax = plt.subplots()
    ax.scatter(D_1_i, D_N_i)
    plt.ylabel('D(Layer i || layer N)')
    plt.xlabel('D(layer 0 || layer i)')  
    for i, txt in enumerate(label):
        ax.annotate(txt, (D_1_i[i], D_N_i[i]))
    plt.savefig('Info_plot_test.pdf')
    plt.close()
    
    plt.figure(1)
    plt.title('D(Layer 1 || i) v/s Layers')
    plt.ylabel('D(Layer 1 || layer i)')
    plt.xlabel('Layers (first to last)')
    plt.scatter(np.arange(len(D_1_i)).tolist(), D_1_i) 
    plt.savefig('Rel_entropy_layer1_test.pdf')
    plt.close()
    
    plt.figure(2)
    plt.title('D(Layer i || N) v/s Layers')
    plt.ylabel('D(Layer i || layer N)')
    plt.xlabel('Layers (first to last)')
    a = np.arange(len(D_N_i)).tolist()
    #a.reverse()
    plt.scatter(a, D_N_i) 
    plt.savefig('Rel_entropy_layerN_test.pdf')
    plt.close()
    

         
    '''
    rel_entropies = RelEnt_allLayers(2, Density_per_layer)
    plt.plot(rel_entropies)
    plt.title('Layer 2')
    plt.ylabel('Relative entropy')
    plt.xlabel('layers')
    plt.savefig('Layer_2')
    '''

if __name__ == '__main__':
    main()

