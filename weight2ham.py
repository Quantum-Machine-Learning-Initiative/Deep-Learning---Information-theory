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


def spin2front(rho, N, k):
    rho_sparse = sp.sparse.csr_matrix(rho)
    permu = sp.sparse.csr_matrix(np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]]))
    M = sp.sparse.kron(sp.sparse.eye(2**(k-1)), sp.sparse.kron(permu, sp.sparse.eye(2**(N-k-1))))
    for i in range(1,k):
        M = sp.sparse.kron(sp.sparse.kron(sp.sparse.eye(2**(k-1-i)), permu), sp.sparse.eye(2**(N-k-1+i))).dot(M)
    DensityMatrix = (M.dot(rho_sparse)).dot(M.transpose())
    return sp.sparse.csr_matrix.toarray(DensityMatrix)

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
    return -H

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
def Apply_n_translations(rho, n, num_spins):
    M = sp.sparse.csr_matrix(rho)
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


def Layer2front(DensityMatrix, N, spins_per_layer, layer_p):
    #NOTE: layer_p > layer_q
    if layer_p == 0:
        return DensityMatrix
    for i in range(int(spins_per_layer[layer_p])):
        spin_k = int(np.array(spins_per_layer[:layer_p + 1]).sum() - 1)
        DensityMatrix = spin2front(DensityMatrix, N, spin_k)
    return DensityMatrix
    

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


def ObtainPartialJoint(DensityMatrix, spins_per_layer, N, layer_p, layer_q, Density_per_layer):
    # NOTE: layer_p => layer_q
    if layer_p == layer_q:
        return Density_per_layer[layer_p]
    spins = int(spins_per_layer[layer_p] + spins_per_layer[layer_q])
    DensityMatrix = Layer2front(DensityMatrix, N, spins_per_layer, layer_p)
    layer_q += 1
    DensityMatrix = Layer2front(DensityMatrix, N, spins_per_layer, layer_q)
    DensityMatrix = PartialTrace(spins, N, DensityMatrix)
    return DensityMatrix




'''
def ObtainPartialJoint(rho, spins_per_layer, N, layer_p, layer_q)
    # NOTE: layer_p > layer_q
    spins = spins_per_layers[layer_p] + spins_per_layers[layer_q]
    if layer_p+1 == spins_per_layer.shape:
        DensityMatrix = Apply_n_translations(rho, 1, N)
        layer_p = 0
        layer_q += 1
        if layer_q+1 == spins_per_layer.shape:
            DensityMatrix = Apply_n_translations(rho, 1, N)
            layer_p = 1
            layer_q = 0
            DensityMatrix = PartialTrace(spins, N, DensityMatrix)
            return DensityMatrix 
        else:
            DensityMatrix = Apply_n_translations(rho, layer_q, N)
            
                    
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
    plt.savefig('Info_D_plot_test.pdf')
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
    plt.savefig('Info_MI_plot_test.pdf')
    plt.close()
    
    plt.figure(3)
    plt.title('Mi(Layer 1 || i) v/s Layers')
    plt.ylabel('Mi(Layer 1 || layer i)')
    plt.xlabel('Layers (first to last)')
    plt.scatter(np.arange(len(M_1_i)).tolist(), M_1_i) 
    plt.savefig('MI_layer1_test.pdf')
    plt.close()
    
    plt.figure(4)
    plt.title('Mi(Layer i , N) v/s Layers')
    plt.ylabel('Mi(Layer i , layer N)')
    plt.xlabel('Layers (first to last)')
    a = np.arange(len(M_N_i)).tolist()
    #a.reverse()
    plt.scatter(a, M_N_i) 
    plt.savefig('MI_layerN_test.pdf')
    plt.close()
             
    
    
def main():
    W = np.load('rbm_weights.npy')
    equal_nodes = False ## True if all layers have same number of nodes. Important for relative entropy calculation
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
    #print(Density_per_layer[1])
    #print(Density_per_layer[2])
    #Obtain_MI(rho, Density_per_layer, num_layers)

    
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
    print(MI_coord)
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
    #test()

