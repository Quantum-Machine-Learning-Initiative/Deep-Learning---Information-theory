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
from qitbias import *

'''
x = np.arange(100)
y = x
t = np.ones(len(x))
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.scatter(x, y, c=t, cmap='viridis')
ax2.scatter(x, y, c=t, cmap='viridis_r')
plt.show()
'''

'''
x = np.arange(100)
y = x
t = np.ones(len(x))
fig, ax1 = plt.subplots()
ax1.scatter(x, y, c=t, cmap='viridis')
plt.show()
'''


def process_wbc(W, b, c, equal_nodes, layer_MIcoord, layer_Dcoord):
    N, num_layers, spins_per_layer, flat_W, num_w = CountSpins(W)
    rho = Ham2Density(BuildHamiltonian4(W, b, c, N, spins_per_layer))
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
            RelEnt_coord.append('vacio') #This is for the iterator in the zi
    for i,(E, I) in enumerate(zip(RelEnt_coord, MI_coord)):
        if equal_nodes:
            layer_Dcoord[i].append((E[0], E[1]))
        layer_MIcoord[i].append((I[0], I[1]))

              




def main():
    layers = 4 # Number of layers, including the visible layer of RBM
    equal_nodes = True ## True if all layers have same number of nodes. Important for relative entropy calculation
    epochs_per_layer = 32000
    snapshot_epoch = 100

    epochs_total = (layers-1)*epochs_per_layer
    #epochs_array = [snapshot_epoch*x*epochs_per_layer/ for x in range(layers)]
    n_files = (int(epochs_per_layer/snapshot_epoch) + 1) * (layers-1)
    #print(list(range(0,int((layer-1)*epochs_per_layer/snapshot_epoch))))
    label = []
    word = 'Layer '
    for i in range(layers):
        label.append(word + str(i))    



    
    layer_MIcoord = [[] for x in range(layers)]
    layer_Dcoord = [[] for x in range(layers)]



    '''
    for k in range(layers-1):
        for i in range(0, epochs_per_layer + 1, snapshot_epoch):
            string1 = '../rbm_weights_' + str(k) + '-' + str(i) + '.npy'
            string2 = '../rbm_bias_b_' + str(k) + '-' + str(i) + '.npy'
            string3 = '../rbm_bias_c_' + str(k) +'-' + str(i) + '.npy'
            #print(string1)
            W = np.load(string1)
            b = np.load(string2)
            c = np.load(string3)                 
            process_wbc(W, b, c, equal_nodes, layer_MIcoord, layer_Dcoord)
            print('layer: ' + str(k) + ' epoch: ' + str(i))
    print(layer_MIcoord[0])
    np.save('layer_MIcoord', layer_MIcoord)
    np.save('layer_Dcoord', layer_Dcoord)
    '''  
    layer_MIcoord = np.load('layer_MIcoord.npy')
    layer_Dcoord = np.load('layer_Dcoord.npy')


    #fig = plt.figure()
    #ax1 = fig.add_subplot(111)    
    #print(layer_Dcoord[-1])
    fig1, ax1 = plt.subplots()
    for k in range(layers):
        x = []
        y = []

 
        for e in layer_MIcoord[k]:
            x.append(e[0])
            y.append(e[1])
        t = np.linspace(0, epochs_per_layer*(layers-1), n_files)
        ax1.plot(x, y, linestyle='-', label=label[k], linewidth=1)
        ax1.scatter(x, y, c=t, cmap='jet')

        #for i, txt in enumerate(label):
        #    ax1.annotate(txt, (M_1_i[i], M_N_i[i]))
    #plt.colorbar()
    plt.title('Mi(Layer i , layer N) (y axis) v/s Mi(Layer i , layer N) (x axis)')
    plt.ylabel('Mi(Layer i , layer N)')
    plt.xlabel('Mi(layer 0 , layer i)')
    plt.legend(loc='upper right')
    ax=plt.gca()
    PCM=ax.get_children()[2]
    cbar = plt.colorbar(PCM, ax=ax)
    cbar.set_label('Epochs', rotation=0)
    plt.savefig('MI_history_qit.pdf')
    plt.close()


    fig2, ax2 = plt.subplots()
    for k in range(layers):
        x = []
        y = []
        for e in layer_Dcoord[k]:
            x.append(e[0])
            y.append(e[1])
        t = np.linspace(0, epochs_per_layer*(layers-1), n_files)
        ax2.plot(x, y, linestyle='-', label=label[k], linewidth=1)
        ax2.scatter(x, y, c=t, cmap='jet')

        #for i, txt in enumerate(label):
        #    ax1.annotate(txt, (M_1_i[i], M_N_i[i]))
    #plt.colorbar()
    plt.title('D(Layer i || layer N) (y axis) v/s D(layer 1 || layer i) (x axis)')
    plt.ylabel('D(Layer i || layer N)')
    plt.xlabel('D(layer 1 || layer i)')
    plt.legend(loc='upper right')
    ax=plt.gca()
    PCM=ax.get_children()[2]
    cbar = plt.colorbar(PCM, ax=ax)
    cbar.set_label('Epochs', rotation=0)
    plt.savefig('D_history_qit.pdf')
    plt.close()

    
    fig3, ax3 = plt.subplots()
    for k in range(layers):
        x = []
        y = []
        for e in layer_MIcoord[k]:
            x.append(e[0])
            y.append(e[1])   
        t = np.linspace(0, epochs_per_layer*(layers-1), n_files)
        ax3.plot([k]*len(x), x, linestyle='-', label=label[k], linewidth=1)
        ax3.scatter([k]*len(x), x, c=t, cmap='jet')   
    plt.title('Mi(Layer i , layer 1) (y axis) v/s Layers first to last (x axis)')
    plt.ylabel('Mi(Layer i , layer 1)')
    plt.xlabel('Layers first to last')
    plt.legend(loc='upper right')
    ax=plt.gca()
    PCM=ax.get_children()[2]
    cbar = plt.colorbar(PCM, ax=ax)
    cbar.set_label('Epochs', rotation=0) 
    plt.savefig('MI_1-i_qit.pdf')
    plt.close()


    fig4, ax4 = plt.subplots()
    for k in range(layers):
        x = []
        y = []
        for e in layer_MIcoord[k]:
            x.append(e[0])
            y.append(e[1])
        t = np.linspace(0, epochs_per_layer*(layers-1), n_files)
        ax4.plot([k]*len(y), y, linestyle='-', label=label[k], linewidth=1)
        ax4.scatter([k]*len(y), y, c=t, cmap='jet')   
    plt.title('Mi(Layer i , layer N) (y axis) v/s Layers first to last (x axis)')
    plt.ylabel('Mi(Layer i , layer N)')
    plt.xlabel('Layers first to last')
    plt.legend(loc='upper right')
    ax=plt.gca()
    PCM=ax.get_children()[2]
    cbar = plt.colorbar(PCM, ax=ax)
    cbar.set_label('Epochs', rotation=0) 
    plt.savefig('MI_N-i_qit.pdf')
    plt.close()


    fig5, ax5 = plt.subplots()
    for k in range(layers):
        x = []
        y = []
        for e in layer_Dcoord[k]:
            x.append(e[0])
            y.append(e[1])   
        t = np.linspace(0, epochs_per_layer*(layers-1), n_files)
        ax5.plot([k]*len(x), x, linestyle='-', label=label[k], linewidth=1)
        ax5.scatter([k]*len(x), x, c=t, cmap='jet')   
    plt.title('D(Layer 1 || layer i) (y axis) v/s Layers first to last(x axis)')
    plt.ylabel('D(Layer 1 || layer i)')
    plt.xlabel('Layers first to last')
    plt.legend(loc='upper right')
    ax=plt.gca()
    PCM=ax.get_children()[2]
    cbar = plt.colorbar(PCM, ax=ax)
    cbar.set_label('Epochs', rotation=0) 
    plt.savefig('D_1-i_qit.pdf')
    plt.close()

    fig6, ax6 = plt.subplots()
    for k in range(layers):
        x = []
        y = []
        for e in layer_Dcoord[k]:
            x.append(e[0])
            y.append(e[1])   
        t = np.linspace(0, epochs_per_layer*(layers-1), n_files)
        ax6.plot([k]*len(y), y, linestyle='-', label=label[k], linewidth=1)
        ax6.scatter([k]*len(y), y, c=t, cmap='jet')   
    plt.title('DD(Layer i || layer N) (y axis) v/s Layers first to last(x axis)')
    plt.ylabel('D(Layer i || layer N)')
    plt.xlabel('Layers first to last')
    plt.legend(loc='upper right')
    ax=plt.gca()
    PCM=ax.get_children()[2]
    cbar = plt.colorbar(PCM, ax=ax)
    cbar.set_label('Epochs', rotation=0) 
    plt.savefig('D_N-i_qit.pdf')
    plt.close()




    '''
    for k, i in enumerate(epochs_array):
        if i != epochs_total:
            string1 = '../rbm_weights_all-' + str(i) + '.npy'
            string2 = '../rbm_bias_b_all-' + str(i) + '.npy'
            string3 = '../rbm_bias_c_all-' + str(i) + '.npy'
            W = np.load(string1)
            b = np.load(string2)
            c = np.load(string3)
            print(string1)
            #process_wbc(W, b, c)
            for j in range(0,epochs_per_layer,snapshot_epoch):
                string1 = '../rbm_weights_' + str(k) + '-' + str(j) + '.npy'
                string2 = '../rbm_bias_b_' + str(k) + '-' + str(j) + '.npy'
                string3 = '../rbm_bias_c_' + str(k) +'-' + str(j) + '.npy'
                print(string1)
        if i == epochs_total:
            string1 = '../rbm_weights_all-' + str(i) + '.npy'
            string2 = '../rbm_bias_b_all-' + str(i) + '.npy'
            string3 = '../rbm_bias_c_all-' + str(i) + '.npy'
            W = np.load(string1)
            b = np.load(string2)
            c = np.load(string3)
    '''                        
            

if __name__ == '__main__':
    main()
