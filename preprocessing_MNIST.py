from __future__ import print_function
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

class PreMnist:

    '''function that transform a batch of mnist entry from grayscale to B/W
     @param 1 is a Boolean: what set to load 1 -> Training; 0 -> Test
     @param 2 is an Integer: # elements
    '''
    def ET(self, trainingFlag, batch_size, printFlag=False):
    
      # Import data
      mnist = input_data.read_data_sets('./data_set', one_hot=True)
      
      batch_xs, label = mnist.train.next_batch(batch_size)  
      
      images, pixels = batch_xs.shape
      
      ret = np.array([np.append(a, b) for(a, b) in zip(batch_xs, label)])
    
      ret = ret.astype(np.float32)    
    
      for im in range(0, images):
        for pix in range(0, pixels):
          if batch_xs[im, pix] > 0.5:
              ret[im, pix] = 1
          else:
              ret[im, pix] = 0
        if printFlag:      
          self.printSet(batch_xs[im], ret[im])
    
      return ret, label
    
    def printSet(self, image1, image2, delta=10):
        
        length = image1.size
        
        print(image1.shape)
        
        print("first")
        for pix in range(0, length):
            if pix % 28 == 0:
                print('\n')
            if image1[pix] > 0.5:    
                print(np.around(image1[pix], decimals=1), end='')
            else: 
                print("   ", end='')
    
        print("second")
        for pix in range(0, length):
            if pix % 28 == 0:
                print('\n')
            if image2[pix] > 0:    
                print(np.around(image2[pix],decimals=1), end='')    
            else: 
              print("   ", end='')
        for p in range(delta):
          print(np.around(image2[p+length],decimals=1), end='')    

'''
#we get the mnist batch as a parameter
def main():
  pr=PreMnist()
  batch, labels = PreMnist.ET(pr, 1, 1, True)

  PreMnist.printSet(pr, batch[0], batch[0], 0)

if __name__ == '__main__':
  main()
'''
