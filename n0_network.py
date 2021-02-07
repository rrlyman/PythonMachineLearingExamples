#import tensorflow as tf  
from tensorflow.compat import v1 as tf
tf.compat.v1.disable_eager_execution()
import numpy as np
from collections import namedtuple
import datetime
import ocr_utils

class base_network(object):
    ''' definition of the network
    '''

        
        

    def fit(self, truthed_data, nEpochs=5000):    

        perfect_count=10
        for i in range(nEpochs):
        
            batch = truthed_data.next_batch(100)
            # assign feature data to each placeholder
            # the batch list is returned in the same order as the features requested
            feed = {self._keep_prob: 0.5}
            for j in range(truthed_data.num_features):
                feed[self._ph[j]] = batch[j]  
                
            if i%100 == 0:

                feed[self._keep_prob] = 1.0
                result = self._sess.run([self._merged, self._accuracy ], feed_dict=feed)    
                summary_str = result[0]
 
                self._writer.add_summary(summary_str, i)
                train_accuracy = result[1]   
                if train_accuracy <= (1.0 - 1e-5  ):
                    perfect_count=10;
                else:
                    perfect_count -= 1
                    if perfect_count==0:
                        break;  
                    
                print ("step %d, training accuracy %g"%(i, train_accuracy),flush=True)
            self._sess.run(self._train_step,feed_dict=feed)
            

    


    def test(self, truthed_data,  title = ''): 

        # assign feature data to each placeholder
        error_images = np.empty((0,self._nRows,self._nCols))
            
        test_accuracy=0
        m=0
   
        for i in range(int(len(truthed_data.features[0])/100)):
        
            batch = truthed_data.next_batch(100)
            # assign feature data to each placeholder
            # the batch list is returned in the same order as the features requested
            feed = {self._keep_prob: 1.0}
            for j in range(truthed_data.num_features):
                feed[self._ph[j]] = batch[j]  
                

            result = self._sess.run([self._accuracy, self._x_image, self._correct_prediction], feed_dict=feed)    
            
            test_accuracy += result[0]
            error_images = np.append(error_images, result[1][:,:,:,0][result[2]==False],axis=0)
            m += 1
        try:        
            print ("test accuracy {} for : {}".format(test_accuracy/m, title),flush=True)       
            ocr_utils.montage(error_images,title='TensorFlow {} Error Images'.format(title))  
        except:  
            if m==0:
                print ("test accuracy 1",flush=True)
            else:                                                                    
                print ("test accuracy {}".format(test_accuracy/m),flush=True)  
                ocr_utils.montage(error_images,title='TensorFlow Error Images') 
            
         
    def predict(self, truthed_features): 
        feed={self._keep_prob: 1.0}
        # assign feature data to each placeholder
        error_images = np.empty((truthed_features.num_rows,truthed_features.num_columns))
            
        test_accuracy=0
        m=0
          
        for j in range(1,truthed_features.num_features):
             feed[self._ph[j]] = truthed_features.features[j]
        result = self._sess.run([self._prediction], feed_dict=feed)    
    
        return result[0]
     
                  