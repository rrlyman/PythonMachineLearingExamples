from tensorflow.compat import v1 as tf
#import tf
import numpy as np
from collections import namedtuple
import datetime
from  n0_network  import  base_network as b_network
import ocr_utils

class network( b_network):
    ''' definition of the network
    '''
       
    def __init__(self, truthed_features, dtype=np.float32):
        self._sess = tf.InteractiveSession()  
      
        lst = []
        extra_features_width = 0 # width of extra features
        
        """# ==============================================================================
        
        Placeholders
        
        Compute the size of various layers 
        
        Create a tensorflow Placeholder for each feature of data returned from the
        dataset
        
        """# ==============================================================================        
        
        for i,nm in enumerate(truthed_features.feature_names):
            
            # features[0], is always the target. For instance it may be m_label_one_hot 
            # the second features[1] is the 'image' that is passed to the convolution layers 
            # Any additional features bypass the convolution layers and go directly 
            # into the fully connected layer.  
            
            # The width of the extra features is calculated in order to allocate 
            # the correct widths of weights,  # and inputs 
            # names are assigned to make the look pretty on the tensorboard graph.
            
            if i == 0:
                nm = 'y_'+nm
            else:
                nm = 'x_'+nm
            if i>1:
                extra_features_width += truthed_features.feature_width[i]
            lst.append(tf.placeholder(dtype, shape=[None, truthed_features.feature_width[i]], name=nm))
            
        # ph is a named tuple with key names like 'image', 'm_label', and values that
        # are tensors.  The display name on the Chrome graph are 'y_m_label', 'x_image, 
        # x_upper_case etc.
    
    
        Place_Holders = namedtuple('Place_Holders', truthed_features.feature_names)   
        self._ph = Place_Holders(*lst) # unpack placeholders into named Tuple
        self._keep_prob = tf.placeholder(dtype,name='keep_prob')    
        self._nRows = truthed_features.num_rows #image height
        self._nCols = truthed_features.num_columns #image width    
        nFc0 = 2048      # size of fully connected layer
        nFc1 = 100      # size of fully connected layer        
        nFc2 = self._nRows*self._nCols      # size of fully connected layer              
        nConv1 = 32     # size of first convolution layer
        nConv2 = 64     # size of second convolution layer
        nTarget = truthed_features.feature_width[0]  # the number of one_hot features in the target, 'm_label'    
        n_h_pool2_outputs = int(self._nRows/4) * int(self._nCols/4) * nConv2 # second pooling layer 
        n_h_pool2_outputsx = n_h_pool2_outputs + extra_features_width # fully connected
            
        """# ==============================================================================
        
        Build a Multilayer Convolutional Network
        
        Weight Initialization
        
        """# ==============================================================================
        
        def weight_variable(shape, dtype):
            initial = tf.truncated_normal(shape, stddev=0.1,dtype=dtype)
            return tf.Variable(initial)
        
        def bias_variable(shape, dtype):
            initial = tf.constant(0, shape=shape, dtype=dtype)
            return tf.Variable(initial)   
        
        """# ==============================================================================
        
        Convolution and Pooling
        
        keep our code cleaner, let's also abstract those operations into functions.
        
        """# ==============================================================================
        
        def conv2d(x, W):
            return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
        
        def max_pool_2x2(x):
            return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                                strides=[1, 2, 2, 1], padding='SAME')
        
        """# ==============================================================================
        
        First Convolutional Layer
        
        """# ==============================================================================
        with tf.name_scope("w_conv1") as scope:
            W_conv1 = weight_variable([5, 5, 1, nConv1],dtype)
            b_conv1 = bias_variable([nConv1],dtype)    
        
        with tf.name_scope("reshape_x_image") as scope:
            self._x_image = tf.reshape(self._ph.image, [-1,self._nCols,self._nRows,1])
        
        image_summ = tf.summary.image("x_image", self._x_image)
        
        """# ==============================================================================
        
        We then convolve x_image with the weight tensor, add the bias, apply the ReLU function,
         and finally max pool.
        
        """# ==============================================================================
        
        with tf.name_scope("convolve_1") as scope:
            h_conv1 = tf.nn.relu(conv2d(self._x_image, W_conv1) + b_conv1)
            
        with tf.name_scope("pool_1") as scope:    
            h_pool1 = max_pool_2x2(h_conv1)
        
        """# ==============================================================================
        
        Second Convolutional Layer
        
        In order to build a deep network, we stack several layers of this type. The second 
        layer will have 64 features for each 5x5 patch.
        
        """# ==============================================================================
        
        with tf.name_scope("convolve_2") as scope:
            W_conv2 = weight_variable([5, 5, nConv1, nConv2],dtype)
            b_conv2 = bias_variable([64],dtype)
            h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)  
             
        with tf.name_scope("pool_2") as scope:
            h_pool2 = max_pool_2x2(h_conv2)
        
        """# ==============================================================================
        
        Densely Connected Layer
        
        Now that the image size has been reduced to 7x7, we add a fully-connected layer 
        with neurons to allow processing on the entire image. We reshape the tensor 
        from the pooling layer into a batch of vectors, multiply by a weight matrix, add 
        a bias, and apply a ReLU.
        
        """# ==============================================================================
        
        with tf.name_scope("W_fc1_b") as scope:
            W_fc0 = weight_variable([n_h_pool2_outputsx, nFc0],dtype)
            b_fc0 = bias_variable([nFc0],dtype)
                
            h_pool2_flat = tf.reshape(h_pool2, [-1, n_h_pool2_outputs])
            
            # append the features, the 2nd on, that go directly to the fully connected layer
            for i in range(2,truthed_features.num_features ):
                h_pool2_flat = tf.concat(1, [h_pool2_flat, self._ph[i]])  
            h_fc0 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc0) + b_fc0)
            
        """# ==============================================================================
            
            Densely Connected Layer 1
            
            We add a fully-connected layer 
            with neurons to allow processing on the entire image. We reshape the tensor 
            from the pooling layer into a batch of vectors, multiply by a weight matrix, add 
            a bias, and apply a ReLU.
        
        """# ==============================================================================  
              
        with tf.name_scope("W_fc1_b") as scope:
            W_fc1 = weight_variable([nFc0, nFc1],dtype)
            b_fc1 = bias_variable([nFc1],dtype)
            
            h_fc1 = tf.nn.relu(tf.matmul(h_fc0, W_fc1) + b_fc1)    
        
        """# ==============================================================================
        
        Densely Connected Layer 2
        
        We add a fully-connected layer 
        with neurons to allow processing on the entire image. We reshape the tensor 
        from the pooling layer into a batch of vectors, multiply by a weight matrix, add 
        a bias, and apply a ReLU.
        
        """# ==============================================================================
        
        with tf.name_scope("W_fc2_b") as scope:
            W_fc2 = weight_variable([nFc1, nFc2],dtype)
            b_fc2 = bias_variable([nFc2],dtype)
            
            h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)
                    
        """# ==============================================================================
        
        Dropout
        
        """# ==============================================================================
        
        
        with tf.name_scope("drop") as scope:
            h_fc2_drop = tf.nn.dropout(h_fc2, self._keep_prob)
        
        """# ==============================================================================
        
        Readout Layer
        
        """# ==============================================================================
        with tf.name_scope("softmax") as scope:
            W_fc3 = weight_variable([nFc2, nTarget],dtype)
            b_fc3 = bias_variable([nTarget],dtype)    
            y_conv=tf.nn.softmax(tf.matmul(h_fc2_drop, W_fc3) + b_fc3)    
        
        with tf.name_scope("xent") as scope:
        
            # 1e-8 added to eliminate the crash of training when taking log of 0
            self._cross_entropy = -tf.reduce_sum(self._ph[0]*tf.log(y_conv+ 1e-8  ))
            #cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
             #   logits, labels, name='xentropy')            
            ce_summ = tf.summary.scalar("cross entropy", self._cross_entropy)
            
        with tf.name_scope("reshape_x_image2") as scope:
            self._x_image2 = tf.reshape(self._ph[0], [-1,int(self._nCols/2),int(self._nRows/2),1])
        
        image_summ2 = tf.summary.image("x_image2", self._x_image2)            
            
        with tf.name_scope("train") as scope:
            self._train_step = tf.train.AdamOptimizer(1e-4).minimize(self._cross_entropy)
            #self._train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)            
    
        with tf.name_scope("test") as scope:        
            self._correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(self._ph[0],1))
            self._prediction = tf.argmax(y_conv,1)
        
            self._accuracy = tf.reduce_mean(tf.cast(self._correct_prediction, dtype))
            accuracy_summary = tf.summary.scalar("accuracy", self._accuracy)    
        """# ==============================================================================
        
        Start TensorFlow Session
        
        """# ==============================================================================      
    
        self._sess.run(tf.initialize_all_variables())  
        self._merged = tf.summary.merge_all()
        tm = ""
        tp = datetime.datetime.now().timetuple()
        for i in range(4):
            tm += str(tp[i])+'-'
        tm += str(tp[4])    
        self._writer = tf.summary.FileWriter("/tmp/ds_logs/"+ tm, self._sess.graph)
        
        def computeSize(s,tens):
            sumC = 1
            tShape = tens.get_shape()
            nDims = len(tShape)
            for i in range(nDims):
                sumC *= tShape[i]
            print ('\t{}\t{}'.format(s,sumC),flush=True)
            return sumC
                
        print ('network size:',flush=True)        
        total = computeSize("W_fc0",W_fc0)+ \
        computeSize ("b_fc0",b_fc0) + \
        computeSize ("W_conv1",W_conv1) + \
        computeSize ("b_conv1",b_conv1) + \
        computeSize ("W_conv2",W_conv2) + \
        computeSize ("b_conv2",b_conv2) + \
        computeSize ("W_fc0",W_fc0) + \
        computeSize ("b_fc0",b_fc0) + \
        computeSize ("W_fc1",W_fc1) + \
        computeSize ("b_fc1",b_fc1) + \
        computeSize ("W_fc2",W_fc2) + \
        computeSize ("b_fc2",b_fc2)  
            
        print('\ttotal\t{}'.format(total),flush=True)
        
    def reset_graph(self):
        tf.reset_default_graph() # only necessary when iterating through fonts
        self._sess.close()  
        
    def test2(self, truthed_data,  title = ''): 

        # assign feature data to each placeholder

        output_images = np.empty((0,int(self._nRows/2),int(self._nCols/2)))
        input_images = np.empty((0,int(self._nRows),int(self._nCols)))            
        test_accuracy=0
        m=0
   
        for i in range(int(len(truthed_data.features[0])/100)):
        
            batch = truthed_data.next_batch(100)
            # assign feature data to each placeholder
            # the batch list is returned in the same order as the features requested
            feed = {self._keep_prob: 1.0}
            for j in range(truthed_data.num_features):
                feed[self._ph[j]] = batch[j]  
                

            result = self._sess.run([self._accuracy, self._x_image, self._correct_prediction, self._x_image2], feed_dict=feed)    
            
            test_accuracy += result[0]
            input_images  = np.append(input_images, result[1][:,:,:,0],axis=0) 
            output_images  = np.append(output_images, result[3][:,:,:,0],axis=0)         
            m += 1
        try:        
            print ("test accuracy {} for : {}".format(test_accuracy/m, title),flush=True)       
            ocr_utils.montage(input_images,title='TensorFlow {} Input Images'.format(title))    
            ocr_utils.montage(output_images,title='TensorFlow {} Output Images'.format(title))             
        except:  
            if m==0:
                print ("test accuracy 1",flush=True)
            else:                                                                    
                print ("test accuracy {}".format(test_accuracy/m),flush=True)  
                ocr_utils.montage(output_images,title='TensorFlow Output Images')         
                ocr_utils.montage(input_images,title='TensorFlow Input Images')        
     
    def fit_entropy(self, truthed_data, nEpochs=5000):    

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
                result = self._sess.run([self._merged, self._cross_entropy ], feed_dict=feed)    
                summary_str = result[0]
 
                self._writer.add_summary(summary_str, i)
                train_entropy = result[1]  
                if train_entropy >= (2000 ):
                    perfect_count=10;
                else:
                    perfect_count -= 1
                    if perfect_count==0:
                        break;   
                    
                print ("step %d, training entropy %g"%(i, train_entropy),flush=True)
            self._sess.run(self._train_step,feed_dict=feed)
            

    
                  