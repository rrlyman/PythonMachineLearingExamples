'''

takes an image input and trains it to make an image output

funnels down to a 'key' and then goes back up to image



'''
from tensorflow.compat import v1 as tf
import numpy as np
from collections import namedtuple
import datetime
import ocr_utils
from  n0_network  import  base_network as b_network

class network(b_network):
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
        # are tensors.  The display name on the Chrome graph are 'y_m_label', 'self._x_image, 
        # x_upper_case etc.
    
    
        Place_Holders = namedtuple('Place_Holders', truthed_features.feature_names)   
        self._ph = Place_Holders(*lst) # unpack placeholders into named Tuple
        self._keep_prob = tf.placeholder(dtype,name='keep_prob')    
        self._nRows = truthed_features.num_rows #image height
        self._nCols = truthed_features.num_columns #image width    
        nSections = 10
    
        in_out_width = self._nRows*self._nCols
        internal_width = int(in_out_width/4)
        w = list(range(nSections*3))
        b = list(range(nSections*3))
        h = list(range(nSections*3+1))        
        nFc1 = 2048      # size of fully connected layer

        nTarget = truthed_features.feature_width[0]  # the number of one_hot features in the target, 'm_label'
            
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
                
        def shapeOuts(n):
            print ('n={}, hin={},w={}, b={} ,hout={}\n'.format(n, h[n].shape, w[n].shape, b[n].shape, h[n+1]._shape))
             
        def section(n):
            with tf.name_scope('section_'+str(n)+'_0') as scope:     
                w[n]=weight_variable([in_out_width, internal_width],dtype)
                b[n]=bias_variable([internal_width],dtype)  
                h[n+1] = tf.nn.relu(tf.matmul(h[n], w[n]) + b[n])
                shapeOuts(n)
                 
            with tf.name_scope('section_'+str(n)+'_1') as scope:  
                w[n+1]=weight_variable([internal_width, internal_width],dtype)
                b[n+1]=bias_variable([internal_width],dtype)     
                               
                h[n+2]=tf.nn.relu(tf.matmul(h[n+1], w[n+1]) + b[n+1])
                shapeOuts(n+1)                  
                                 
            with tf.name_scope('section_'+str(n)+'_2') as scope:  
                w[n+2]=weight_variable([internal_width, in_out_width],dtype)
                b[n+2]=bias_variable([in_out_width],dtype)   
                z= tf.nn.relu(tf.matmul(h[n+2], w[n+2]) + b[n+2])
                h[n+3]= tf.add(z   ,h[n]) #n+3   
                          
                print('z shape ={}'.format(z._shape)) 
                shapeOuts(n+2)                  
            return    
                   
        def computeSize(s,tens):
            sumC = 1
            tShape = tens.get_shape()
            nDims = len(tShape)
            for i in range(nDims):
                sumC *= tShape[i]
            print ('\t{}\t{}'.format(s,sumC),flush=True)
            return sumC
                         
        """# ==============================================================================        
        Build sectional network
         
        """# ==============================================================================      
        h[0]= self._ph[1]
        for i in range(nSections):
            section(3*i)
                 
        """# ==============================================================================        
        Dropout
         
        """# ==============================================================================
        self._keep_prob = tf.placeholder(dtype,name='keep_prob')
         
        with tf.name_scope("drop") as scope:
            h_fc2_drop = tf.nn.dropout(h[nSections*3], self._keep_prob)
         
        """# ==============================================================================
         
        Readout Layer
         
        """# ==============================================================================
        with tf.name_scope("softmax") as scope:
            w_fc3 = weight_variable([in_out_width, nTarget],dtype)
            b_fc3 = bias_variable([nTarget],dtype)    
            y_conv=tf.nn.softmax(tf.matmul(h_fc2_drop, w_fc3) + b_fc3)
         
        print ('network size:',flush=True)
        total = 0
        for i in range(nSections*3):
            total = total + computeSize("w{}".format(i),w[i])
        total = total + computeSize ("b_fc3",b_fc3) + \
            computeSize ("w_fc3",w_fc3)      
         
        print('\ttotal\t{}'.format(total),flush=True)
         
            
        with tf.name_scope("reshape_self._x_image") as scope:        
            self._x_image = tf.reshape(self._ph.image, [-1,self._nCols,self._nRows,1])
         
        with tf.name_scope("xent") as scope:
            # 1e-8 added to eliminate the crash of training when taking log of 0
            cross_entropy = -tf.reduce_sum(self._ph[0]*tf.log(y_conv+1e-8))
            ce_summ = tf.summary.scalar("cross entropy", cross_entropy)
                 
        with tf.name_scope("train") as scope:
            self._train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
             
        with tf.name_scope("test") as scope:        
            self._correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(self._ph[0],1))
            self._prediction = tf.argmax(y_conv,1)            
         
            self._accuracy = tf.reduce_mean(tf.cast(self._correct_prediction, dtype))
            accuracy_summary = tf.summary.scalar("accuracy", self._accuracy)    

        """# ==============================================================================
        
        Start TensorFlow Interactive Session
        
        """# ==============================================================================          
        
        self._sess.run(tf.initialize_all_variables())  
        self._merged = tf.summary.merge_all()
        tm = ""
        tp = datetime.datetime.now().timetuple()
        for i in range(4):
            tm += str(tp[i])+'-'
        tm += str(tp[4])    
        
        # To see the results in Chrome, 
        # Run the following in terminal to activate server.
        # tensorboard --logdir '/tmp/ds_logs/'
        # See results on localhost:6006 
        
        self._writer = tf.summary.FileWriter("/tmp/ds_logs/"+ tm, self._sess.graph)
        
    def computeSize(s,tens):
        sumC = 1
        tShape = tens.get_shape()
        nDims = len(tShape)
        for i in range(nDims):
            sumC *= tShape[i].value
        print ('\t{}\t{}'.format(s,sumC),flush=True)
        return sumC
        

        
    def __exit__(self, exc_type, exc_value, traceback):
        tf.reset_default_graph() # only necessary when iterating through fonts
        self._sess.close()  
            

    def reset_graph(self):
        tf.reset_default_graph() # only necessary when iterating through fonts
        self._sess.close()  
        
#      
#     def encode(self):  
#         
#         return key
#         
#     def decode(self, key):            