"""# ==========================================================================

# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

This sample program is a modified version of the Google mnist convolutional 
network tutorial example.  See the mnist tutorial in www.tensorflow.org 

This graph has multiple sections  3 layers each, 400 100 400 followed
by a fully connected layer.

see tensor_flow_graph.png
"""# ==============================================================================
import ocr_utils
import datetime
from collections import namedtuple
import numpy as np
import pandas as pd
import n1_residual3x4 as nnetwork      
from tensorflow.compat import v1 as tf
dtype = np.float32
#with tf.device('/GPU:0'):
#with tf.device('/cpu:0'): 
       
    
if True:
    # single font train
    
    # examples
    # select only images from 'OCRB'  scanned font
    # input_filters_dict = {'font': ('OCRA',)}
    
    # select only images from 'HANDPRINT'  font
    #input_filters_dict = {'font': ('HANDPRINT',)}
    
    # select only images from 'OCRA' and 'OCRB' fonts with the 'scanned" fontVariant
    # input_filters_dict = {'font': ('OCRA','OCRB'), 'fontVariant':('scanned',)}
    
    # select everything; all fonts , font variants, etc.
    # input_filters_dict = {}
    
    # select the digits 0 through 9 in the E13B font
    # input_filters_dict = {'m_label': range(48,58), 'font': 'E13B'}
    
    # select the digits 0 and 2in the E13B font
    # input_filters_dict = {'m_label': (48,50), 'font': 'E13B'}
    
    # output the character label, image, italic flag, aspect_ratio and upper_case flag
    # output_feature_list = ['m_label_one_hot','image','italic','aspect_ratio','upper_case']    
    
    # output only the character label and the image
    # output_feature_list = ['m_label_one_hot','image'] 
    
    #   identify the font given the input images
    #output_feature_list = ['font_one_hot','image','italic','aspect_ratio','upper_case']   

    # train the digits 0-9 for all fonts
    #input_filters_dict = {'m_label': range(48,58)}
    input_filters_dict = {'font':'ARIAL','m_label': list(range(48,58))+list(range(65,91))+list(range(97,123))}    
    #input_filters_dict = {}    
    output_feature_list = ['m_label_one_hot','image']    
 
    """# ==============================================================================
    
    Train and Evaluate the Model
    
    """# ==============================================================================
    ds = ocr_utils.read_data(input_filters_dict = input_filters_dict, 
                                output_feature_list=output_feature_list,
                                test_size = .1,
                                engine_type='tensorflow',dtype=dtype)    
    nn = nnetwork.network(ds.train)
    nn.fit( ds.train,  nEpochs=5000)  
    nn.test(ds.test)
      
#     train_a_font(input_filters_dict,  output_feature_list, nEpochs = 50000)    
    
else:
    # loop through all the fonts and train individually

    # pick up the entire list of fonts and font variants. Train each one.
    df1 = ocr_utils.get_list(input_filters_dict={'font': ()})      
    
    import pprint as pprint
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(df1)
   
    output_feature_list = ['m_label_one_hot','image','italic','aspect_ratio','upper_case','font_one_hot']
    
    # Change nEpochs to 5000 for better results
    for l in df1:
        input_filters_dict= {'font': (l[0],)}       
        train_a_font(input_filters_dict,output_feature_list, nEpochs = 500) 
    
    
print ('\n########################### No Errors ####################################')

