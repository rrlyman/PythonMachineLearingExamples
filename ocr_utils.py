'''
Created on Jun 20, 2016
ocr_utils

Utility programs for accessing the database of character images.

Contains database access functions:
    load_E13B: special purpose function to return computed column Sums from E13B font
    read_data: general purpose function for returning a list of features from
        the database
    get_list: returns a list of the types of items in the database

Miscellaneous Plot Functions:

    see the show_plot global variable below
    
    montage: plots a grid of images 
    plot_decision_regions: shows the shape of a classifier's decision regions 
        and a scatter plot of the data from Python Machine Learning
    scatter_plot shows a scatter plot


@author: richard
'''

######################################################
show_plot = False   #set True to show plot on screen, set False to save to file
plot_dir = '/tmp/plots'
#####################################################

extension = '.jpg'

##############################################################################
default_zip_file = "fonts.zip"   #small data set
#default_zip_file = 'fonts_all.zip' #for the big data set
##############################################################################

import numpy as np
import pandas as pd
import math
#from pandas.io.common import ZipFile
from zipfile import ZipFile
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import sys
import os
import subprocess
from PIL import Image
import io


def report(blocknr, blocksize, size):
    current = blocknr*blocksize
    print("{0:.2f}%".format(100.0*current/size),end='\r')

import urllib.request as urllib2

def downloadFile(url):
 
    fname = url.split('/')[-1]
    urllib2.urlretrieve(url, fname, report)
        
def read_file(pathName, input_filters_dict, random_state=None):
    '''
    Reads the .csv file containing the labeled data images.
    
    Parameters
    ------------
        pathName :path name of the zip file containing all the training data
        
        input_filters_dict: ['font', font_name] 
            a font_name is a string or tuple containing a list of the fonts
                to be read from the database or
                empty to return all fonts
            or a string containing a single font name
            or None which will return all fonts.
            
        random_state: None for random seed chosen by the system
            or integer seed for the random seed for repeatable calls
            
    Returns   
    ------------
    a pandas shuffled Dataframe containing the columns from the csv file 
    
    Note: The file to be read is a .zip file that in turn contains .csv
        files. Each .csv file contains images for a given font.
        This make access to a font, such as OCRA, fast because only one
        .csv file needs to be accessed.
     '''    

    if os.path.exists(pathName)==False:
        print('{} does not exist!  Downloading it from the web'.format(default_zip_file), flush=True)            
        downloadFile('http://lyman.house/download/{}'.format(default_zip_file))
        #downloadFile('http://lyman.house/download/fonts_chinese.zip')        

    try :
        rd_font = input_filters_dict['font']
        if isinstance(rd_font, str): 
            rd_font = (rd_font,)            
    except:
        rd_font = ()   

    
    with ZipFile(pathName, 'r') as myzip:
        if len(rd_font) == 0:
            names = myzip.namelist()            
            print ('\nreading all files...please wait')
            df = pd.concat(apply_column_filters(pd.read_csv(myzip.open(fname,'r')), input_filters_dict) for fname in names)     
        else:
            try:
                df = pd.concat(apply_column_filters(pd.read_csv(myzip.open(font+".csv",'r')), input_filters_dict) for font in rd_font)   
            except:
                raise ValueError('Could not find font file {} in the zip file'.format(rd_font))
        myzip.close()
    assert df.size >0
        
    return  df.sample(frac=1, random_state=random_state)

def get_list(pathName=default_zip_file,input_filters_dict={}): 
    '''
    Read the entire database of fonts to find out what unique entries are 
    available.
    
    Parameters
    ---------------
        pathName : the path of the zip file containing the database of characters
        input_filters_dict : a dictionary containing columns in the .csv file to
            be extracted.  keys = column heading, values = value to be 
            allowed in that column.  Returns an entire column if a key is not
            provided for it.
        
    Returns
    --------------
        a dataframe of all the all the unique lines in the dataset.
        
    Example:
    --------------    
    print(ocr_utils.get_list(columns=('font','fontVariant')))    

    '''
    
    # speed up list if only the font list is needed 
    try:
        if (len(input_filters_dict)==1) and (len(input_filters_dict['font'])==0):
            with ZipFile(pathName, 'r') as myzip:
                y = sorted(myzip.namelist())
            for i,l in enumerate(y):
                y[i] = [l.replace('.csv','')]
            return y
    except:
        pass
    
    df = read_file(pathName,input_filters_dict)      
    df = df.loc[:,:'r0c0']                              
    keys=list(input_filters_dict.keys())
    df = df[keys]
    df= df.drop_duplicates()
    return df
    
       
class TruthedCharacters(object):
    """TrainedImages database.
    Holds the training features and size information

    """
    def __init__(self,  features, output_feature_list, one_hot_map, engine_type,h,w, dtype):

        self._num_examples = features[0].shape[0]
        self._nRows = h
        self._nCols = w
        self._features = features                   # list of features
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._feature_names = output_feature_list   # list of names of features
        
        for i,nm in enumerate(self._feature_names):
            if nm in self._feature_names[i+1:]:
                self._feature_names[i+1] = nm + "_"+ str(i)
        self._num_features = len(features)
        self._one_hot_map = one_hot_map             # list 0 for non one-hots, or # of one_hots 
        self._engine_type= engine_type        
        self._dtype = dtype
        
        self._feature_width=[]
        for i in range(self._num_features ):
            try:
                if one_hot_map[i] == 0:
                    self._feature_width += [features[i].shape[1]]
                else:
                    self._feature_width += [one_hot_map[i]]
            except:
                self._feature_width += [1]             

        
    @property
    def num_features(self):
        return self._num_features
    
    @property
    def feature_width(self):
        return self._feature_width
          # fixup for formats required by various engines   
        # features that are straight from the .CSV file, without 
        # modifications for one-hot or scaling fall here.
        # tensorflow requires a shape (:,1), thus the reshaping
        
    def engine_conversion(self,t1,colName):
        if self._engine_type=='tensorflow' and len(t1.shape)==1:       
            t1=np.reshape(t1,(-1,1))
            
        if self._engine_type=='theano' and colName=='image':
            t1=np.reshape(t1,(-1,1,self._nRows,self._nCols))
        return t1
    
    def get_features(self, i, start, end):
        '''
        memory saving version of features[i][start:end]
        '''
        t1 = self._features[i][start:end]
        n_hots = self._one_hot_map[i]

        if n_hots==0:
            rtn=self.engine_conversion(t1, self._feature_names[i])
        else:
            rtn= self.engine_conversion(np.eye(n_hots, dtype=self._dtype )[t1], self._feature_names[i])            
        return rtn    
    
    @property
    def features(self):
        # wait until last moment to compute one_hots to save memory        
        rtn = []
        for t1, nm, n_hots in zip(self._features, self._feature_names,  self._one_hot_map):
            if n_hots==0:
                rtn.append(self.engine_conversion(t1, nm) )
                #assert(np.all(rtn[-1]==t1))
            else:
                rtn.append( self.engine_conversion(np.eye(n_hots, dtype=self._dtype )[t1], nm)   )             
        return rtn
    
    @property
    def feature_names(self):
        return self._feature_names  
          
    @property
    def num_rows(self):
        return self._nRows
    
    @property
    def num_columns(self):
        return self._nCols

    def next_batch(self, batch_size):
        """
        Get the next `batch_size` examples from this data set.
        Fetches rows from a feature tables.
        
        Parameters:
        --------------
            batch_size: The number of examples to return        
        
        Returns:
        --------------        
            A list of npArrays, one for each feature requested
            
        """        
        batch_size = min(batch_size, self._num_examples)
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            for i in range(len(self._features)):
                self._features[i] = self._features[i][perm]

            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size

        end = self._index_in_epoch
        outs = []
        for i in range(self._num_features):
            outs += [self.get_features(i,start,end)]
           
        return outs
        
    def dump_values(self): 
        ''' 
        prints out the feature names and a short sample of the data in each feature
        
        example:
        -------------------------------------
            ds = ocr_utils.read_data(input_filters_dict = input_filters_dict, 
                                output_feature_list=output_feature_list,
                                test_size = .2,
                                engine_type='tensorflow')

            ds.train.dump_values()
    ''' 
        print('\nfeature output sample;')        
        for ftr,ftrName in zip(self.features,self.feature_names):            
            print('\t'+ftrName.ljust(20), end=': ')    
            if len(ftr.shape)==1:
                for k in range(0,5):
                    s= '{}'.format(ftr[k]).ljust(20)
                    print(s[:10],end=' ')  
            else:              
                for k in range(0,5):
                    s= '{}'.format(ftr[k,0]).ljust(20)
                    print(s[:10],end=' ')
            print('  ...')  
            
 
                
def apply_column_filters(df, input_filters_dict ):
    ''' apply the column filters to the incoming data
    
    parameters:
         input_filters_dict: filters to be applied to dataframe
    
    return:
        filtered datafram
    '''
    for key,value in input_filters_dict.items():
    
        if isinstance(value, str): 
            value = (value,) 
        if hasattr(value, '__iter__')==False:
            value = (value,) 
        if len(value) > 0:
            criterion = df[key].map(lambda x: x in value)
            df = df[criterion]  
#     try:
#         print (df['font'].iloc[0])
#     except:
#         print (df)
    return df

def convert_to_unique(t1):
    ''' convert unique values in an numpy.array into
    indices into the unique array
    
    arguments:
    t1 numpy scalar array
    
    return
    t1 with each value changed to an index 0 to number of unique
    values in t1-1
    '''
    t2 = np.zeros((len(t1),),dtype=np.int32)
    unique = np.unique(t1)
    for i,u in enumerate(unique):
        t2[t1==u]=i
    return t2


def read_data(fileName=default_zip_file, 
              input_filters_dict={}, 
              output_feature_list=[], 
              test_size=0.0, 
              evaluation_size=0.0, 
              dtype=np.float32, 
              engine_type='', 
              random_state=None  ):

    """
    Reads data from a given .zip file holding .csv files, 
    filters the data to extract the requested features and 
    outputs requested features in training and test feature lists.
    
    Parameters
    ------------------ 
    fileName : the path name of the zip file containing the .csv files 
        to be used as input
        
    input_filters_dict : a dictionary containing columns in the .csv file to
        be extracted.  keys = column heading, values = value to be 
        allowed in that column.  Returns an entire column if a key is not
        provided for it.
        
    output_feature_list : a list of names of features to be returned in a list.  
        The features include the column names in the .csv file as well as 
        computed features such as aspect_ratio
        
    test_size : the portion (0 to 1) of the data set to be returned for testing
    
    evaluation_size : the portion (0 to 1) of the data set to be returned 
        for evaluation   
        
    dtype: data type of image data to be returned
    
    engine_type: string,  'tensorflow', 'theano' specifies an output 
        format compatible with a classification engine.
        
    random_state: None for random seed chosen by the system
        or integer seed for the random seed for repeatable calls
        
    Returns
    ------------------
    a class with three attributes, 'train', 'test', and evaluation which are 
        instances of the TruthedCharacters class
    Each TruthedCharacters class contains the requested features in a list as 
        well as the size information

    Examples
    ------- 
    select only images from 'OCRA' and 'OCRB' fonts with 'scanned' fontVariant
    
    input_filters_dict = {'font': ('OCRA','OCRB'), 'fontVariant':('scanned,)')}
    
    output only the character label and the image
      
    output_feature_list = ['m_label_one_hot','image'] 
    
    ds = ocr_data.read_data(input_filters_dict = input_filters_dict, 
                            output_feature_list=output_feature_list)
    y = ds.train.features[0]  # this is the one_hot labels
    X = ds.train.features[1]  # this is image
    
    select everything; all fonts , font variants, etc.
    input_filters_dict = {}
    
    select the digits 0 through 9 in the E13B font
    input_filters_dict = {'m_label': range(48,58), 'font': 'E13B'}
    
    output the character label, image, italic flag, aspect_ratio and upper_case flag
    output_feature_list = ['m_label_one_hot','image','italic','aspect_ratio','upper_case']    
 
    output only the character label and the image
    output_feature_list = ['m_label_one_hot','image'] 
   
    filters available: {'h': , 'orientation': , 'm_top': , 'm_left': , 'w': , 
        'fontVariant': , 'originalH': , 'strength': , 'originalW': , 'italic': , 
        'm_label': , 'font': }
        
    features available: ['font', 'fontVariant', 'm_label', 'strength', 'italic', 
        'orientation', 'm_top', 'm_left', 'originalH', 'originalW', 'h', 'w', 
        'aspect_ratio', 'column_sum[]', 'upper_case', 'image', 'font_one_hot', 
        'fontVariant_one_hot', 'm_label_one_hot']

    
    Notes
    ------- 
    feature 'image' is the image data in .csv columns r0c0 to r19c19, scaled 0.0 
        to 1.0
    feature 'aspect_ratio' is the width / height for each character image
    feature 'upper_case' is 1 where the m_label for a character is upper case 
        else 0
    feature 'column_sum' is the sum of the vertical pixels in an image for the
         selected columns 
             c0          c1         ...       c18           c19
        r0
        r1
        .
        .
        .
        r18
        r19
            column_sum0  column_sum1  ...     column_sum18  column_sum19
            
        column_sum can be sliced, i.e. column_sum[2:4] or column_sum[4,7]
        
    All other features, such as 'italic' return the value from the the
         respective column in the .csv file
    
    """


    '''
    1) read in the fonts applying the input filter to extract only the fonts, 
    font variants and labels requested in the input_filter_list
    
    2) construct the derived features from those requested in the output_list
    
    3) make one-hot, 2D images or other computed features
    
    4) break into training and test sets
    
    5) construct training and test set TruthedCharacters classes and return them    
    '''

    df = read_file(fileName, input_filters_dict,random_state=random_state)
 
    return read_df(df, 
        input_filters_dict=input_filters_dict, 
        output_feature_list=output_feature_list, 
        test_size=test_size, 
        evaluation_size=evaluation_size, 
        dtype=dtype, 
        engine_type=engine_type, 
        random_state=random_state  )
                    
def read_df(df, 
              input_filters_dict={}, 
              output_feature_list=[], 
              test_size=0.0, 
              evaluation_size=0.0, 
              dtype=np.float32, 
              engine_type='', 
              random_state=None  ):
    
    class DataSets(object):
        pass
    
    print('\nparameter: input_filters_dict\n\t{}'.format(sorted(input_filters_dict.items())))
    print('parameter: output_feature_list\n\t{}'.format(output_feature_list))    
    
    engine_type = engine_type.lower()
    data_sets = DataSets()
    available_columns = []
    for key in df.columns:
        if key=='r0c0':  #omit the image
            break;
        available_columns.append(key)

    print('input filters available: \n\t{}:'.format(available_columns))
            
    h=int((df.iloc[0])['h'])  # get height and width of the image
    w=int((df.iloc[0])['w'])  # assumes that h and w are the same for all rows
    
    additional_features = ['aspect_ratio', 'column_sum[]', 'upper_case', 'image', 'font_one_hot', 'fontVariant_one_hot', 'm_label_one_hot']
    
    print('output features available:\n\t{}'.format(available_columns + additional_features))
    
    if len(output_feature_list)==0:
        output_feature_list = available_columns + additional_features

    nCount = df.shape[0]       
    nTestCount = math.floor(nCount*test_size)
    nEvaluationCount = math.floor(nCount*evaluation_size)
    nTrainCount = nCount - nTestCount - nEvaluationCount
                
    # construct features, one_hots, computed features etc
 
    outvars = [] 
    feature_name=[]  
    one_hot_map = []   
     
    for colName in output_feature_list:       
        one_hot_map.append(0)
        if colName=="aspect_ratio":  
            t1  = np.array(df['originalW'] ,dtype=dtype)
            t2  = np.array(df['originalH'] ,dtype=dtype) 
            t1 = t1[:]/t2[:]
            feature_name.append(colName)  
            
        elif colName=="upper_case":
            boolDF1 = df['m_label']>=64
            boolDF2 = df['m_label']<=90   
            boolDF = boolDF1 & boolDF2     
            t1 = np.array(boolDF,dtype=dtype) 
            feature_name.append(colName)                                
                   
        elif colName=='image':  
            t1 = np.array(df.loc[:,'r0c0':],dtype=dtype) #extract the images with is everything to the right of row 0 column 0
            t1 = np.multiply(t1, 1.0 / 256.0)     
            feature_name.append(colName)  
            
        elif colName=='low_pass_image':  
            ''' Create unique images for each of the labels, using a single font
            These are the concepts.
            Make an np array of the original images with the concept images replacing
            the originals.
            '''
            
            boolDF1 = df['fontVariant'] == 'BANKGOTHIC MD BT'
            
            criterion = df['fontVariant'].map(lambda x: x in 'BANKGOTHIC MD BT')
            x = df[criterion] 
            dx = pd.DataFrame(df)
            labels = np.array(x['m_label'])
            
            for i,label in enumerate(labels):  
                to_be_replaced =  df['m_label']!=label     
                dx =dx.where(to_be_replaced, other= x.iloc[i], axis=1)        

            
            t1 = np.array(dx.loc[:,'r0c0':],dtype=dtype) #extract the images with is everything to the right of row 0 column 0
            t1 = np.reshape(t1, (t1.shape[0],h,w))
            t2 = np.zeros((t1.shape[0],h,int(w/2)),dtype=dtype)            
            for col in range(w):
                t2[:,:,int(col/2)] = np.sum(t1[:,:,col:col+2],axis=2)
            t3 = np.zeros((t1.shape[0], int(h/2),int(w/2)),dtype=dtype)
            for row in range(h):
                t3[:,int(row/2),:] = np.sum(t2[:,row:row+2,:],axis=1)    
            t1 = np.reshape(t3,(t3.shape[0],int(h*w/4)) )        
            t1=(t1/4)/256.0                  
            feature_name.append(colName)                                        
            
        elif colName=='m_label_one_hot': 
            t1  = np.array(df['m_label'])
            t1 = convert_to_unique(t1)
            one_hot_map[-1] = len(np.unique(t1))    
            feature_name.append(colName)                     
            
        elif colName=='font_one_hot': 
            t1 =  np.array(df['font'])
            t1 = convert_to_unique(t1)    
            one_hot_map[-1] = len(np.unique(t1))   
            feature_name.append(colName) 
            
        elif colName=='orientation_one_hot': 
            t1 =  np.array(df['orientation'])
            t1 = convert_to_unique(t1)    
            one_hot_map[-1] = len(np.unique(t1))   
            feature_name.append(colName)             
            
        elif colName=='fontVariant_one_hot':  
            t1 = np.array(df['fontVariant'] ) 
            t1 = convert_to_unique(t1)
            one_hot_map[-1] = len(np.unique(t1))   
            feature_name.append(colName)    
                                     
        elif colName.find('column_sum')==0:
            # compute the sum of each vertical column
            t1 = df.loc[:,'r0c0':]
            t1 = np.multiply(t1, 1.0 / 256.0)  
            npx=np.array(t1,dtype=dtype)
            t1 = compute_column_sum(npx,h,w)
  
            n = colName.find('[')
            # if column_sum is sliced evaluate column_sum[n,m]
            if n>0:
                column_list = colName[n:]
                l = eval(column_list)     
            else:
                assert(n>0)   
            if len(l) == 0:
                feature_name.append('column_sum[:]')                   
            else:
                t1 = t1[:,l]   
                feature_name.append('column_sum{}'.format(l))    


        else: 
            if colName in df.columns  :     
                t1=np.array(df[colName], dtype=dtype)
                feature_name.append(colName)              
            else:
                raise ValueError('Invalid ouput_feature_name: {}: it is not in the the database'.format(colName))          
            
        outvars.append(t1)                                             

    outvars_train =[]       
    outvars_test = [] 
    outvars_evaluation = []     
    for ot in outvars:
        outvars_train.append( ot[nTestCount+nEvaluationCount:])
        outvars_test.append( ot[:nTestCount])    
        outvars_evaluation.append(ot[nTestCount:nTestCount+nEvaluationCount])
         
    data_sets.train = TruthedCharacters(outvars_train, feature_name, one_hot_map, engine_type, h, w, dtype)
    data_sets.test = TruthedCharacters(outvars_test, feature_name, one_hot_map,  engine_type, h,  w, dtype)
    data_sets.evaluation = TruthedCharacters(outvars_evaluation,feature_name,  one_hot_map,  engine_type, h,  w, dtype)    
    print ('feature results:')    
    print ('\tnumber of train Images = ',nTrainCount)
    print ('\tnumber of test Images = ',nTestCount) 
    print ('\tnumber of evaluation Images = ',nEvaluationCount)     
    print ('\toutput features returned:')
    for i,colName in enumerate(output_feature_list):
        print ('\t\t{}, width={}'.format(colName,data_sets.train.feature_width[i]))
    return data_sets
            

        
def load_E13B(chars_to_train=(48,49) , columns=(9,17), nChars=None, test_size=0,random_state=0):
    '''
    simplified data access for Python Machine Learning.  Use in place of Iris
    dataset
    
    Reads data from a given .zip file holding .csv files, 
    filters the data to extract the labels 
    outputs requested features in training and test feature lists.
    
    Parameters
    -----------------------
    chars_to_train : a tuple containing the ASCII values to get from the dataset
    
    columns : the column numbers 0-19 to return the sum of columns as features
    
    nChars : the number of characters to return
        
    test_size : the portion (0 to 1) of the data set to be returned for testing
        
    random_state: None for random seed chosen by the system
        or integer seed for the random seed for repeatable calls        
         
   Returns
    -------

    y_train: array-like, shape = [n_samples]
        Vector of target class labels
    
         X_train, y_test, X_test
         
    X_train : {array-like},
        shape = [n_samples, n_features]
        Matrix of training samples.
        
    y_test: array-like, shape = [n_samples]
        Vector of target class labels
    
         X_train, y_test, X_test
    X_test : {array-like},
        shape = [n_samples, n_features]
        Matrix of test samples.        

    labels: [list strings describing each feature]
        length = num_features
           
    Examples
    -----------------------      
    charsToTrain=range(48,58)
    columnsXY = tuple(range(0,20))    
    y_train, X_train, y_test, X_test, labels = ocr_utils.load_E13B(labels=charsToTrain , columns=columnsXY, nChars=1000)  

     
     Notes:
    ----------       
     The E13B font also known as the MICR font was invented in the 1950s to be 
     used on the bottom of checks.  The ink was magnetic. 
     
     The ink was magnetized and then
     the check was pulled under a single coil magnetic read head. Because
     a wire passing through a magnetic field produces a voltage, the output
     of the read head would form a waveform as it passed over a character.
     
     The characters were designed to be read as 2D images by humans but as
     a 1D waveform by the read head.  The waveform could be used in 
     identifying the character.
     
     As a result, summing the amount of black vertically for each column
     gives a pattern that can be used to distinguish
     characters.  The more black, the higher the column sum.  
     
     In the Python Machine Learning book, the IRIS data set is used for 
     about half the examples.  The type of flower can be determined by
     using two of the features, such as stem length, from the data set.
     Plotting each of the two features on the X and Y axis yields a two
     dimensional plot that is good for understanding the usage
     of various algorithms, and for developing an intuitive
     feel for pattern recognition.
     
     Using the column sums from the E13B font, some characters can be 
     recognized based upon two column sums and therefore may be plotted 
     with each column sum on the X and Y axes.  For instance, the 
     characters '0' and '1' can be differentiated based upon the sum
     of the pixels in the rows above column 9 of the image 
     and the sum of column 17.
     
     Because of this ability to be used in 2D plots, the E13B font
     can be used to replace the IRIS data set in most examples in the
     Python Machine Learning book. 
         

    '''
    ds = read_data(
                   input_filters_dict={'font': 'E13B', 'm_label': chars_to_train}, 
                   output_feature_list= ['m_label','column_sum{}'.format(list(columns))],  
                   test_size=test_size, 
                   evaluation_size=0, 
                   dtype=np.float32, 
                   random_state = random_state)
    
    if nChars == None:
        nChars = ds.train.features[0].shape[0]
        
    labels= ['column {} sum'.format(columns[i]) for i in range(len(columns))]
    #assert(np.all(ds.train.features[0]==ds.train._features[0]))
    #assert(np.all(ds.train.features[1]==ds.train._features[1]))
    #assert(np.all(ds.test.features[0]==ds.test._features[0]))
    #assert(np.all(ds.test.features[1]==ds.test._features[1]))    
        
    return ds.train.features[0][:nChars],  ds.train.features[1][:nChars], ds.test.features[0][:nChars], ds.test.features[1][:nChars],  labels


def compute_column_sum(npx,h,w):
    '''
    returns a numpy array of the sums of the rows in each column
    
    Parameters
    ----------------
    npx: {array-like}, shape (:,h * w)  The input images, each row is a 
        flattened image
    h: the height in rows of the image
    w: the width in columns of the image
    
    Returns
    --------------
    numpy Array, shape (npx.shape[0], w)
    
    Notes:   
    see column sum notes under read_data
    '''
    npx = np.reshape(npx,(npx.shape[0],h,w))
    return np.sum(npx,axis=1) # sum of rows in each column  
  
import skimage.transform as af
  
def shear(X, skew):
    ''' given a 2D image, shear and return a 2D image
    
    parameters:
        X is the 2D image of shape (nRows, nColumns)
        skew is the amount to shear in the range 0 to 1.0
    '''
    
    rows = X.shape[0]
    cols = X.shape[1]    
    ratioY = skew*cols/rows
    matrix =  np.array( [[1, ratioY, 0] ,[0, 1, 0] ,[0, 0, 1 ]])                                         
    tp=af.ProjectiveTransform(matrix=matrix) 
    #tp  = tf.AffineTransform(scale=(.3,.3), shear=skew)    
    f = af.warp(X, tp)      
    return f
# 
# class file_names(object):
#     ''' store variants of file a file name with .jpg, .png, .box variations
#     '''
#     def __init__(selp, base_name, dir_name = ''):
#         base = base_name
#         jpeg = base_name + '.jpg'
#         png = base_name + '.png'
#         box = base_name + '.box'
        
def file_to_df(base_file, character_size, title = "", white_space = 0, input_filters_dict={}):
    ''' 
    Given a 2D image file with some characters, uses Tesseract to cut out
    the characters, , plots the characters and the boxes that Tesseract
    found, then places the characters into a pandas dataframe
    and also returns the images in 2D format
    parameters:
        image_file, string  is the path name of the file
        character size, integer,  is the number of pixels of the return images
            in pixels, height and width are equal
        title, string is the name to be in the character/box plotted file
        white_space is the amount of padding to be placed around the characters
    '''
    input_image_file_jpg = base_file + extension
    print('input_image_file_jpg = {}'.format(input_image_file_jpg ))
    subprocess.run(['tesseract', input_image_file_jpg, base_file,  'batch.nochop', 'makebox'])
    #subprocess.run(['tesseract', input_image_file_jpg])    
    im = Image.open(input_image_file_jpg)
    
    images =[]
    recognized_label = []
    tops=[]
    originalH=[]
    lefts=[]
    originalW=[]
    orientation = 0

    f = open(base_file+'.box','r')

    for line in f: 
        coords = line.split(' ')        
        top = im.height-int(coords[4])
        bottom = im.height-int(coords[2])
        left = int(coords[1])
        right = int(coords[3])
        tops.append(top)
        originalW.append(right-left)
        lefts.append(left)
        originalH.append(bottom-top)
        
        img2 = Image.new('L',(character_size,character_size),color=255)
        img =  im.crop(box=(left, top, right, bottom))       
        img = img.convert('LA')     
        img2.paste(img, box=(white_space,white_space))    
                    
        imgByteArr = img2.tobytes()
        lst = list(imgByteArr)
        image = np.array(lst)/255.0 
        image = 1.0 - image        
        images.append(image)
        try:
            recognized_label.append(ord(coords[0]))
        except:
            print (coords[0])
            recognized_label.append(ord('_'))
 
    df = make_df(images, character_size, character_size, originalH, originalW, tops, lefts, orientation, recognized_label )
    df= apply_column_filters(df, input_filters_dict )
    t1 = np.array(df.loc[:,'r0c0':]) #extract the images with is everything to the right of row 0 column 0 
    t1 = np.reshape(t1,(t1.shape[0],character_size,character_size )) 
    montage(t1, title=title)
    return df,t1
        
def make_df(images, character_w, character_h, originalH, originalW, tops, lefts, orientation, recognized_label ):
    ''' Given an numpy array of images and attributes of each image, place these in a pandas dataframe
    '''
    x = np.zeros((len(images), character_w*character_h+8))
    x[:,0] = recognized_label
    x[:,1] = orientation   
    x[:,2] = tops
    x[:,3] = lefts  
    x[:,4] = originalH
    x[:,5] = originalW   
    x[:,6] = character_h  
    x[:,7] = character_w
#     print (' appending images')
    x[:,8:] = images                 
#     print (' DONE appending images')            
    
#     images = np.array(images)   
#     images = np.insert(images, 0, character_w, axis=1)
#     images = np.insert(images, 0, character_h, axis=1)
#     images = np.insert(images, 0, originalW, axis=1)    
#     images = np.insert(images, 0, originalH, axis=1)
#     images = np.insert(images, 0, lefts, axis=1)      
#     images = np.insert(images, 0, tops, axis=1)
#     images = np.insert(images, 0, orientation, axis=1) 
#     images = np.insert(images, 0, recognized_label, axis=1)
        
    columns = ['m_label', 'orientation','m_top','m_left','originalH','originalW', 'h','w']
    for i in range(character_h):
        for j in range(character_w):
            columns.append('r{}c{}'.format(i,j))

    df = pd.DataFrame(x, columns=columns)          

    return  df
                
###############################################  Miscellaneous Plot Routines ############################################################

num_fig = 0 # used to give each saved plot a unique name
def program_name():
    pg = sys.argv[0]
    pg2 = os.path.split(pg)
    return  os.path.splitext(pg2[1])[0]

def show_figures(plt, title="untitled"):
    '''
    Shows the plot or just writes it to a file based on the global boolean 
        show_plot at the top of this module
        
        If show_plot is true, the the plot is shown on the screen.
        If show_plot is false, the plot will be saved to a file in the
        /plots folder
        
        The files are given unique names based on the plot title
    args
        plt is the matplotlib plot to show or save
        title is the title to put on the plot window
    '''
    
    global num_fig 
    fig = plt.gcf()
    fig.canvas.set_window_title(title)  
 
    try:
        plt.tight_layout()
    except:
        print("Oops!  Tight layout error")
    plt.draw()
    if show_plot:
        plt.show()   
    else:    

        try:
            os.mkdir(plot_dir)
        except:
            pass

        #\/:*?"<>|
        title_file = title.replace('/','_')
        save_file_name= '{}/{}_{}_{}.png'.format(plot_dir, program_name(), num_fig,  title_file )
        print ('plotting {}'.format(save_file_name))
        plt.savefig(save_file_name, dpi=300)        
        plt.clf() # savefig does not clear the figure like show does
        plt.cla()
    num_fig += 1
#     print ('END PLOT')

def scatter_plot(X=None, y=None, legend_entries=[],axis_labels=("",""), title="",xlim=None, ylim=None):
    ''' 
        make a 2 dimensional scatter plot for all the data in X 
        corresponding to the unique numbers in y
        args:
            X is an nparray shape - (-1,2)
            
            y is a nparray shape containing the unique labels to plot
            
            axis_labels is a tuple containing the labels for the axes
            
            title is a string used in the file name and axes title
            
            legend_entries is a list of strings to show in the legend.  If
            legend_entries is not specified then the unique labels in y will
            be used in the legend
            
            xlim and ylim set the x and y axis limits
            
        '''
    assert X.shape[1]==2
    if len(legend_entries)==0:
        legend_entries= np.unique(y)
    
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan','orange','green','brown','lightblue','pink')  
    
    if ylim != None:
        plt.ylim(ylim)
    if xlim != None:
        plt.xlim(xlim) 
        
    for i,ys in enumerate(np.unique(y)):
        
        plt.scatter(
                    X[y == ys, 0], 
                    X[y == ys, 1], 
                    color=colors[i%len(colors)],  marker='o', 
                    label=legend_entries[i]) 
    
    plt.xlabel(axis_labels[0])
    plt.ylabel(axis_labels[1] ) 
    
        
    plt.legend(loc='upper left')
    plt.title(title)
    show_figures(plt, title)  
   
  
    
def plot_decision_regions(X=None, y=None, classifier=None, resolution = .005, test_idx=None, title="untitled", labels=("","")):
    '''
    plot_decision_regions is based on the function in Python Machine Learning.
    
    Given an array of images and an array of labels, shows the points as a
        scatter plot and shows the decision regions. when run through the
        classifieer provided.  It makes a meshgrid and tries each point
        on the grid, painting the area in a unique color for each label, y
        
        X: {array-like} shape (:,2) the x-coordinate and y-coordinate of
        each point to be scatter plotted. 
        
        y: {array-like} shape (:) the y labels of each of the X points
        
        classifier, a classifier with the predict method that can be used
        to predict a label given point
        
        resolution , a number specifying how fine the meshgrid will be
        test_idx, a list of y labels to be circle for emphasis
        labels are the X and Y axis labels
        
        title is the title to be put at the top of the plot     
    
    '''
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan','orange','green','brown','lightblue','pink')  
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    d= min((x2_max-x2_min)*resolution,(x1_max-x1_min)*resolution)
    xx1, xx2 = np.meshgrid(np.arange(x1_min-d, x1_max+2*d, d),
                         np.arange(x2_min-d, x2_max+2*d, d))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min()-d, xx1.max()+d)
    plt.ylim(xx2.min()-d, xx2.max()+d)

    # plot class samples

    for idx, cl in enumerate(np.unique(y)):
        xs = X[y == cl, 0]
        ys = X[y == cl, 1]
        c =cmap(idx)
        plt.scatter(xs, ys,
                    alpha=0.8, color=c,
                    marker=markers[idx%len(markers)], label=cl) 
        
    # highlight test samples
    if test_idx!=None:
    # broadcasting convert array to scalar if there is only one entry being
    # broadcast.  
        try:
            X_test_ = X[test_idx, :]  
            plt.scatter(X_test_[:, 0], X_test_[:, 1], c='', 
                        alpha=1.0, linewidth=1, marker='o',  
                        s=70, label='test set')   
        except:
            plt.scatter(X_test_[0], X_test_[1] , c='', 
                        alpha=1.0, linewidth=1, marker='o',  
                        s=70, label='test set')       
    plt.xlabel(labels[0])
    plt.ylabel(labels[1] ) 
    
    plt.legend(loc='upper left')
    plt.title(title)
    plt.tight_layout()
    show_figures(plt, title)   
    

def montage(X, maxChars = 2500, title=''):  
    ''' 
    montage displays a square matrix of characters
    
    ----------------------------------
    arguments:
        X is an np.array of character images of shape (:,h,w)
        maxChars is the maximum number of characters to display
        title = the title of the plot
    ---------------------------------
    example:
        misc_plot_routines.montage(X,maxChars=100, title='E13B Characters')
    '''   

    count,h, w = np.shape(X)    

    separator_size = 5
    count = min(maxChars,count)

    nCol = int(math.ceil(math.sqrt(count)))
    if nCol > 0:    
        nRow = int(math.ceil( count/nCol))
        M = np.zeros((nRow * h + (nRow-1)*separator_size, nCol * w + (nCol-1) * separator_size))
        image_id = 0
        for j in range(nRow):
            for k in range(nCol):
                if image_id >= count: 
                    break
                sliceH, sliceW = j * ( h+separator_size), k * (w+separator_size)              
                M[sliceH:sliceH + h, sliceW:sliceW + w] = X[image_id,:,:]
                image_id += 1
       
        plt.imshow(M, cmap='gray_r') 
        plt.title(title)
        plt.axis('off')     
        show_figures(plt, title)        

    
def show_examples(X2D,y, title='Sample Characters'): 
    ''' 
    plot a character map of one each image for each label in y
    
    parameters:
        X2D: nparray: shape(num_samples,rows,cols)
            images for each sample
            
        y:    shape(num_samples)
            classifications labels for each X2d 
        title: 
            title to be placed on the plot
    '''
    yn = np.unique(y)
    total_unique_labels= len(yn)
    if total_unique_labels>0:

        zz = np.zeros((total_unique_labels, X2D.shape[1], X2D.shape[2]))
        for i,ys in enumerate(yn):
            x = X2D[y==ys] 
            zz[i,:] = x[0,:]
      
        montage(zz, title=title, maxChars=1024)   
    
    