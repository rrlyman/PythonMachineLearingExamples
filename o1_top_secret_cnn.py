#!/usr/bin/python


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


encode a secret message in the angle of rotation of characters

Train a neural network on rotated versions of characters with the output of 
the network being the angle of rotation.  

Thus, given a rotated character, the neural network will yield a value 
that is the amount of rotation of the character.

Encode a test set by applying a secret message with one bit for each character.
Decode the secret message by running the rotated characters through the 
neural network, yielding the pattern of bits.


@author: richard lyman

"""# ==============================================================================
import ocr_utils


import numpy as np
from PIL import Image, ImageDraw
import io
#import n1_2cnv1fc as nnetwork     
#import n1_residual3x4 as nnetwork 
import n1_2cnv2fc as nnetwork     
import skimage.transform as af  
from bitarray import bitarray 


input_filters_dict = {'m_label': list(range(48,58))+list(range(65,91))}  
output_feature_list  = ['orientation_one_hot','image']   
dtype = np.float32

skewRange = np.linspace(-0.2,0.2,2)
       
'''
 pick up the base character
 
 make a training set by rotating them through n angles
 
 train
 
 pick up the base characters
 encode the secret message n bits at a time into the characters
 this is the testing set
 
 test secret message yielding a vector of rotations
 
 convert the rotation back into bits
 
 assemble the bits into the secret message.
 '''


# pick up the base characters from training_image_file
# produce some sheared versions 
# make into a training set
# place in a ocr_utils TruthedCharacters class so we can use the
# one hot and batch functions 

character_size = 100
white_space=8

image_file= '15-01-01 459_Mont_Lyman'
image_file_jpg = image_file+'.jpg'

df,t1  = ocr_utils.file_to_df(image_file,character_size,title='Characters to Train',white_space=white_space)

shp = t1.shape
totalN = len(skewRange)*shp[0]

images=[]
originalH=[]
originalW=[]
tops=[]
lefts=[]
orientation=[]
recognized_label =[]



for j in range(shp[0]):
    for i,skew in enumerate(skewRange):
        k = i+j*len(skewRange)
        
        images.append(ocr_utils.shear(t1[j],skew))
        originalH.append(df['originalH'][j])
        tops.append(df['m_top'][j])        
        originalW.append(df['originalW'][j])
        lefts.append(df['m_left'][j])
        
        orientation.append(skew)
        recognized_label.append( df['m_label'][j])
images=np.array(images)
ocr_utils.montage(images, title='Base Characters Skewed')  

images = np.reshape(images,(images.shape[0],images.shape[1]*images.shape[2]))
df = ocr_utils.make_df(images, character_size, character_size, originalH, originalW, tops, lefts, orientation, recognized_label )
#df = ocr_utils.make_df(images, character_size, character_size, bottoms, rights, tops, lefts, orientation, recognized_label )


# input_filters_dict = {'m_label': list(range(48,58))+list(range(65,91))}  
input_filters_dict = {'m_label': list(range(48,58))+list(range(65,91))}  
output_feature_list  = ['orientation_one_hot','image']   
ds = ocr_utils.read_df(df,input_filters_dict = input_filters_dict, 
                            output_feature_list=output_feature_list,
                            test_size = 0,
                            engine_type='tensorflow',
                            dtype=dtype) 

nn = nnetwork.network(ds.train) 
"""# ==============================================================================

Train and Evaluate the Model

"""# ==============================================================================
    
nn.fit( ds.train ,  nEpochs=5000)

#######################################################################################

# now that the font is trained, pick up some text and encode a message
image_file= '15-01-01 459_Mont_Lyman'
image_file_jpg = image_file+'.jpg'
df,t1 = ocr_utils.file_to_df(image_file,character_size, title = 'unencrypted file',white_space=white_space)
 
 
secret_message = "top secret"
a = bitarray()   
a.frombytes(secret_message.encode('utf_8'))

index = 0
encoded_skews=[]
def convert_to_shear(a):
    index = 0
    while True:
        if index < len(a):        
            bits = a[index:index+1].to01()      
            index += 1
            #c = int(bits,2)
            c = int(bits)            
            yield c
        else:
            yield -1   
            
gen= convert_to_shear(a)

im = Image.open(image_file_jpg)     
img2 = Image.new('L',(im.height,im.width),color=255)
img3 = Image.new('L',(im.height,im.width),color=255)
draw = ImageDraw.Draw(img3) 
for i in range(t1.shape[0]):
    left = int(df['m_left'][i])
    right = left + int(df['originalW'][i])
    top = int(df['m_top'][i])  
    bottom = top + int(df['originalH'][i])  
    skew_index = next(gen)
    #print ('i={}, skew_index={}, left={}, top={}, right={}, bottom={}'.format(i,skew_index, left,top,right,bottom))    
    encoded_skews.append(skew_index)
    if skew_index >= 0:
        t1[i] = ocr_utils.shear(t1[i], skewRange[skew_index]) 
    im_clip = Image.fromarray(256.0-t1[i]*256.0)  
    img2.paste(im_clip, box= (left , top))
    img3.paste(im_clip, box= (left , top))  
         

    draw.rectangle((left,top,right+2*white_space,bottom+2*white_space), outline=0)
    
gen.close()

###########################################################################vvvvvvv
image_file= '/tmp/plots/01_encrypted_file'
image_file_jpg = image_file+'.jpg'
img2.save(image_file_jpg)   


image_file3= '/tmp/plots/01_03_encrypted_file_with_box'
image_file3_jpg = image_file3+'.jpg'
img3.save(image_file3_jpg)    

''' test the new encrptyed file
'''
df,t1  = ocr_utils.file_to_df(image_file,character_size, title = 'Encrypted File',white_space=white_space)

ds = ocr_utils.read_df(df,input_filters_dict = input_filters_dict, 
                            output_feature_list=output_feature_list,
                            test_size = 1,
                            engine_type='tensorflow',
                            dtype=dtype)
    
results = nn.predict(ds.test)  
correct_characters=[]
incorrect_characters=[]
for i,x in enumerate(df['m_label']):
    try:
        print('index={}, original character={}, result= {}, skew={}'.format(i, chr(int(x)),results[i], encoded_skews[i])   )  
        if encoded_skews[i] >=0:
            if results[i] == encoded_skews[i]:
                correct_characters.append(chr(int(x)))
            else:
                incorrect_characters.append(chr(int(x)))
    except:
        print ('index out of bounds={}'.format(i))
print ('correct characters={}'.format(correct_characters))       
print ('incorrect characters={}'.format(incorrect_characters))
     
print ('\n########################### No Errors ####################################')

