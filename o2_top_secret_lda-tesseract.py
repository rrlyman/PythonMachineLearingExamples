'''

Created on Oct, 2016
T

@author: richard
'''
import ocr_utils
import numpy as np
from PIL import Image, ImageDraw
import io
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA   
from sklearn.metrics import accuracy_score
from ruamel_yaml.compat import utf8

inputs = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghiklnopqrstuvwxyz'
inputs_list = list(ord(x) for x in inputs)
input_filters_dict = {'m_label': inputs_list}
# input_filters_dict={}
             
output_feature_list  = ['orientation','image']   
dtype = np.float32

#if -0.3 whitespace 8 is not enough
#if 0-.2 then whitespace 6 is just enough
character_size = 100
white_space=6
skewRange = np.linspace(-0.1,0.1,4)
       
'''
 pick up the base character via tesseract
 
 make a training set by shearing them 
 
 save images
 
 retrieve and unbox with tesseract
 
 train
 
 pick up the base characters
 encode the secret message n bits at a time into the characters
 this is the testing set
 
 test secret message yielding a vector of rotations
 
 convert the rotation back into bits
 
 assemble the bits into the secret message.
 '''


# pick up the base characters from training_image_file
# produce some skeared versions 
# make into a training set
# place in a ocr_utils TruthedCharacters class so we can use the
# one hot and batch functions 


def encode_and_save_file(input_base, output_base, character_size, white_space, secret_message=''):       
    input_image_file_jpg = input_base+ocr_utils.extension
    ouput_encoded_file = output_base +ocr_utils.extension
    output_box_file =  output_base +'_box' +ocr_utils.extension
    print ('input_base = {}'.format(input_base))
    print ('input_image_file_jpg = {}'.format(input_image_file_jpg))
    print ('ouput_encoded_file = {}'.format(ouput_encoded_file))    
    print ('output_box_file = {}'.format(output_box_file))
    
    df,t1 = ocr_utils.file_to_df(input_base, character_size, title = 'unencrypted file', white_space=white_space, input_filters_dict=input_filters_dict)
    
    from bitarray import bitarray  
    a = bitarray()   
    a.frombytes(secret_message.encode('utf-8') )   
    index = 0

    def convert_to_shear(a):
        index = 0
        while True:
            if index < len(a)-1:        
                bits = a[index:index+2].to01()      
                index += 2
                c = int(bits,2)
                #c = int(bits)            
                yield c
            else:
                yield -1   
                
    def draw_encoded_images(skews_indices, offset=0):
            
        for i in range(len(t1)):
            left = right = top = bottom = 0
            try:
                left = int((df['m_left']).iloc[i])
                right = left + int((df['originalW']).iloc[i])
                top = int((df['m_top']).iloc[i]) +offset 
                bottom = top + int((df['originalH']).iloc[i])  
                skew_index = skews_indices[i]
                #print ('i={}, skew_index={}, left={}, top={}, right={}, bottom={}'.format(i,skew_index, left,top,right,bottom))    
    
                if skew_index >= 0:
                    z = ocr_utils.shear(t1[i], skewRange[skew_index]) 
                else:
                    z=t1[i]
                im_clip = Image.fromarray(256.0-z*256.0)  
                img2.paste(im_clip, box= (left , top))
                img3.paste(im_clip, box= (left , top))           
                draw.rectangle((left,top,right+2*white_space,bottom+2*white_space), outline=0)
            except:
                print (left,right,top,bottom,df.columns)
        return bottom
            
    im = Image.open(input_image_file_jpg)     
         
  
    bottom = 0             
    if len(secret_message)==0:
        img2 = Image.new('L',(im.width,im.height*3),color=255)  
        img3 = Image.new('L',(im.width,im.height*3),color=255) 
        draw = ImageDraw.Draw(img3)
        for skew_index in range(len(skewRange)):
            skew_indices = []                    
            for i in range(len(t1)):
                skew_indices.append(skew_index)
            bottom = draw_encoded_images(skew_indices, offset=bottom+16)             
    else:
        img2 = Image.new('L',(im.width,im.height),color=255)        
        img3 = Image.new('L',(im.width,im.height),color=255)        
        draw = ImageDraw.Draw(img3)        
        gen= convert_to_shear(a)      
        skew_indices = []       
        for i in range(len(t1)):
            skew_indices.append(next(gen))  
        draw_encoded_images(skew_indices, offset=0)
        gen.close()
        
    img2.save(ouput_encoded_file)   
    img3.save(output_box_file)   
    
    return output_base,skew_indices


######################################################################################
# us the original document as the source of characters to shear and train
######################################################################################

base_file= '15-01-01 459_Mont_Lyman'
next_base = '/tmp/plots/'+base_file+'_training'

# shear the characters
base_file,skew_indices = encode_and_save_file(base_file, next_base , character_size, white_space)

# use tesseract to make the boxes around each skewed character.
df,t1  = ocr_utils.file_to_df(base_file,  character_size,title='Characters to Train',white_space=white_space,input_filters_dict=input_filters_dict)

ds = ocr_utils.read_df(df,input_filters_dict = input_filters_dict, 
                            output_feature_list=output_feature_list,
                            test_size = 0,
                            engine_type='tensorflow',
                            dtype=dtype) 



X_train = ds.train.features[1]
# the characters were written once for each entry in skewrange
# fill in the y_train with the skew_index
y_train = np.zeros(len(X_train), dtype=np.int32)
for i in range(len(X_train)):
    y_train[i] = i / (len(X_train)/len(skewRange))
                   
print (y_train)
print (y_train.shape)
print (X_train.shape)

######################################################################################
# train the characters. The resultant logistic regression is the key to decoded
######################################################################################

n_components = 2
lda = LDA(n_components=n_components)

X_train_lda = lda.fit_transform(X_train, y_train)

print('\nLDA components = {}'.format(lda.n_components))
lr = LogisticRegression()
logistic_fitted = lr.fit(X_train_lda, y_train)

y_train_pred = logistic_fitted.predict(X_train_lda)

print('\nLDA Train Accuracy: {:4.6f}, n_components={} coefficients={}'.format(accuracy_score(y_train, y_train_pred),lda.n_components,lr.coef_.shape))
# print('LDA Test Accuracy: {:4.6f}, n_components={} coefficients={}'.format(accuracy_score(y_test, y_test_pred),lda.n_components,lr.coef_.shape))

X_errors_image = X_train[y_train!=y_train_pred]

X_errors2D=np.reshape(X_errors_image, (X_errors_image.shape[0], character_size, character_size))
ocr_utils.montage(X_errors2D,title='LDA Error Images, components={}'.format (n_components))

#  X_combined = np.vstack((X_train_lda, X_test_lda))
    #  y_combined = np.hstack((y_train, y_test))
if X_train_lda.shape[1] > 1:
    ocr_utils.plot_decision_regions(
        X=X_train_lda,                                        
        y=y_train,                                        
        classifier=lr,  
        labels = ['LDA1','LDA2']  ,     
        title='logistic_regression after 2 component LDA')
    
######################################################################################
# now that the font is trained, pick up some text and encode a message
######################################################################################

base_file = '15-01-01 459_Mont_Lyman'
output_base = '/tmp/plots/15-01-01 459_Mont_Lyman_encrypted'
base_file,skew_indices = encode_and_save_file(base_file, output_base, character_size, white_space, secret_message='your first born is mine')  
print ('base file to decode = {}'.format(base_file))   


df,t1  = ocr_utils.file_to_df(base_file, character_size, title = 'Encrypted File',white_space=white_space,input_filters_dict=input_filters_dict)

ds = ocr_utils.read_df(df,input_filters_dict = input_filters_dict, 
                            output_feature_list=output_feature_list,
                            test_size = 0,
                            engine_type='tensorflow',
                            dtype=dtype)

print ('document length in chars={}'.format(len(t1)))
X_train = ds.train.features[1]
X_train_lda = lda.transform(X_train)    
results = logistic_fitted.predict(X_train_lda)  
correct_characters=[]
incorrect_characters=[]
error_characters=[]
decoded_message = ''
dc = 0

for i,x in enumerate(df['m_label']):
    try:
 
        if skew_indices[i] >=0:
            dc = dc * 4 + skew_indices[i]
            if (i+1) % 4 == 0:
                decoded_message = decoded_message + chr(dc)
                dc = 0
                
            
            print('index={}, original character={}, result= {}, skew={}'.format(i, chr(int(x)),results[i], skew_indices[i])   )             
            if results[i] == skew_indices[i]:
                correct_characters.append(chr(int(x)))
            else:
                incorrect_characters.append(chr(int(x)))
                error_characters.append(X_train[i])
    except:
        print ('.',end='')
error_characters = np.array(error_characters)

error_characters=np.reshape(error_characters, (error_characters.shape[0], character_size, character_size))
ocr_utils.montage(error_characters,title='LDA Encrption Errors, components={}'.format (n_components))
print ('\ncorrect characters={}'.format(correct_characters))       
print ('incorrect characters={}'.format(incorrect_characters))
print ("decoded message={}".format(decoded_message))
     
######################################################################################
# decode the message
######################################################################################

print ('\n########################### No Errors ####################################')
