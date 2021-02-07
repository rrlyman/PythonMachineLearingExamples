'''
Created on Oct 23, 2016

Created on Jul 12, 2016
This program shows how Principal Component Analysis removes affine 
transformation distortions.

Parallel lines in an image remain parallel after an affine transformation. 
For instance, if an image is rotated or sheared, lines remain parallel.

PCA and LDA can remove affine transformations.  This is shown by making 3 shapes
and then making a number of shear versions of the shapes.  Running 
Principal Component Analysis reduces the number of features necessary to
recognize the features during Logistic Regression with 100% accuracy, 
down to 2 from 400 (20 columns by 20 rows).

We make three images and then make about 80 copies of each image created by
shearing the original image.

Since there is very little noise introduced by the shearing, almost all of
the explained variance is due to the shearing. PCA finds eigenvectors
that line up with shearing.

1) For a couple of shapes, make sheared version.
2) train and print accuracies without PCA 
3) repeat, but use PCA first before training.
4) observe the improvement

Do the same thing for Linear Discriminant Analysis
encode a secret message in the angle of rotation of characters

Train a neural network on rotated versions of characters with the output of 
the network being the angle of rotation.  

Thus, given a rotated character, the neural network will yield a value 
that is the amount of rotation of the character.

Encode a test set by applying a secret message with one bit for each character.
Decode the secret message by running the rotated characters through the 
neural network, yielding the pattern of bits.

 pick up the base character
 
 make a training set by rotating them through n angles
 
 train
 
 pick up the base characters
 encode the secret message n bits at a time into the characters
 this is the testing set
 
 test secret message yielding a vector of rotations
 
 convert the rotation back into bits
 
 assemble the bits into the secret message.


@author: richard lyman


'''# ==============================================================================

import ocr_utils


import numpy as np
from PIL import Image, ImageDraw
import io
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
#from sklearn.model_selection
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# input_filters_dict = {'m_label': list(range(48,58))+list(range(65,91))}  
# output_feature_list  = ['orientation_one_hot','image']   
dtype = np.float32
character_size = 100
white_space = 10       
skewRange = np.linspace(-0.2,0.2,4)

class c_box(object):
    def __init__(self, top, left, right, bottom):
        self._top = top
        self._left = left
        self._right = right
        self._bottom = bottom


def find_min_max(sums):
    case = 0 
    mins = []
    maxes = []
    for i,sum in enumerate(sums):   
        '''
            case 0, going through area between characters
                if sum ==0 stay in case 0
                if sum != 0 set the top to i and switch to case 1
            case 1, going through a character
                if sum ==0 set the bottom to i and drop to case 0
                    also append the box to the list using 
                    left = 0, and right = the width of the image                
                if sum !=0 then continue in case 1
        '''
      
        if case==0 :
            if sum != 0 :
                case = 1
                min= i
        else:
            if sum == 0    :             
                case = 0
                max= i
                mins.append(min)
                maxes.append(max)
    return mins, maxes
          


# pick up the base characters from training_image_file
# produce some skeared versions 
# make into a training set
# place in a ocr_utils TruthedCharacters class so we can use the
# one hot and batch functions 

im = Image.open('15-01-01 459_Mont_Lyman.png')
#im = Image.open('CourierFont.png')
im = im.convert(mode='L')  
data = 255-np.asarray( im, dtype="int32" )
sums = np.sum(data,axis=1)
mins, maxes = find_min_max(sums)
boxes = []
for top,bottom in zip(mins,maxes):
    line = data[top:bottom]
    line_sums = np.sum(line,axis=0)
    lefts,rights = find_min_max(line_sums)
    for left,right in zip(lefts,rights):
        boxes.append(c_box(top,left,right,bottom))
    
images=[]
orientation=[]
recognized_label =[]
for box in boxes:
      
    img2 = Image.new('L',(character_size,character_size),color=255)
    
    img =  im.crop(box=(box._left, box._top, box._right, box._bottom))     
    img2.paste(img,box=(white_space,white_space))    
    
    imgByteArr = img2.tobytes()
    lst = list(imgByteArr)
    image = np.array(lst)/255.0 
    image = 1.0 - image        
    images.append(image)

height = im.height
width = im.width

t1 = np.array(images)
t1=np.reshape(t1,(t1.shape[0],character_size,character_size))
ocr_utils.montage(t1, title='characters from file')

shp = t1.shape
totalN = len(skewRange)*shp[0]
images = []
import skimage.transform as af  

for j in range(shp[0]):
    for i,skew in enumerate(skewRange):       
        images.append(ocr_utils.shear(t1[j],skew))       
        orientation.append(skew)

images=np.array(images)
ocr_utils.montage(images, title='characters being trained')
images=np.reshape(images,(len(images),character_size*character_size))
ys = ocr_utils.convert_to_unique(orientation)


X_train , X_test, y_train, y_test = train_test_split(images, ys, test_size=0.3, random_state=0)
print (y_test.shape)

lr = LogisticRegression()
lr.fit(X_train, y_train)
y_train_pred = lr.predict(X_train)
y_test_pred = lr.predict(X_test)

print('\nTrain Accuracy: {:4.6f} coefficients={}'.format(accuracy_score(y_train, y_train_pred), lr.coef_.shape))
print('Test Accuracy: {:4.6f} coefficients={}'.format(accuracy_score(y_test, y_test_pred), lr.coef_.shape))

#########################################################################
# run Principal Component analysis first, then Logistic Regression

n_components = 2
pca = PCA(n_components=n_components)

X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

print('\nPCA components = {}'.format(pca.components_.shape))

lr = LogisticRegression()
logistic_fitted = lr.fit(X_train_pca, y_train)

y_train_pred = logistic_fitted.predict(X_train_pca)
y_test_pred = logistic_fitted.predict(X_test_pca)

print('\nPCA Train Accuracy: {:4.6f}, n_components={} coefficients={}'.format(accuracy_score(y_train, y_train_pred),pca.n_components,lr.coef_.shape))
print('PCA Test Accuracy: {:4.6f}, n_components={} coefficients={}'.format(accuracy_score(y_test, y_test_pred),pca.n_components,lr.coef_.shape))

X_errors_image = X_test[y_test!=y_test_pred]
y_errors = y_test[y_test!=y_test_pred]
X_errors_pca = X_test_pca[y_test!=y_test_pred]

# change to a 2D shape 
X_errors2D=np.reshape(X_errors_image, (X_errors_image.shape[0], character_size, character_size))
ocr_utils.montage(X_errors2D,title='PCA Error Images, components={}'.format (n_components))

X_combined = np.vstack((X_train_pca, X_test_pca))
y_combined = np.hstack((y_train, y_test))

ocr_utils.plot_decision_regions(
                                         X=X_combined,                                        
                                         y=y_combined,                                        
                                         classifier=lr,  
                                         labels = ['PC1','PC2']  ,     
                                         title='logistic_regression after 2 component PCA')


#########################################################################
# run Linear Discriminant Analysis first then Logistic Regression


n_components = 2
lda = LinearDiscriminantAnalysis(n_components=n_components)

X_train_lda = lda.fit_transform(X_train, y_train)
X_test_lda = lda.transform(X_test)

print('\nLDA components = {}'.format(pca.components_.shape))
lr = LogisticRegression()
logistic_fitted = lr.fit(X_train_lda, y_train)

y_train_pred = logistic_fitted.predict(X_train_lda)
y_test_pred = logistic_fitted.predict(X_test_lda)

print('\nLDA Train Accuracy: {:4.6f}, n_components={} coefficients={}'.format(accuracy_score(y_train, y_train_pred),lda.n_components,lr.coef_.shape))
print('LDA Test Accuracy: {:4.6f}, n_components={} coefficients={}'.format(accuracy_score(y_test, y_test_pred),lda.n_components,lr.coef_.shape))

X_errors_image = X_test[y_test!=y_test_pred]

# change to a 2D shape 
X_errors2D=np.reshape(X_errors_image, (X_errors_image.shape[0], character_size, character_size))
ocr_utils.montage(X_errors2D,title='LDA Error Images, components={}'.format (n_components))

X_combined = np.vstack((X_train_lda, X_test_lda))
y_combined = np.hstack((y_train, y_test))
if X_combined.shape[1] > 1:
    ocr_utils.plot_decision_regions(
                                             X=X_combined,                                        
                                             y=y_combined,                                        
                                             classifier=lr,  
                                             labels = ['LDA1','LDA2']  ,     
                                             title='logistic_regression after 2 component LDA')
print ('\n########################### No Errors ####################################')
