import pandas as pd
import os
from skimage.transform import resize
from skimage.io import imread
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pickle

### Predict new image
Categories=['cats','dogs']

path='./pictures/pic02.jpg'
print(f'[{datetime.now()}] Predict a newly selected image at {path}')

# load the model from disk
filename = 'model-svm01.sav'
svm_model = pickle.load(open(f'./image-prediction/saved-models/{filename}', 'rb'))

img=imread(path)
plt.imshow(img)
plt.show()
img_resize=resize(img,(150,150,3))
l=[img_resize.flatten()]
probability=svm_model.predict_proba(l)
for ind,val in enumerate(Categories):
	print(f'{val} = {probability[0][ind]*100}%')
print(f'*** The predicted image is [{Categories[svm_model.predict(l)[0]]}] with {probability[0][0]*100}% probability')

