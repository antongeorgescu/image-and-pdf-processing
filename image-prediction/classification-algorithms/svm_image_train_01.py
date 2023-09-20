import pandas as pd
import os
from skimage.transform import resize
from skimage.io import imread
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from datetime import datetime
import pickle

print(f'[{datetime.now()}] Start training script run...')

### (1) Load the image and convert it to a data frame.
print(f'[{datetime.now()}] Load images and converting them to dataframe')
Categories=['cats','dogs']
flat_data_arr=[] #input array
target_arr=[] #output array
datadir='./data-samples/images/kaggle/dogs-and-cats/'
#path which contains all the categories of images
for i in Categories:
	
	print(f'*** Loading... category : {i}')
	path=os.path.join(datadir,i)
	for img in os.listdir(path):
		img_array=imread(os.path.join(path,img))
		img_resized=resize(img_array,(150,150,3))
		flat_data_arr.append(img_resized.flatten())
		target_arr.append(Categories.index(i))
	print(f'[{datetime.now()}] Loaded category:{i} successfully')
flat_data=np.array(flat_data_arr)
target=np.array(target_arr)

#dataframe
df=pd.DataFrame(flat_data)
df['Target']=target
df.shape

### (2) Separate input features and targets.

print(f'[{datetime.now()}] Separate input features and targets')
#input data
x=df.iloc[:,:-1]
#output data
y=df.iloc[:,-1]

print(f'[{datetime.now()}] Split the data into training and testing sets')
# Splitting the data into training and testing sets
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,
											random_state=77,
											stratify=y)


### Build and train the model
# Defining the parameters grid for GridSearchCV
print(f'[{datetime.now()}] Build and train the model')
param_grid={'C':[0.1,1,10,100],
			'gamma':[0.0001,0.001,0.1,1],
 			'kernel':['rbf','poly']}

print(f'[{datetime.now()}] Create a support vector classifier')
# Creating a support vector classifier
svc=svm.SVC(probability=False)

print(f'[{datetime.now()}] Create a model using GridSearchCV with the parameters grid')
# Creating a model using GridSearchCV with the parameters grid
model=GridSearchCV(svc,param_grid,n_jobs=-1)

print(f'[{datetime.now()}] Train the model using the training data')
# Training the model using the training data
model.fit(x_train,y_train)

### Evaluate the model (calculate accuracy)
# Testing the model using the testing data
print(f'[{datetime.now()}] Test the model using the testing data')
y_pred = model.predict(x_test)

# Calculating the accuracy of the model
print(f'[{datetime.now()}] Calculate the accuracy of the model')
accuracy = accuracy_score(y_pred, y_test)

# Print the accuracy of the model
print(f"*** The model is {accuracy*100}% accurate")


# Generate Classification Report
print(classification_report(y_test, y_pred, target_names=['cat', 'dog']))

# save the model to disk
filename = 'model-svm01.sav'
pickle.dump(model, open(f'./image-prediction/saved-models/{filename}', 'wb'))
print(f'[{datetime.now()}] Saved model to file {filename}')

print(f'[{datetime.now()}] End training script run')






