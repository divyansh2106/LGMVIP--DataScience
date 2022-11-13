#!/usr/bin/env python
# coding: utf-8

# # DIVYANSH SINGHAL
# # DATA SCIENCE INTERN - LetsGrowMore
# # TASK 1 - IRIS FLOWER CLASSFICATION MACHINE LEARNING PROJECT

# In[34]:


#Importing the necessary libraries
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
import warnings
warnings.filterwarnings("ignore")


# In[7]:


df=pd.read_csv('Iris_Dataset.csv')


# In[8]:


df.describe()


# In[10]:


#column heads
df.columns


# In[12]:


#information abou the dataset
df.info()


# In[13]:


#Number of rows and columns in the dataset
df.shape


# In[17]:


#splitting the data
x=df[['sepal_length','sepal_width','petal_length','petal_width']]
y=df[['species']]


# In[18]:


#splitting the feature variables
x.head()


# In[19]:


#segregating the target variable
y.head()


# In[29]:


#Training and testing
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0)

x_train


# In[30]:


x_test


# In[31]:


y_train


# In[32]:


y_test


# In[35]:


from sklearn.linear_model import LogisticRegression
model=LogisticRegression()

model.fit(x_train,y_train)


# In[22]:


#Value Prediction
y_pred=model.predict(x_test)
print(y_pred)


# In[23]:


#Accuracy and confusion matrix
from sklearn.metrics import accuracy_score,confusion_matrix
confusion_matrix(y_test,y_pred)
k=accuracy_score(y_test,y_pred)*100

print('The accuracy is ',k)

