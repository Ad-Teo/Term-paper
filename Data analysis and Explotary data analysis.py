#!/usr/bin/env python
# coding: utf-8

# In[76]:


import pandas as pd  # data processing, CSV file I/O
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

import plotly.graph_objecs as go
pip install folium
import folium
import plotly.express as px
import os
pip install tensorflow 
import tensorflow as tf
import tensorflow.keras as keras


# In[39]:


_img = keras.preprocessing.image.load_img('C:/Users/user/Downloads/archive (1)/Face Data/Face Dataset/10/0.jpg')
plt.imshow(_img)
plt.show()


# In[30]:


pip install tensorflow


# In[31]:


import tensorflow.keras as keras


# In[36]:


import matplotlib.pyplot as plt


# In[54]:


df = pd.read_csv("C:/Users/user/downloads/face.csv")
print(df.head())
df.columns


# In[46]:


import pandas as pd


# In[55]:


df.columns


# In[56]:


df= df[['RowId','ImageId','FeatureName','Location']]
df.head()


# In[57]:


cor = df.corr()
cor


# In[64]:


plt.figure(figsize=(9,6))
sns.heatmap(data = cor)


# In[63]:


import seaborn as sns


# In[71]:


df.boxplot(column=["RowId"])


# In[102]:


import tensorflow as tf
import tensorflow.keras as keras

import numpy as np
print('tensorflow{}'.format(tf.__version__))
print('keras{}'.format(keras.__version__))
import matplotlib.pyplot as plt

model = keras.applications.VGG16(weights='imagenet')


# In[18]:


#preprocess image to get it into the right formatfor the model
from tensorflow import keras
from keras.preprocessing.image import load_img

img = keras.preprocessing.image.img_to_array(_img)
img = img.reshape((1, *img.shape))
y_pred = model.predict(img)

print(y_pred)


# In[122]:


from sklearn.model_selection import train_test_split

 img = keras.preprocessing.image.img_to_array(_img)
 
# In[135]:


sns.pairplot(df)


# In[17]:


from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import train_data_dir
train_datagen = ImageDataGenerator(rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2) # set validation split

train_generator = train_datagen.flow_from_directory(
    train_data_dir('C:/Users/user/Downloads/archive (1)/Face Data/Face Dataset/10/0.jpg'),
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    subset='training') # set as training data

validation_generator = train_datagen.flow_from_directory(
    train_data_dir, # same directory as training data
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    subset='validation') # set as validation data


# In[7]:


import numpy as np


# In[16]:


import keras
from keras import layers


# In[ ]:




