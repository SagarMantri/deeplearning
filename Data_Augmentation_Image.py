#!/usr/bin/env python
# coding: utf-8

# In[6]:


from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator 


# In[11]:


img=image.load_img('cat_image.jpeg',target_size=(200,200))


# In[12]:


import matplotlib.pyplot as  plt
plt.imshow(img)


# In[20]:


datagen= ImageDataGenerator(
    rotation_range=0.2,
    shear_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2)


# In[21]:


img=image.img_to_array(img)


# In[22]:


img.shape


# In[23]:


input_batch=img.reshape(1,200,200,3)


# In[25]:


i=0

for output in datagen.flow(input_batch,batch_size=1,save_to_dir='aug'):
    i=i+1
    
    if i==10:
        break


# In[19]:


input_batch.shape

