
from keras.models import Sequential
from keras.layers import Activation,Dropout,Dense,Flatten,Conv2D,MaxPooling2D,BatchNormalization
from keras.preprocessing.image import ImageDataGenerator


# In[3]:


from keras.layers import Convolution2D,BatchNormalization,AveragePooling2D
input_shape=(48,48,3)
model = Sequential()
model.add(Convolution2D(filters=32, kernel_size=(3, 3), padding='same',
                            name='image_array', input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(BatchNormalization())

model.add(Convolution2D(filters=32, kernel_size=(3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(BatchNormalization())

model.add(Convolution2D(filters=64, kernel_size=(3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(BatchNormalization())

model.add(Convolution2D(filters=64, kernel_size=(5, 5), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(BatchNormalization())


model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(.50))

model.add(Dense(5))
model.add(Activation('softmax'))
model.compile(loss='sparse_categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])


# In[4]:


from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


# In[5]:


model.summary()


# In[6]:


batch=50
image_gen=ImageDataGenerator(featurewise_center=False,
                        featurewise_std_normalization=False,
                        rotation_range=10,
                        width_shift_range=0.1,
                        height_shift_range=0.1,
                        zoom_range=.1,
                        horizontal_flip=True
                        )
image_gen.flow_from_directory('Images')
train_image_gen=image_gen.flow_from_directory('Images/train',target_size=(48,48),batch_size=batch,
                                             class_mode='binary')
test_image_gen=image_gen.flow_from_directory('Images/test',target_size=(48,48),batch_size=batch,
                                             class_mode='binary')
import warnings
warnings.filterwarnings('ignore')

classindex=train_image_gen.class_indices
print(classindex)


# In[7]:



steps_for=28273//50
validate=7067//50


result=model.fit_generator(train_image_gen,epochs=100,steps_per_epoch=steps_for,validation_data=test_image_gen,verbose=1,validation_steps=validate)


# In[12]:




# In[18]:


model.save('Trained_Model.h5')



