from keras.models import load_model

model=load_model("Trained_Model.h5")

import numpy as np
import cv2
from keras.preprocessing import image
img = cv2.imread('Images/test/Fear/4199.jpg')
img = cv2.resize(img,(48,48))
img = np.reshape(img,[1,48,48,3])
ar=model.predict(img)
print(" \n Test for Fear Image ")
for i in range(0,len(ar[0])):
    print(ar[0][i]*100)

print("\n Test for Angry Image  ")
img = cv2.imread('Images/test/Angry/4018.jpg')
img = cv2.resize(img,(48,48))
img = np.reshape(img,[1,48,48,3])
ar=model.predict(img)

for i in range(0,len(ar[0])):
    print(ar[0][i]*100)

print(" \n Test for Happy Image  ")
img = cv2.imread('Images/test/Happy/7234.jpg')
img = cv2.resize(img,(48,48))
img = np.reshape(img,[1,48,48,3])
ar=model.predict(img)

for i in range(0,len(ar[0])):
    print(ar[0][i]*100)


print(" \n Test for Sad Image  ")
img = cv2.imread('Images/test/Sad/4971.jpg')
img = cv2.resize(img,(48,48))
img = np.reshape(img,[1,48,48,3])
ar=model.predict(img)

for i in range(0,len(ar[0])):
    print(ar[0][i]*100)