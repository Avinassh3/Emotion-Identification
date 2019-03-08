
import numpy as np
import pandas as pd
import cv2
import os

data=pd.read_csv('dataset.csv')
pixels=data['pixels'].tolist()
width,height=48,48
faces=[]
emoj=data['emotion'].tolist()
typ=data['Usage'].tolist()

for x in pixels:
    face=[int(j) for j in x.split(' ')]
    face=np.asarray(face).reshape(width,height)
    a=(48,48)
    faces.append(face)
faces = np.asarray(faces)
emotions = emoj


d={0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0}
emo=['Angry/','Disgrace/','Fear/','Happy/','Neutral/','Sad/','Surprise/']
cls=['test','train']
for i in range(len(emotions)):
        get=emotions[i]
        #print('visited')
        val=str(d[emotions[i]])
        if(typ[i]=='Training'):
            cv2.imwrite('Images/train/'+emo[get]+val+'.jpg',faces[i])
            d[emotions[i]]=d[emotions[i]]+1
        else:
            cv2.imwrite('Images/test/'+emo[get]+val+'.jpg',faces[i])
            d[emotions[i]]=d[emotions[i]]+1            
print(d)
