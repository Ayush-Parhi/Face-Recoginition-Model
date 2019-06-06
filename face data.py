 
import numpy as np
import cv2
cam=cv2.VideoCapture(0)
facec=cv2.CascadeClassifier('C:/opencv/build/etc/haarcascades/haarcascade_frontalface_default.xml')
data=[]
ix=0
while True:
    ret,fr=cam.read()
    if ret == True:
        gray=cv2.cvtColor(fr,cv2.COLOR_BGR2GRAY)
        faces=facec.detectMultiScale(gray,1.3,7)
        for (x,y,w,h) in faces:
            fc=fr[y:y+h,x:x+w,:]
            r = cv2.resize(fc,(50,50))
            if ix%10==0 and len(data)<20:
                print (ix)
                data.append(r)
            cv2.rectangle(fr,(x,y),(x+w,y+h),(255,255,255),2)
        ix+=1
        cv2.imshow('frame',fr)
        if cv2.waitKey(1)==27 or len(data)>=20:
            break
    else:
        print "error"
        break
        
cv2.destroyAllWindows()

data=np.asarray(data)
np.save('face1',data)
