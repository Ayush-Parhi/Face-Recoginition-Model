import numpy as np
import cv2
cam=cv2.VideoCapture(0)
font=cv2.FONT_HERSHEY_COMPLEX
facec=cv2.CascadeClassifier('C:/opencv/build/etc/haarcascades/haarcascade_frontalface_default.xml')
f_01=np.load('face1.npy').reshape((20,-1))
f_02=np.load('face2.npy').reshape((20,-1))
f_03=np.load('face3.npy').reshape((20,-1))
names={
    0:'poonam (53 yr)',
    1:'Ayush (19 yr)',
    2:'Shiv ji'
}
data=np.concatenate((f_01,f_02,f_03))
labels=np.zeros((data.shape[0],1))
labels[20:40]=1.0
labels[40:]=2.0
def distance(x1,x2):
    d=np.sqrt(((x1-x2)**2).sum())
    return d
def knn(X_train, y_train, xt, k=7):
    vals = []
    for ix in range(X_train.shape[0]):
        d=distance(X_train[ix],xt)
        vals.append([d, y_train[ix]])
    sorted_labels = sorted(vals ,key=lambda z: z[0])
    neighbours = np.asarray(sorted_labels)[:k, -1]
    freq = np.unique(neighbours, return_counts=True)
    return freq[0][freq[1].argmax()]

while True:
    ret,fr=cam.read()
    if ret == True:
        gray=cv2.cvtColor(fr,cv2.COLOR_BGR2GRAY)
        faces=facec.detectMultiScale(gray,1.3,7)
        for (x,y,w,h) in faces:
            fc=fr[y:y+h,x:x+w,:]
            r = cv2.resize(fc,(50,50)).flatten()
            ans=knn(data,labels,r)
            text=names[int(ans)]
           
            cv2.rectangle(fr,(x,y),(x+w,y+h),(255,255,255),2)
            cv2.putText(fr,text,(x,y),font,1,(255,255,255),2)
        
        cv2.imshow('frame',fr)
        if cv2.waitKey(1)==27:
            break
    else:
        print "error"
        break
        
cv2.destroyAllWindows()
