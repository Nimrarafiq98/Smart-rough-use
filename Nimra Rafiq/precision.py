import cv2,glob
from tkinter import *
from tkinter.filedialog import askdirectory
import os
import time

from sklearn.metrics import confusion_matrix
import numpy as np


start = time.time()

detect = cv2.CascadeClassifier("C:\Python\Python37\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml")

root = Tk()
root.withdraw()
folder_selected = askdirectory()
print(folder_selected)
path=str(folder_selected)+'/*.jpg'
print(path)



for bb,timg in enumerate (glob.glob(path)):
    print(bb,timg)
    img = cv2.imread(timg)
    arr = np.asarray(img,dtype="float32")
  
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face = detect.detectMultiScale(gray,1.20,5)
    for(x,y,w,h) in face:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
   

    
    
   
    cv2.imshow("Detect Multi Images",img)
    path = 'C:\Python36\Final Year project\pics\pred\detect{}.jpg'
    cv2.imwrite(path.format(bb),img)
    
    cv2.waitKey(2000)
    cv2.destroyAllWindows()

   
    
   


##with open(r'C:\Python36\Final Year project\actual.txt', 'r') as infile:
##    true_values = [int(i) for i in infile]
##with open(r'C:\Python36\Final Year project\predicted.txt', 'r') as infile:
##    predictions = [int(i) for i in infile]

def getvalues(filename):
    try:
        file= open(filename,'r')
    except IOError:
        print('problem with file',filename)

    values=[]
    for line in file:
        values.append(int(line))
    return values

actual=getvalues('actual.txt')
predicted=getvalues('predicted.txt')

print(actual)
print(predicted)
# Make confusion matrix
confusion = confusion_matrix(actual, predicted)

print(confusion)

accuracy=(confusion[0][0]+confusion[1][1])/(confusion[0][0]+confusion[0][1]+confusion[1][0]+confusion[1][1])
print(accuracy*100)

end = time.time()
print(end - start)


