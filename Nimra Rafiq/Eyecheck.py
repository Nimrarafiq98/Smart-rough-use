import cv2,glob
from tkinter import *
from tkinter.filedialog import askdirectory
import os
import time
start = time.time()

detect = cv2.CascadeClassifier("C:\Python\Python37\Lib\site-packages\cv2\data\nose.xml")

root = Tk()
root.withdraw()
folder_selected = askdirectory()
print(folder_selected)
path=str(folder_selected)+'/*.jpg'
print(path)


for bb,timg in enumerate (glob.glob(path)):
    print(bb,timg)
    img = cv2.imread(timg)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
    face = detect.detectMultiScale(gray,1.20,5)
   

    for(x,y,w,h) in face:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
   

    cv2.imshow("Detect Multi Images",img)
    path = 'C:\Python\Python37\detect{}.jpg'
    cv2.imwrite(path.format(bb),img)
    cv2.waitKey(2000)
    cv2.destroyAllWindows()

end = time.time()
print(end - start)


