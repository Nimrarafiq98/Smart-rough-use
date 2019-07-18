


import cv2,glob
from tkinter import *
from tkinter import filedialog
import time
import numpy as np
from PIL import Image
start= time.time()

#detect = cv2.CascadeClassifier("C:\Python\Python37\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml")


root = Tk()
root.withdraw()
folder_selected = filedialog.askdirectory()
print(folder_selected)
path = str(folder_selected)+'/*.jpg'
print (path)

for bb,timg in enumerate (glob.glob(path)):
    print(bb,timg)
    image = cv2.imread(timg)
    img = np.array(image)

    # Apply a transformation where we multiply each pixel rgb 
    # with the matrix for the sepia

    filt = cv2.transform( img, np.matrix([[ 0.393, 0.769, 0.189],
                                          [ 0.349, 0.686, 0.168],
                                          [ 0.272, 0.534, 0.131]                                  
    ]) )

    # Check wich entries have a value greather than 255 and set it to 255
    filt[np.where(filt>255)] = 255

    # Create an image from the array 
    sepia = Image.fromarray(filt)

    #face = detect.detectMultiScale(gray,1.20,5)
  

    #for(x,y,w,h) in face:
        #cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
   


    path = 'C:\pythoncheck\My check\detect{}.jpg'
    cv2.imwrite(path.format(bb),sepia)
    
end = time.time()
print(end-start)
