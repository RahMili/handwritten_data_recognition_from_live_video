#Support Vector Machine (SVM)
#importing the librries
import cv2 
import pytesseract 
import numpy as np
import struct
import pandas as pd
import matplotlib.pyplot as plt
#from sklearn import neighbors, metrics

#importing the dataset
dataset = pd.read_csv('A_Z_Handwritten_Data.csv')
X = dataset.iloc[:, 1:].values
Y = dataset.iloc[:, 1].values

#splitting the dataset into train and test
#from sklearn.model_selection import train_test_split
#X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

#Feature Scaling
#from sklearn.preprocessing import StandardScaler
#sc_X = StandardScaler()
#X_train = sc_X.fit_transform(X_train)
#X_test = sc_X.fit_transform(X_test)

#Fitting SVM to the dataset
from sklearn import neighbors
classifier = neighbors.KNeighborsClassifier()
classifier.fit(X, Y)


#predicting the test set results using webcam
#Y_pred = classifier.predict(X_test)
#open cv code here
cap=cv2.VideoCapture(0)

while True:
    ret,test_img=cap.read()# captures frame and returns boolean value and captured image
    if not ret:
        continue
    gray_img= cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

    ret, thresh1 = cv2.threshold(gray_img, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV) 

    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18)) 
    dilation = cv2.dilate(thresh1, rect_kernel, iterations = 1) 
    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL,  
                                                 cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        i=0;
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,0,0),thickness=7)
        cropped = dilation[y:y + h, x:x + w] #cropping region of interest i.e. face area from  image
        cropped2=cv2.resize(cropped,(28,28))
        Y_img = np.reshape(cropped2, (1, 28*28))

        Y_pred = classifier.predict(Y_img)
        
        numbers = ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z')
        predicted_number = numbers[Y_pred[i]]
        cv2.putText(test_img, predicted_number, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        i=i+1
    resized_img = cv2.resize(test_img, (600, 400))
    cv2.imshow('Handwritten digit prediction',resized_img)



    if cv2.waitKey(10) == ord('q'):#wait until 'q' key is pressed
        break

cap.release()
cv2.destroyAllWindows
    
    


#Making the confusion matrix
#from sklearn.metrics import confusion_matrix
#cm = confusion_matrix(Y_test, Y_pred)
