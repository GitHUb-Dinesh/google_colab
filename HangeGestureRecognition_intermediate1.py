#imports
import numpy as np
import math
import cv2
capture=cv2.VideoCapture(0)
while capture.isOpened():
    #capture frame from  camera
    ret,frame=capture.read()
    
    cv2.rectangle(frame,(100,100),(300,300),(255,0,0),1)
    cv2.imshow('frame',frame)
    crop_image=frame[100:300,100:300]
    #cv2.imshow('crop_image',crop_image)
    print(crop_image.shape[:2])
    #thresh=cv2.cvtColor(crop_image,cv2.COLOR_BGR2GRAY)
    #cv2.imshow('thresh',thresh)
    #ret,img=cv2.threshold(thresh,50,255, cv2.THRESH_BINARY_INV)# it will not work for making distinguishable, gesture image from background
    #cv2.imshow('binariesed',img)
    blur=cv2.GaussianBlur(crop_image,(5,5),0)
    #print(type(blur))
    cv2.imshow("blur",blur)
    #color conversion from bgr to hsv hue(0-179),saturation(0-255),value(0-255)
    hsv=cv2.cvtColor(blur,cv2.COLOR_BGR2HSV)
    cv2.imshow('hsv',hsv)
    #creating a binary image white is skin and rest is black
    lower_color=np.array([2,0,0])#np.array([108,23,82]) 
    upper_color=np.array([20,255,255])#np.array([179,255,255])
    mask1=cv2.inRange(hsv,lower_color,upper_color)
    cv2.imshow('mask1',mask1)
    #define kernel
    kernel=np.ones(5,5)
    #marphological operation
    dilate=cv2.dilate(mask2,kernel,iterations=1)
    erosion=cv2.erode(erosion,kernel,iterations=1)
    #gaussian blur and threshold
    filtered=cv2.GaussianBlur(crop_image,(3,3),0)
    ret,thresh=cv2.threshold(filtered,127,255,0)
    #show threshold image
    cv2.imshow('thresholded_input_hand_image',thresh)
    image,contours,heirarchy=cv2.findContours(thresh.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    try:
        #find contour with maximum area
        contour=max(contours,key=lambda x:cv2.contourArea(x))
        #creating bounding rectangle arounding maximum area contour
        x,y,w,h=cv2.boundingRect(contour)
        cv2.rectangle(crop_image,(x,y),(x+w,y+h),(0,0,255),0)
        #find comnvex hull
        hull =cv2.convexHull(contour)
        #draw contour blank image
        drawing=np.zeros(crop_image,np.uint8)
        cv2.drawContours(drawing,[contour],-1,(0,255,0),0)
        cv2.drawContoours(drawing,[hull],-1,(0,0,255),0)
        #find convexity defects
        hull=cv2.convexHull(contour,returnPoints=False)
        defects=cv2.convexityDefects(contour,hull)

        #using cosine find angle of embedded of defect point(farpoint)
        count_defect=0


        for i in range(defects.shape[0]):
            s,e,f,d=defects[i,0]
            start=tuple(contour[s][0])
            end=tuple(contour[e][0])
            far=tuple(contour[f][0])


            a=math.sqrt((end[0]-start[0])**2+(end[1]-start[1])**2)
            b=math.sqrt((far[0]-start[0])**2+(far[1]-start[1])**2)
            a=math.sqrt((end[0]-far[0])**2+(end[1]-far[1])**2)
            angle=(math.acos((b**2+c**2-a**2)/(2*b*c))*180)/3.14
            if angle<=90:
                count_defects+=1
                cv2.circle(crop_image,far,1,[0,0,255],-1)
            cv2.line(crop_image,start,end,[0,255,0],2)
        #print number of defects
        if count_defects==0:
            cv2.puttext(frame,'1-One',(50,50),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),2)
        elif count_defects==1:
            cv2.puttext(frame,'2-Two',(50,50),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),2)
        elif count_defects==2:
            cv2.puttext(frame,'3-Three',(50,50),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),2)

        elif count_defects==3:
            cv2.puttext(frame,'4-Four',(50,50),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),2)
        elif count_defects==4:
            cv2.puttext(frame,'5-Five',(50,50),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),2)
        
        else:
            cv2.puttext(frame,'Please, put your fingre in the right spot',(50,50),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),2)
            pass
    except:
        pass
    #show required image
    cv2.imshow('Gesture',frame)
    all_image=np.hstack((drawing,crop_image))
    cv2.imshow('contours',all_image)
    #closed camera if q pressed
    if cv2.waitKey(1)==ord('q'):
        break
capture.release()
cv2.destroyAllWindows()
    



        

        
    
