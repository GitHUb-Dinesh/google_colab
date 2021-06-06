#imports
import numpy as np
import math
import cv2
capture=cv2.VideoCapture(0)
while capture.isOpened():
    #capture frame from  camera
    #ret,frame=capture.read()
    frame=cv2.imread('hand_palm2.jpg')
    #cv2.resize(frame,(600,600),interpolation=cv2.INTER_NEAREST)
    cv2.rectangle(frame,(30,30),(545,545),(248,25,62),1)#put your hands inside redbox
    cv2.putText(frame,'put your hand in specified bluebox',(0,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255))
    
    crop_image=frame[30:545,30:545]
    crop_image=crop_image.copy()
    #cv2.imshow('crop_image',crop_image)
    #print(crop_image.shape[:2])
    #thresh=cv2.cvtColor(crop_image,cv2.COLOR_BGR2GRAY)
    #cv2.imshow('thresh',thresh)
    #ret,img=cv2.threshold(thresh,50,255, cv2.THRESH_BINARY_INV)# it will not work for making distinguishable, gesture image from background
    #cv2.imshow('binariesed',img)
    blur=cv2.GaussianBlur(crop_image,(5,5),0)
    #print(type(blur))
    ##cv2.imshow("blur",blur)
    #color conversion from bgr to hsv hue(0-179),saturation(0-255),value(0-255)
    hsv=cv2.cvtColor(blur,cv2.COLOR_BGR2HSV)
    ##cv2.imshow('hsv',hsv)
    #creating a binary image white is skin and rest is black
    lower_color=np.array([2,0,0])#np.array([108,23,82]) 
    upper_color=np.array([20,255,255])#np.array([179,255,255])
    mask1=cv2.inRange(hsv,lower_color,upper_color)
    ##cv2.imshow('mask1',mask1)
    #morphological transformations
    kernel=np.ones((3,3))
    open_mask1=cv2.morphologyEx(mask1,cv2.MORPH_OPEN,kernel,3)
    ##cv2.imshow('open_mask1',open_mask1)
    #close_mask1=cv2.morphologyEx(mask1,cv2.MORPH_CLOSE,kernel,3)
    #cv2.imshow('close_crop_image',close_mask1)
    #define kernel
    #kernel=np.ones((5,5))
    #marphological operation
    #dilate=cv2.dilate(mask1,kernel,iterations=1)
    #cv2.imshow('dilate',dilate)
    #dl_er=cv2.erode(dilate,kernel,iterations=1)
    #cv2.imshow('dl_er',dl_er)
    #gaussian blur and threshold
    filtered=cv2.GaussianBlur(open_mask1.copy(),(3,3),0)
    ret,thresh=cv2.threshold(filtered,127,255,0)
    #show threshold image
    ##cv2.imshow('thresholded&blured_mask1',thresh)
    contours,heirarchy=cv2.findContours(thresh.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    #cv2.drawContours(crop_image,contours,-1,(0,255,0),2)
    ##l=[cv2.contourArea(x) for x in contours]
    ##print(l,len(contours))
    #cv2.imshow('drawn_contours',crop_image)
    try: 
        #find contour with maximum area
        contour=max(contours,key=lambda x:cv2.contourArea(x))
        #creating bounding rectangle arounding maximum area contour
        area_of_hand=cv2.contourArea(contour)
        x,y,w,h=cv2.boundingRect(contour)
        img=frame[x:x+w,y:y+h+50]
        cv2.imshow("img",img)
        area_of_boundingRect=w*h
        determiner_ratio=area_of_hand/area_of_boundingRect
        print(determiner_ratio)
        
        cv2.rectangle(crop_image,(x,y),(x+w,y+h),(0,0,255),1)
        #find convex hull
        
        hull=cv2.convexHull(contour)
        #draw contour blank image
        drawing=np.ones(crop_image.shape)
        cv2.drawContours(crop_image,[contour],-1,(0,255,0),1)
        cv2.drawContours(crop_image,[hull],-1,(45,214,234),2)
        
        ##cv2.imshow("drawing", drawing)
        #find convexity defects
        hull=cv2.convexHull(contour,returnPoints=False)
        defects=cv2.convexityDefects(contour,hull)
        #print(defects)[[[strt_point, endpiont,fathest_point,distance of convex hull from farthest point]],.....]
        #using cosine find angle of embedded of defect point(farpoint)
        count_defects=0
        
        ##print(defects.shape)#19,1,4 for standard image of hanad
        l=defects[1,0]
        #print(tuple(contour[l[0]][0]))
        for i in range(len(defects)):

            s,e,f,d=defects[i,0]
            
            start=tuple(contour[s][0])# x,y points of starting point of defect i
            end=tuple(contour[e][0])
            far=tuple(contour[f][0])
            #d is farthest point distance from convex hull
            a=math.sqrt((end[0]-start[0])**2+(end[1]-start[1])**2)
            b=math.sqrt((far[0]-start[0])**2+(far[1]-start[1])**2)
            c=math.sqrt((end[0]-far[0])**2+(end[1]-far[1])**2)
            angle=(math.acos((b**2+c**2-a**2)/(2*b*c))*180)/math.pi
            #cv2.imshow("Wcrop_image_without circle",crop_image)
            if angle<=90:
                count_defects=count_defects+1
                cv2.circle(crop_image,far,2,[0,23,255],-1)
                #print(count_defects)
            cv2.line(crop_image,start,far,[123,128,45],1)
            cv2.line(crop_image,end,far,[123,128,45],1)
        cv2.imshow("crop_image",crop_image)
        #print number of defects
        print("count_defects", count_defects)
        if count_defects==0:
            cv2.putText(frame,'1-One',(0,60),cv2.FONT_HERSHEY_SIMPLEX,1,(24,255,45))
        elif count_defects==1:
            cv2.putText(frame,'2-Two',(0,60),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255))
        elif count_defects==2:
            cv2.putText(frame,'3-Three',(0,60),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255))

        elif count_defects==3:
            cv2.putText(frame,'4-Four',(0,60),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255))
        elif count_defects==4:
            cv2.putText(frame,'5-Five',(0,60),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255))
        
        else:
            cv2.putText(frame,'Please, put your finger in the right spot',(0,60),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),2)
            pass
    except:
        print('HELLO !!')
        pass
    cv2.imshow('frame',frame)
    #show required image
    #cv2.imshow('Gesture',frame)
    all_image=np.hstack((drawing,crop_image))
    #cv2.imshow('contours',all_image)
    #closed camera if q pressed
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
capture.release()
cv2.destroyAllWindows()
    



        

        
    
