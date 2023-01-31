import numpy as np
import cv2

filename = 'video_2'
cap = cv2.VideoCapture('video/'+filename+'.mp4')
fourcc = cv2.VideoWriter_fourcc(*'MP4V')

background_subtractor = cv2.createBackgroundSubtractorKNN()
#Player scores
p1, p2 = 0, 0
if filename == 'video_1':
    down, up = True, False
    out = cv2.VideoWriter('output1.mp4', fourcc, 20.0, (1221,980))
else:
    down, up = False, True
    height = int(cap.get(4))
    out = cv2.VideoWriter('output2.mp4', fourcc, 20.0, (1121,height))


while True:
    ret, frame = cap.read()

    #  Define a Region of interest for each video
    if filename == 'video_1':
        roi = frame[100:, 396:1617]
    else:
        roi = frame[:, 396:1517]

    frame = roi

    #Apply the background subtraction to get the mask
    mask = background_subtractor.apply(roi)

    #Apply a threshold
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)

    #Apply Erosion
    kernel1 = np.ones((4,4),np.uint8)
    mask = cv2.erode(mask,kernel1,iterations = 1)

    #Apply Dilation
    kernel2 = np.ones((5,5),np.uint8)
    mask = cv2.dilate(mask,kernel2,iterations = 10)

    # Get contours
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # cv2.imshow('Frame', mask)

    for cnt in contours:
        # Calculate area and remove small elements
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        #Filtering for the first video
        if filename== 'video_1':
            if area>1500:
                if (area<3500 and y>100 and x>200):
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 100, 255), 3)
                    if (y>250 and y<1600 and up == True):
                        up = False
                        down = True
                        p1= p1+ 1   #count hit
                        cv2.putText(frame, "Hit", (500,500),0,3,(255,10,255),10)
                    elif (y<250 and y>150 and up == False):
                        up = True
                        down = False
                        p2= p2+1    #count hit
                        cv2.putText(frame, "Hit", (500,500),0,3,(255,10,255),10)
                else:
                    x, y, w, h = cv2.boundingRect(cnt)
                    if x>300:
                        if (y>300):
                            cv2.rectangle(frame, (x-20, y-20), (x + w+20, y + h+20), (0, 255, 0), 3)
                        else:
                            cv2.rectangle(frame, (x-20, y-20), (x + w+20, y + h+20), (255, 0, 0), 3)

        #Filtering for the second video
        else:
            if area > 1500 and x>290:
                if area < 2700 and y>100:
                    cv2.rectangle(frame, (x,y), (x+w, y+h), (0,100,255),3)
                    #check for hit
                    if (y>250 and y<1600 and up == True):
                        up=False
                        down=True
                        p1= p1+ 1   #count hit
                        cv2.putText(frame, "Hit", (500,500),0,3,(255,10,255),10)        
                    elif (y<250 and y>150 and up == False):
                        up = True
                        down = False
                        p2= p2+1    #count hit
                        cv2.putText(frame, "Hit", (500,500),0,3,(255,10,255),10)
                else:
                    x, y, w, h = cv2.boundingRect(cnt)
                    if x>300:
                        if (y>300):
                            cv2.rectangle(frame, (x-20, y-20), (x + w+20, y + h+20), (0, 255, 0), 3)
                        else:
                            cv2.rectangle(frame, (x-20, y-20), (x + w+20, y + h+20), (255, 0, 0), 3)

    cv2.putText(frame, "Player1: "+str(p1), (800, 50), 0, 1, (255,0,0), 4)
    cv2.putText(frame, "Player2: "+str(p2), (800, 100), 0, 1, (0,255,0), 4)
    cv2.putText(frame, "Player1", (200, 120), 0, 1, (255,0,0), 4)
    cv2.putText(frame, "Player2", (200, 850), 0, 1, (0,255,0), 4)

    out.write(frame)
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
