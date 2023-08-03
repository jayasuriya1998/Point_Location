#Testing Version 1.0

import cv2
import numpy as np

# Load image
img1= cv2.imread('img11.jpg')
img2= cv2.imread('img11.jpg')

img1=cv2.resize(img1,(400,400))
img2=cv2.resize(img2,(400,400))

img2 = cv2.rotate(img2, cv2.ROTATE_180)
r = cv2.selectROI("select the area", img1)
cropped_image = img1[int(r[1]):int(r[1]+r[3]),int(r[0]):int(r[0]+r[2])]

#feature matching
bf=cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

# Convert image to grayscale
gray1 = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
          
# Initialize SIFT detector
sift = cv2.SIFT_create()
#sift = cv2.xfeatures2d.SIFT_create()
#Compute SIFT features from custom keypoints

kp1, des1 =sift.detectAndCompute(gray1,None)#ROI Selecting Area
kp2, des2=sift.detectAndCompute(gray2,None)
##kp2, des2 = sift.detectAndCompute(gray2,None)

matches = bf.match(des1,des2)
print(len(matches))

if matches is None:
    print("NONE")
else:
    print("found")
    matches = sorted(matches, key = lambda x:x.distance)
    mat= matches[0]
    img1_idx = mat.queryIdx
    img2_idx = mat.trainIdx
    #print(img1_idx,img2_idx)
    (x1, y1) = kp1[img1_idx].pt
    (x2, y2) = kp2[img2_idx].pt
    print(f'x1 is {x1} y1 is {y1} ')
    print(f'x2 is {x2} y2 is {y2} ')
    #cv2.circle(frame,center_position,2,(0,0,255),-1)

    ### Draw custom keypoints on the image
    ##img_kp = cv2.drawKeypoints(img1, kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    img3 = cv2.drawMatches(gray1,kp1,gray2,kp2,matches[:1],gray2, flags=2)
    
    
    # Display the result
    cv2.imshow('Custom Keypoints using SIFT', img3)
    cv2.circle(img2,(72,106),2,(0,0,255),-1)
    cv2.imshow('Testing', img2)
cv2.waitKey(0)
cv2.destroyAllWindows()
