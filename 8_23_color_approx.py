import pydicom
import matplotlib
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
import pandas as pd
from pytesseract import Output
import pytesseract
import torch

path = './20230807 PWI source example/230802(5870)/CBF/CBF0001.dcm'

dcm=pydicom.dcmread(path)

img=dcm.pixel_array
#img=cv.cvtColor(img,cv.COLOR_BGR2RGB)

gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
ret,thresh = cv.threshold(gray,150,255,0)
contours, hierarchy = cv.findContours(thresh, 1,2)

print("Number of contours detected:", len(contours))

for cnt in contours:
   approx = cv.approxPolyDP(cnt, 0.01*cv.arcLength(cnt, True), True)
   if len(approx) == 4:
      x, y, w, h = cv.boundingRect(cnt)
      if h>100:
        print('FIND!')
        box_img = cv.drawContours(img, [cnt], -1, (255,255,255), 1)
        print(' x={}, y={}, w={}, h={}'.format(x,y,w,h))

# plt.subplot(1,2,2)
# plt.imshow(img) #129 ~ 382
# plt.show()

# cv.imshow("Shapes", box_img)
# cv.waitKey(0)
# cv.destroyAllWindows()

print(img.shape)
max=246.
min=3.85
step=(max-min)/254.
dictionary={}
bar_rgb=[]
for i in range(254):
    dictionary[tuple(img[129+i,501])]=max-step*i
    bar_rgb.append(img[129+i,501])

bar_rgb=np.array(bar_rgb)
    
count_fail=0
count_success=0

check=np.full((512,512,3),0)

print(check.shape)
for i in range(512):
    for j in range(512):
        if tuple(img[i,j]) in dictionary:
            count_success=count_success+1
            continue
        else :
            tmp1=np.int16(img[i,j])
            count_fail=count_fail+1
            min=255*3
            near=img[i,j]
            gap=np.int16(0)
            for key in bar_rgb:
                tmp2=np.int16(key)
                gap=abs(tmp1[0]-tmp2[0])+abs(tmp1[1]-tmp2[1])+abs(tmp1[2]-tmp2[2]) #abs() 절대값 함수
                if gap<min: #user 입력값과 numbers 내의 값의 차이가 앞선 최소값보다 적으면, 그 때의 값과 차이를 near, min으로 덮어쓴다
                    near=key
                    min=gap
            if min >= 10 :
                check[i,j]=[255,255,255]
                
check=cv.cvtColor(check,cv.COLOR_RGB2GRAY)
text = pytesseract.image_to_string(check)
print(text)

plt.imshow(check)
plt.show()
            
