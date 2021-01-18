# Imageprocessing 
# 1. Develop a program to display gray scale image using read and write operation.
## Grayscale Images:
A grayscale (or graylevel) image is simply one in which the only colors are shades of gray. The reason for differentiating such images from any other sort of color image is that less information needs to be provided for each pixel. In fact a `gray' color is one in which the red, green and blue components all have equal intensity in RGB space, and so it is only necessary to specify a single intensity value for each pixel, as opposed to the three intensities needed to specify each pixel in a full color image.
Grayscale to RGB Conversion, Grayscale to RGB Conversion - We have already define the RGB color model and gray scale format in our tutorial of Image types. Now we will convert an color​  Average method is the most simple one. You just have to take the average of three colors. Since its an RGB image, so it means that you have add r with g with b and then divide it by 3 to get your desired grayscale image. Its done in this way.
## code
## importing opencv
Import cv2
## Load our input image
Image = cv2.imread ("cat.jpg")
cv2.imshow ('Original', image)
## we use cvtColor, to convert to grayscale
gray_image = cv2.cvtColor (image, cv2.COLOR_BGR2GRAY)
cv2.imshow ('Grayscale', gray_image)
cv2.imwrite(‘graying.jpg’, gray_image)
cv2.waitKey (0)
## Window shown waits for any key pressing event
cv2.destroyAllWindows ()

## output
![image](https://user-images.githubusercontent.com/77378707/104426913-96ee5300-55a8-11eb-989f-d1845ba29496.png)
![image](https://user-images.githubusercontent.com/77378707/104427064-cdc46900-55a8-11eb-9bff-54d046ab28a3.png)



# 2. Develop a program to perform linear transformation on an image.
 # linear transformation
Piece-wise Linear Transformation is type of gray level transformation that is used for image enhancement. It is a spatial domain method. It is used for manipulation of an image so that the result is more suitable than the original for a specific application.
# a).Scaling
In computer graphics and digital imaging, image scaling refers to the resizing of a digital image. ... When scaling a raster graphics image, a new image with a higher or lower number of pixels must be generated. In the case of decreasing the pixel number (scaling down) this usually results in a visible quality loss.
# b)Rotation 
Image rotation is a common image processing routine with applications in matching, alignment, and other image-based algorithms. ... An image rotated by 45°. The output is the same size as the input, and the out of edge values are dropped.
## a)scaling 1
import cv2 as c
import numpy as np
image = c.imread("image.jpg")
gray = c.cvtColor(image,c.COLOR_BGR2RGB)
h,w = image.shape[0:2]
width = int(w * .5)
hight = int(h *.5)
res = c.resize(image,(width,hight))
c.imshow("Fist Lab",res)
c.waitKey(0)
c.destroyAllWindows()

## output:

![image](https://user-images.githubusercontent.com/77378707/104428156-20525500-55aa-11eb-907b-32511e3626c7.png)

## scaling 2
import cv2 as c
import numpy as np
image = c.imread("image.jpg")
gray = c.cvtColor(image,c.COLOR_BGR2RGB)
h,w = image.shape[0:2]
width = int(w * 2)
hight = int(h *.5)
res = c.resize(image,(width,hight))
c.imshow("Fist Lab",res)
c.waitKey(0)
c.destroyAllWindows()
## output:

![image](https://user-images.githubusercontent.com/77378707/104428496-9060db00-55aa-11eb-95dd-d0d589ee3bdf.png)

## b).Rotation

import cv2 
import numpy as np
image = cv2.imread("image.jpg")
gray = cv2.cvtColor(image,c.COLOR_BGR2RGB)
h,w = image.shape[0:2]
rotationMatrix = cv2.getRotationMatrix2D((w/2, h/2), 200, .5)
rotated_image = cv2.warpAffine(image,rotationMatrix,(w,h))
cv2.imshow("Fist Lab",rotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

## output:


![image](https://user-images.githubusercontent.com/77378707/104429179-53e1af00-55ab-11eb-9c3f-508c7d5d76ce.png)
     

# 3. Develop a program to find the sum and mean of a set of images
In digital image processing, the sum of absolute differences (SAD) is a measure of the similarity between image blocks. It is calculated by taking the absolute difference between each pixel in the original block and the corresponding pixel in the block being used for comparison. These differences are summed to create a simple metric of block similarity, the L1 norm of the difference image or Manhattan distance between two image blocks.
Consider a set of scalar observation v1,…,vn. Assume we have good reasons to believe that all these values really should be the same (and equal to some unknown value t) but due to measurement errors, natural variation or some other unknown disturbances the values are not the same.

## code  
import cv2
import os
path = 'D:\Pictures'
imgs = []

files = os.listdir(path)
for file in files:
    filepath=path+"\\"+file
    imgs.append(cv2.imread(filepath))
i=0
im = []
for im in imgs:
   
    im+=imgs[i]
    i=i+1
cv2.imshow("sum of four picturesmeanImg ", im)
meanImg=im/len(files)
cv2.imshow("mean of four pictures",meanImg)
cv2.waitKey(0)

## output:

![image](https://user-images.githubusercontent.com/77378707/104430506-c3a46980-55ac-11eb-948b-e9025352fbae.png) 
![image](https://user-images.githubusercontent.com/77378707/104430607-e9317300-55ac-11eb-836c-8e9e49ef876c.png)

# 4. Convert color image gray scale to binary image
Thresholding is the simplest method of image segmentation and the most common way to convert a grayscale image to a binary image. ... Here g(x, y) represents threshold image pixel at (x, y) and f(x, y) represents greyscale image pixel at (x, y).

## code
import cv2 
originalImage= cv2.imread("image.jpg")
grayImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)
(thresh, bw_img) = cv2.threshold(originalImage, 120, 200, cv2.THRESH_BINARY)
cv2.imshow('Original image',originalImage)
cv2.waitKey(0)
cv2.imshow('Gray image', grayImage)
cv2.waitKey(0)  
cv2.imshow('binary image', bw_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


## output:
![image](https://user-images.githubusercontent.com/77378707/104431519-ff8bfe80-55ad-11eb-8496-2b768687b554.png)


![image](https://user-images.githubusercontent.com/77378707/104431720-395d0500-55ae-11eb-8838-fcae3087fbee.png)

![image](https://user-images.githubusercontent.com/77378707/104431815-598cc400-55ae-11eb-8582-5170772f3d47.png)

# 5. convert  a color image to different color space
   ## What are color spaces?
Color spaces are different types of color modes, used in image processing and signals and system for various purposes. Some of the common color spaces are:

RGB
CMY’K
Y’UV
YIQ
Y’CbCr
HSV
Color space conversion is the translation of the representation of a color from one basis to another. This typically occurs in the context of converting an image that is represented in one color space to another color space, the goal being to make the translated image look as similar as possible to the original.
## code
import cv2
image=cv2.imread('cat.jpg')
cv2.imshow('pic',image)
cv2.waitKey(0)
image1 = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
cv2.imshow('img1',image1)
cv2.waitKey(0)
image2 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
cv2.imshow('img2',image2)
cv2.waitKey(0)
image3 = cv2.cvtColor(image,cv2.COLOR_RGB2YUV)
cv2.imshow('img2',image3)
cv2.waitKey(0)
cv2.destroyAllWindows()

## output:
![image](https://user-images.githubusercontent.com/77378707/104432582-2e56a480-55af-11eb-87b1-ae2d8f8e84ca.png)
![image](https://user-images.githubusercontent.com/77378707/104432764-56de9e80-55af-11eb-96be-dcd0dd8b2073.png)
![image](https://user-images.githubusercontent.com/77378707/104432879-78d82100-55af-11eb-9067-a644c6f3831c.png)
![image](https://user-images.githubusercontent.com/77378707/104433243-e2f0c600-55af-11eb-94a4-fee9756617cf.png)

# 6. Develop a program to create an image from 2D array generate an array of random size.
A digital image is nothing more than data—numbers indicating variations of red, green, and blue at a particular location on a grid of pixels. Most of the time, we view these pixels as miniature rectangles sandwiched together on a computer screen. With a little creative thinking and some lower level manipulation of pixels with code, however, we can display that information in a myriad of ways. This tutorial is dedicated to breaking out of simple shape drawing in Processing and using images (and their pixels) as the building blocks of Processing graphics.
## code
import numpy as np
from PIL import Image
import cv2
array = np.linspace(0,1,256*256) 
mat = np.reshape(array,(256,256))
img = Image.fromarray(np.uint8(mat * 255) , 'L')
img.show()
cv2.waitKey(0)
array = np.linspace(0,1,256*256) 
mat = np.reshape(array,(256,256)) 
img = Image.fromarray( mat , 'L')
img.show()
cv2.waitKey(0)
##  output:
![image](https://user-images.githubusercontent.com/77378707/104434628-74146c80-55b1-11eb-9313-35f379bf65de.png)

![image](https://user-images.githubusercontent.com/77378707/104434230-0405e680-55b1-11eb-9f17-b0ef297b175e.png)


## code
import numpy as np
from PIL import Image
array = np.linspace(0,1,256*256)
mat = np.reshape(array,(256,256))
img = Image.fromarray( mat , 'HSV')
img.show()

### output:
![image](https://user-images.githubusercontent.com/77378707/104434947-da998a80-55b1-11eb-807b-ee98c94f7034.png)

# 7. Find the neighbors matrix.
A pixel's neighborhood is some set of pixels, defined by their locations relative to that pixel, which is called the center pixel. The neighborhood is a rectangular block, and as you move from one element to the next in an image matrix, the neighborhood block slides in the same direction.

## code

import numpy as np

axis = 3
x =np.empty((axis,axis))
y = np.empty((axis+2,axis+2))
s =np.empty((axis,axis))
x = np.array([[1,4,3],[2,8,5],[3,4,6]])


'''
for i in range(0,axis):
    for j in range(0,axis):
        print(int(x[i][j]),end = '\t')
    print('\n')'''

print('Temp matrix\n')

for i in range(0,axis+2):
    for j in range(0,axis+2):
        if i == 0 or i == axis+1 or j == 0 or j==axis+1:
            y[i][j]=0
        else:
            #print("i = {}, J = {}".format(i,j))
            y[i][j]=x[i-1][j-1]
           

for i in range(0,axis+2):
    for j in range(0,axis+2):
        print(int(y[i][j]),end = '\t')
    print('\n')
   
   
print('Output calculated Neigbhors of matrix\n')      
for i in range(0,axis):
    for j in range(0,axis):
        s[i][j]=((y[i][j]+y[i][j+1]+y[i][j+2]+y[i+1][j]+y[i+1][j+2]+y[i+2][j]+y[i+2][j+1]+y[i+2][j+2])/8)
        print(s[i][j],end = '\t')
    print('\n')

## output:
Temp matrix

0	0	0	0	0	

0	1	4	3	0	

0	2	8	5	0	

0	3	4	6	0	

0	0	0	0	0	

Output calculated Neigbhors of matrix

1.75	2.375	2.125	

2.5	3.5	3.125	

1.75	3.0	2.125	

# 8. To find sum of nieghbor matrix
Given a M x N matrix, find sum of all K x K sub-matrix 2. Given a M x N matrix and a cell (i, j), find sum of all elements of the matrix in constant time except the elements present at row i & column j of the matrix. Given a M x N matrix, calculate maximum sum submatrix of size k x k in a given M x N matrix in O (M*N) time. Here, 0 < k < M, N.
## code
import numpy as np def sumNeighbors(M,x,y): l = [] for i in range(max(0,x-1),x+2): # max(0,x-1), such that no negative values in range() for j in range(max(0,y-1),y+2): try: t = M[i][j] l.append(t) except IndexError: # if entry doesn't exist pass return sum(l)-M[x][y] # exclude the entry itself

M = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

M = np.asarray(M) N = np.zeros(M.shape)

for i in range(M.shape[0]): for j in range(M.shape[1]): N[i][j] = sumNeighbors(M, i, j)

print("Original matrix:\n",M) print("Summed neighbors matrix:\n",N)

##  Output
Original matrix: 

[[1 2 3]

[4 5 6] 

[7 8 9]] 

Summed neighbors matrix: 

[[11. 19. 13.]

[23. 40. 27.]

[17. 31. 19.]]
