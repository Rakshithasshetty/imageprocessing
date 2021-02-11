# Imageprocessing 
# 1. Develop a program to display gray scale image using read and write operation.
## Grayscale Images:
A grayscale (or graylevel) image is simply one in which the only colors are shades of gray. The reason for differentiating such images from any other sort of color image is that less information needs to be provided for each pixel. In fact a `gray' color is one in which the red, green and blue components all have equal intensity in RGB space, and so it is only necessary to specify a single intensity value for each pixel, as opposed to the three intensities needed to specify each pixel in a full color image.
Grayscale to RGB Conversion, Grayscale to RGB Conversion - We have already define the RGB color model and gray scale format in our tutorial of Image types. Now we will convert an color​  Average method is the most simple one. You just have to take the average of three colors. Since its an RGB image, so it means that you have add r with g with b and then divide it by 3 to get your desired grayscale image. Its done in this way.
## code
## importing opencv
```
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
```
## output
![image](https://user-images.githubusercontent.com/77378707/104426913-96ee5300-55a8-11eb-989f-d1845ba29496.png)
![image](https://user-images.githubusercontent.com/77378707/104427064-cdc46900-55a8-11eb-9bff-54d046ab28a3.png)



# 2. Develop a program to perform linear transformation on an image.
 # linear transformation
Piece-wise Linear Transformation is type of gray level transformation that is used for image enhancement. It is a spatial domain method. It is used for manipulation of an image so that the result is more suitable than the original for a specific application.
# a)Scaling
In computer graphics and digital imaging, image scaling refers to the resizing of a digital image. ... When scaling a raster graphics image, a new image with a higher or lower number of pixels must be generated. In the case of decreasing the pixel number (scaling down) this usually results in a visible quality loss.
# b)Rotation 
Image rotation is a common image processing routine with applications in matching, alignment, and other image-based algorithms. ... An image rotated by 45°. The output is the same size as the input, and the out of edge values are dropped.
## a)scaling 1

```
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
```
## output:

![image](https://user-images.githubusercontent.com/77378707/104428156-20525500-55aa-11eb-907b-32511e3626c7.png)

## scaling 2
```
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
```
## output:

![image](https://user-images.githubusercontent.com/77378707/105331179-29ba6d80-5bf9-11eb-90c6-2e0a60874197.png)

## b)Rotation
```
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
```
## output:
![image](https://user-images.githubusercontent.com/77378707/105331350-5e2e2980-5bf9-11eb-92eb-64b8b5f7ccba.png)







     

# 3. Develop a program to find the sum and mean of a set of images
In digital image processing, the sum of absolute differences (SAD) is a measure of the similarity between image blocks. It is calculated by taking the absolute difference between each pixel in the original block and the corresponding pixel in the block being used for comparison. These differences are summed to create a simple metric of block similarity, the L1 norm of the difference image or Manhattan distance between two image blocks.
Consider a set of scalar observation v1,…,vn. Assume we have good reasons to believe that all these values really should be the same (and equal to some unknown value t) but due to measurement errors, natural variation or some other unknown disturbances the values are not the same.

## code
```
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
```
## output:

![image](https://user-images.githubusercontent.com/77378707/105331378-67b79180-5bf9-11eb-8f2c-1fbe327fb1a2.png)
![image](https://user-images.githubusercontent.com/77378707/105331414-6ede9f80-5bf9-11eb-83d7-60edd3e5e992.png)

# 4. Convert color image gray scale to binary image
   Thresholding is the simplest method of image segmentation and the most common way to convert a grayscale image to a binary image. ... Here g(x, y) represents threshold image pixel at (x, y) and f(x, y) represents greyscale image pixel at (x, y).

## code
```
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
```

## output:
![image](https://user-images.githubusercontent.com/77378707/105331448-79993480-5bf9-11eb-9c67-2177fc6bf15e.png)
![image](https://user-images.githubusercontent.com/77378707/105331480-83229c80-5bf9-11eb-98d1-05c8988e47de.png)
![image](https://user-images.githubusercontent.com/77378707/105331508-8ae24100-5bf9-11eb-9a38-31d6c604a2b8.png)

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
```
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
```
## output:
![image](https://user-images.githubusercontent.com/77378707/104432582-2e56a480-55af-11eb-87b1-ae2d8f8e84ca.png)
![image](https://user-images.githubusercontent.com/77378707/104432764-56de9e80-55af-11eb-96be-dcd0dd8b2073.png)
![image](https://user-images.githubusercontent.com/77378707/104432879-78d82100-55af-11eb-9067-a644c6f3831c.png)
![image](https://user-images.githubusercontent.com/77378707/104433243-e2f0c600-55af-11eb-94a4-fee9756617cf.png)

# 6. Develop a program to create an image from 2D array generate an array of random size.
A digital image is nothing more than data—numbers indicating variations of red, green, and blue at a particular location on a grid of pixels. Most of the time, we view these pixels as miniature rectangles sandwiched together on a computer screen. With a little creative thinking and some lower level manipulation of pixels with code, however, we can display that information in a myriad of ways. This tutorial is dedicated to breaking out of simple shape drawing in Processing and using images (and their pixels) as the building blocks of Processing graphics.
## code
```
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
```
##  output:
![image](https://user-images.githubusercontent.com/77378707/104896815-a0f3c580-599d-11eb-9c0d-fea25d56b158.png)

![image](https://user-images.githubusercontent.com/77378707/104897062-f0d28c80-599d-11eb-9c7e-7f36af28f207.png)


## code
```
import numpy as np
from PIL import Image
array = np.linspace(0,1,256*256)
mat = np.reshape(array,(256,256))
img = Image.fromarray( mat , 'HSV')
img.show()
```
### output:
![image](https://user-images.githubusercontent.com/77378707/104897226-224b5800-599e-11eb-9d9a-e32b610b7d0b.png)

# 7. Find the neighbors matrix.
A pixel's neighborhood is some set of pixels, defined by their locations relative to that pixel, which is called the center pixel. The neighborhood is a rectangular block, and as you move from one element to the next in an image matrix, the neighborhood block slides in the same direction.

## code
```
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
```
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

# 8. Develop a program to find the sum of neighbour of each elements in the matrix
Given a M x N matrix, find sum of all K x K sub-matrix 2. Given a M x N matrix and a cell (i, j), find sum of all elements of the matrix in constant time except the elements present at row i & column j of the matrix. Given a M x N matrix, calculate maximum sum submatrix of size k x k in a given M x N matrix in O (M*N) time. Here, 0 < k < M, N.
## code
```
import numpy as np
ini_array = np.array([[1, 2,5, 3], [4,5, 4, 7], [9, 6, 1,0]])
print("initial_array : ", str(ini_array));
def neighbors(radius, rowNumber, columnNumber):
    return[[ini_array[i][j]if i >= 0 and i < len(ini_array) and j >= 0 and j < len(ini_array[0]) else 0
            for j in range(columnNumber-1-radius, columnNumber+radius)]
           for i in range(rowNumber-1-radius, rowNumber+radius)]
neighbors(1, 2, 2)
```
##  Output
initial_array :  [[1 2 5 3]
 [4 5 4 7]
 [9 6 1 0]]
[[1, 2, 5], [4, 5, 4], [9, 6, 1]]


# 9.Develop a program to implement  Negative transformation

### Image Negatives (Negative Transformation)
The negative of an image with gray level in the range
[0, L-1], where L = Largest value in an image, is
obtained by using the negative transformation’s
expression:
s = L – 1 – r
Which reverses the intensity levels of an input
image , in this manner produces the equivalent of a
photographic negative.
The negative transformation is suitable for enhancing
white or gray detail embedded in dark regions of an
image, especially when the black area are dominant
in size
Advantages of negative :
  Produces an equivalent of a photographic negative.
  Enhances white or gray detail embedded in dark regions.
## code
```
  import cv2
import numpy as np
img=cv2.imread('cat.jpg')
cv2.imshow('original',img)
cv2.waitKey(0)
img_neg=255-img
cv2.imshow('negative',img_neg)
cv2.waitKey(0)
```
## output:
![image](https://user-images.githubusercontent.com/77378707/105326562-e0b3ea80-5bf3-11eb-9f47-42d3f2e631cd.png)
![image](https://user-images.githubusercontent.com/77378707/105326596-e90c2580-5bf3-11eb-91ef-c4460331677b.png)

# 10.Develop a program to implement brightness thresholding.
The simplest thresholding methods replace each pixel in an image with a black pixel if the image intensity I i,j is less than some fixed constant T (that is, {\displaystyle Ii,j< T Ii,j < T, or a white pixel if the image intensity is greater than that constant. In the example image on the right, this results in the dark tree becoming completely black, and the white snow becoming completely white.

## code

```
import cv2  
import numpy as np  
image1 = cv2.imread('apple.jpg')  
img = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY) 
  
ret, thresh1 = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY) 
ret, thresh2 = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY_INV) 
ret, thresh3 = cv2.threshold(img, 120, 255, cv2.THRESH_TRUNC) 
ret, thresh4 = cv2.threshold(img, 120, 255, cv2.THRESH_TOZERO) 
ret, thresh5 = cv2.threshold(img, 120, 255, cv2.THRESH_TOZERO_INV) 

cv2.imshow('Binary Threshold', thresh1) 
cv2.imshow('Binary Threshold Inverted', thresh2) 
cv2.imshow('Truncated Threshold', thresh3) 
cv2.imshow('Set to 0', thresh4) 
cv2.imshow('Set to 0 Inverted', thresh5) 
  
if cv2.waitKey(0) & 0xff == 27:  
    cv2.destroyAllWindows()  
```
![image](https://user-images.githubusercontent.com/77378707/105324526-8dd93380-5bf1-11eb-951d-684f09bd0c31.png)
![image](https://user-images.githubusercontent.com/77378707/105324574-99c4f580-5bf1-11eb-8833-26164b841faf.png)
![image](https://user-images.githubusercontent.com/77378707/105324609-a3e6f400-5bf1-11eb-9a87-2f525afa6cb6.png)
![image](https://user-images.githubusercontent.com/77378707/105324771-dabd0a00-5bf1-11eb-8c88-398455eebf22.png)
![image](https://user-images.githubusercontent.com/77378707/105324886-fd4f2300-5bf1-11eb-871c-a346bdd82263.png)

# 11.Program to implement contrast enchancement.
Image enhancement techniques have been widely used in many applications of image processing where the subjective quality of images is important for human interpretation. Contrast is an important factor in any subjective evaluation of image quality. Contrast is created by the difference in luminance reflected from two adjacent surfaces. In other words, contrast is the difference in visual properties that makes an object distinguishable from other objects and the background. In visual perception, contrast is determined by the difference in the colour and brightness of the object with other objects. Our visual system is more sensitive to contrast than absolute luminance; therefore, we can perceive the world similarly regardless of the considerable changes in illumination conditions. Many algorithms for accomplishing contrast enhancement have been developed and applied to problems in image processing.
## code
```
from PIL import Image, ImageEnhance
img = Image.open("car.jpg")
img.show()
img=ImageEnhance.Color(img)
img.enhance(2.0).show()
```
## output:
![image](https://user-images.githubusercontent.com/77378707/105329329-0b537280-5bf7-11eb-8bc3-c7f6a1b25139.png)
![image](https://user-images.githubusercontent.com/77378707/105329391-1d351580-5bf7-11eb-981c-0713df32f1ce.png)

# 12.Program to implement power law(gamma) transformation.
A variety of devices for image capture, printing, and display respond according to a power law. The exponent in power law equation is referred to as gamma Þ process used to correct this power law response phenomena is called gamma correction. eg. CRT devices have intensity.
Gamma correction is extremely important as use of digital images for commercial purposes over the internet has increased.
There are further two transformation is power law transformations, that include nth power and nth root transformation. These transformations can be given by the expression:

s=cr^γ

This symbol γ is called gamma, due to which this transformation is also known as gamma transformation.

Variation in the value of γ varies the enhancement of the images. Different display devices / monitors have their own gamma correction, that’s why they display their image at different intensity.

This type of transformation is used for enhancing images for different type of display devices. The gamma of different display devices is different. For example Gamma of CRT lies in between of 1.8 to 2.5, that means the image displayed on CRT is dark.

Correcting gamma.
s=cr^γ

s=cr^(1/2.5)

The same image but with different gamma values has been shown here.
## code
```
import numpy as np
import cv2
img = cv2.imread('apple.jpg')
gamma_two_point_two = np.array(230*(img/255)**2.1,dtype='uint8')
gamma_point_four = np.array(255*(img/255)**0.1,dtype='uint8')
img3 = cv2.hconcat([gamma_two_point_two,gamma_point_four])
cv2.imshow('a2',img3)
cv2.waitKey(0)
```
## output: 
![image](https://user-images.githubusercontent.com/77378707/105330801-ba447e00-5bf8-11eb-8cf6-97563d01046e.png)


# 13.Histogram of an image:
   a)through your code
   b)through the built in function
   c)to verify a) and b) are one and the same.
   
 ## code
 #by Code
import cv2
from matplotlib import pyplot as plt
img = cv2.imread('img1.jpg',0)
histr = cv2.calcHist([img],[0],None,[256],[0,256])
plt.plot(histr)
plt.show()
#by function
import cv2
from matplotlib import pyplot as plt
img = cv2.imread('img1.jpg',0)
plt.hist(img.ravel(),256,[0,256])
plt.show()
# output:
![image](https://user-images.githubusercontent.com/77378707/107616896-e641a500-6c74-11eb-9233-b585000a9d3c.png)
  
![image](https://user-images.githubusercontent.com/77378707/107616976-0b361800-6c75-11eb-8388-ce119f483188.png)

# 14.program to enhance image using image arithmetic and logicb operations.
  ## Addition
import cv2  
import numpy as np  
image1 = cv2.imread('app.jpg')  
image2 = cv2.imread('chrome.jpg')
weightedSum = cv2.addWeighted(image1, 0.5, image2, 0.4, 0)
cv2.imshow('Weighted Image', weightedSum)
if cv2.waitKey(0) & 0xff == 25:  
    cv2.destroyAllWindows() 

![image](https://user-images.githubusercontent.com/77378707/107617825-9663dd80-6c76-11eb-9a5b-04ec000a1aa8.png)

  # Subtract
 import cv2  
import numpy as np  
image1 = cv2.imread('chrome.jpg')  
image2 = cv2.imread('app.jpg')
sub = cv2.subtract(image1, image2)
cv2.imshow('Subtracted Image', sub)
if cv2.waitKey(0) & 0xff == 27:  
    cv2.destroyAllWindows() 
    
![image](https://user-images.githubusercontent.com/77378707/107618006-d9be4c00-6c76-11eb-97f0-e3d1caa84709.png)

# logical operations
import cv2
img1 = cv2.imread("app.jpg")
img2 = cv2.imread("chrome.jpg")
bitwise_and = cv2.bitwise_and(img2, img1)
cv2.imshow("bit_and", bitwise_and)
bitwise_or = cv2.bitwise_or(img2, img1)
cv2.imshow("bitwise_or", bitwise_or)
bitwise_xor = cv2.bitwise_xor(img2, img1)
cv2.imshow("bitwise_xor", bitwise_xor)
bitwise_not = cv2.bitwise_not(img1)
cv2.imshow("bitwise_not1", bitwise_not)
bitwise_not = cv2.bitwise_not(img2)
cv2.imshow("bitwise_not2", bitwise_not)
cv2.waitKey(0)
cv2.destroyAllWindows()


    
