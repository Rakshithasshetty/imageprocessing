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
 A histogram is a graph. A graph that shows frequency of anything. Usually histogram have bars that represent frequency of occurring of data in the whole data set.
A Histogram has two axis the x axis and the y axis.
The x axis contains event whose frequency you have to count.
The y axis contains frequency.
The different heights of bar shows different frequency of occurrence of data.  

Applications of Histograms

Histograms has many uses in image processing. The first use as it has also been discussed above is the analysis of the image. We can predict about an image by just looking at its histogram. Its like looking an x ray of a bone of a body.
The second use of histogram is for brightness purposes. The histograms has wide application in image brightness. Not only in brightness, but histograms are also used in adjusting contrast of an image.
Another important use of histogram is to equalize an image.
And last but not the least, histogram has wide use in thresholding. This is mostly used in computer vision.
 ## code
 ##by Code
```
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
```
# output:
![image](https://user-images.githubusercontent.com/77378707/107616896-e641a500-6c74-11eb-9233-b585000a9d3c.png)
  
![image](https://user-images.githubusercontent.com/77378707/107616976-0b361800-6c75-11eb-8388-ce119f483188.png)

# 14.program to enhance image using image arithmetic and logical operations.
  Image arithmetic applies one of the standard arithmetic operations or a logical operator to two or more images. The operators are applied in a pixel-by-pixel way, i.e. the value of a pixel in the output image depends only on the values of the corresponding pixels in the input images. Hence, the images must be of the same size. Although image arithmetic is the most simple form of image processing, there is a wide range of applications. A main advantage of arithmetic operators is that the process is very simple and therefore fast. 
Logical operators are often used to combine two (mostly binary) images. In the case of integer images, the logical operator is normally applied in a bitwise way.
  ## Addition
```
import cv2  
import numpy as np  
image1 = cv2.imread('app.jpg')  
image2 = cv2.imread('chrome.jpg')
weightedSum = cv2.addWeighted(image1, 0.5, image2, 0.4, 0)
cv2.imshow('Weighted Image', weightedSum)
if cv2.waitKey(0) & 0xff == 25:  
    cv2.destroyAllWindows() 
```
![image](https://user-images.githubusercontent.com/77378707/107617825-9663dd80-6c76-11eb-9a5b-04ec000a1aa8.png)

  # Subtract
```  
import cv2  
import numpy as np  
image1 = cv2.imread('chrome.jpg')  
image2 = cv2.imread('app.jpg')
sub = cv2.subtract(image1, image2)
cv2.imshow('Subtracted Image', sub)
if cv2.waitKey(0) & 0xff == 27:  
    cv2.destroyAllWindows() 
 ```   
![image](https://user-images.githubusercontent.com/77378707/107618006-d9be4c00-6c76-11eb-97f0-e3d1caa84709.png)

# logical operations
```
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
``` 
![image](https://user-images.githubusercontent.com/77378707/107618190-2c980380-6c77-11eb-9c07-659d17106b98.png)
![image](https://user-images.githubusercontent.com/77378707/107618255-446f8780-6c77-11eb-8192-f127b976af34.png)
![image](https://user-images.githubusercontent.com/77378707/107618344-6ec14500-6c77-11eb-8e9f-62a78c4dd45c.png)
![image](https://user-images.githubusercontent.com/77378707/107618410-8ac4e680-6c77-11eb-83c9-8b869d0f7e61.png)
![image](https://user-images.githubusercontent.com/77378707/107618468-9f08e380-6c77-11eb-84c3-fce8a841dcfb.png)


# 15. program for gray level slicing with an without background.
Grey level slicing is equivalent to band pass filtering. It manipulates group of intensity levels in an
image up to specific range by diminishing rest or by leaving them alone. This transformation is applicable in
medical images and satellite images such as X-ray flaws, CT scan. Two different approaches are adopted for
grey level slicing [6][7].
1) Grey level slicing without background: It displays high values in the specific region of an image and low
value to other regions by ignoring background. grey levels by reducing all others to a constant level. 
2) Grey level slicing with background: by preserving all other levels. 
displays high values in specific region of an image and original grey level to other region by preserving
background.
 ## code
  ### with background
```
import cv2
import numpy as np
from matplotlib import pyplot as plt
image=cv2.imread('app.jpg',0)
x,y=image.shape
z=np.zeros((x,y))
for i in range(0,x):
    for j in range(0,y):
        if(image[i][j]>50 and image[i][j]<150):
            z[i][j]=255
        else:
            z[i][j]=image[i][j]
equ=np.hstack((image,z))
plt.title('Original ||   Graylevel slicing with background')
plt.imshow(equ,'gray')
plt.show()
```
![image](https://user-images.githubusercontent.com/77378707/107618990-71706a00-6c78-11eb-9c61-668eeb89746f.png)

### without background
```
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

image=cv2.imread('app.jpg',0)
x,y=image.shape
z=np.zeros((x,y))
for i in range(0,x):
    for j in range(0,y):
        if(image[i][j]>50 and image[i][j]<150):
            z[i][j]=255
        else:
            z[i][j]=0
equ=np.hstack((image,z))
plt.title('Original ||  Graylevel slicing w/o background')
plt.imshow(equ,'gray')
plt.show()
```
![image](https://user-images.githubusercontent.com/77378707/107619139-b8f6f600-6c78-11eb-88f2-2dc5919043da.png)

# 16. program for an image enhancement using histogram equalisation.
Histogram Equalization
Histogram Equalization is a computer image processing technique used to improve contrast in images. It accomplishes this by effectively spreading out the most frequent intensity values, i.e. stretching out the intensity range of the image. This method usually increases the global contrast of images when its usable data is represented by close contrast values. This allows for areas of lower local contrast to gain a higher contrast.

A color histogram of an image represents the number of pixels in each type of color component. Histogram equalization cannot be applied separately to the Red, Green and Blue components of the image as it leads to dramatic changes in the image’s color balance. However, if the image is first converted to another color space, like HSL/HSV color space, then the algorithm can be applied to the luminance or value channel without resulting in changes to the hue and saturation of the image.


```
from matplotlib.pyplot import imread, imshow, show, subplot, title, get_cmap, hist
from skimage.exposure import equalize_hist
import numpy as np


img = imread('tom.jpg')
eq = np.asarray(equalize_hist(img) * 255, dtype='uint8')

subplot(221); imshow(img, cmap=get_cmap('gray')); title('Original')
subplot(222); hist(img.flatten(), 256, range=(0,256)); title('Histogram of origianl')
subplot(223); imshow(eq, cmap=get_cmap('gray'));  title('Histogram Equalized')
subplot(224); hist(eq.flatten(), 256, range=(0,256));

show()
```
![image](https://user-images.githubusercontent.com/77378707/107619652-813c7e00-6c79-11eb-80de-e107df63d6a1.png)


# 17. To perform following operation in an image 
   ##  1.opening
   ##  2.closing
   Opening and closing are two important operators from mathematical morphology. They are both derived from the fundamental operations of erosion and deletion. Like those operators they are normally applied to binary images, although there are also graylevel versions.  Opening is generally used to remove small objects from the foreground (usually taken as the bright pixels) of an image, placing them in the background whereas Closing is generally used to remove small holes in the foreground, changing small islands of background into foreground.

The combination of opening and closing is generally used to clean up artifacts in the segmented image before using the image for digital analysis.

## What You Need To Know About Opening In Digital Image Processing
Opening is a process in which first erosion operation is performed and then dilation operation is performed.
Opening removes small objects from the foreground (usually taken as the bright pixels) of an image, placing them in the background.
Opening preserves the shape and size of larger objects in the image.
Opening is used for removing internal noise of the obtained image.
Opening eliminates the thin protrusions of the obtained image.
The opening operation erodes an image and then dilates the eroded image using the same structuring element for both operations.
Operation performed on X & Y is represented by (AoB).
Opening operation performed on X & Y is the union of all translations of Y that fit entirely within X.
## What You Need To Know About Closing In Digital Image Processing
Closing is a process in which first dilation operation is performed and then erosion operation is performed.
Closing removes small holes in the foreground, changing small islands of background into foreground.
Closing preserves the shape and size of larger objects in the image.
Closing is used for smoothening of contour and fusing of narrow breaks.
Closing eliminates the small holes from the obtained image.
The closing operation dilates an image and then erodes the dilated image, using the same structuring element for both operations.
Closing operation performed on X & Y is represented by (A.B).
Closing operation performed on X & Y is the complement of the union of all translations of Y that do not fit entirely within X.
```
import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
img=cv2.imread('car.jpg')
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()
img2=cv2.imread('car.jpg')

kernel=np.ones((4,4),np.uint8)

OPEN=cv2.morphologyEx(img2, cv2.MORPH_OPEN,kernel)
plt.show()
plt.title('OPEN image')
plt.imshow(cv2.cvtColor(OPEN, cv2.COLOR_BGR2RGB))
plt.show()

close=cv2.morphologyEx(img2, cv2.MORPH_CLOSE,kernel)
plt.show()
plt.title('close image')
plt.imshow(cv2.cvtColor(close, cv2.COLOR_BGR2RGB))
plt.show()
```
![image](https://user-images.githubusercontent.com/77378707/107619989-02941080-6c7a-11eb-9ca8-a8c703222f57.png)
![image](https://user-images.githubusercontent.com/77378707/107620014-0de73c00-6c7a-11eb-84d5-b22cc1fe4549.png)
![image](https://user-images.githubusercontent.com/77378707/107620032-16d80d80-6c7a-11eb-820a-9e55eb03a3c9.png)



