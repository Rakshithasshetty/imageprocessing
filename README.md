# imageprocessing 
## 1. Develop a program to display gray scale image using read and write operation.
Grayscale Images
A grayscale (or graylevel) image is simply one in which the only colors are shades of gray. The reason for differentiating such images from any other sort of color image is that less information needs to be provided for each pixel. In fact a `gray' color is one in which the red, green and blue components all have equal intensity in RGB space, and so it is only necessary to specify a single intensity value for each pixel, as opposed to the three intensities needed to specify each pixel in a full color image.
Grayscale to RGB Conversion, Grayscale to RGB Conversion - We have already define the RGB color model and gray scale format in our tutorial of Image types. Now we will convert an color​  Average method is the most simple one. You just have to take the average of three colors. Since its an RGB image, so it means that you have add r with g with b and then divide it by 3 to get your desired grayscale image. Its done in this way.
## code
# importing opencv
Import cv2
# Load our input image
Image = cv2.imread ("cat.jpg")
cv2.imshow ('Original', image)

# we use cvtColor, to convert to grayscale
gray_image = cv2.cvtColor (image, cv2.COLOR_BGR2GRAY)
cv2.imshow ('Grayscale', gray_image)
cv2.imwrite(‘graying.jpg’, gray_image)
cv2.waitKey (0)
# Window shown waits for any key pressing event
cv2.destroyAllWindows ()

## output

![image](https://user-images.githubusercontent.com/77378707/104426913-96ee5300-55a8-11eb-989f-d1845ba29496.png)
![image](https://user-images.githubusercontent.com/77378707/104427064-cdc46900-55a8-11eb-9bff-54d046ab28a3.png)



## 2. Develop a program to perform linear transformation on an image.
    linear transformation
    Piece-wise Linear Transformation is type of gray level transformation that is used for image enhancement. It is a spatial domain method. It is used for manipulation of an image so that the result is more suitable than the original for a specific application.
##  a).Scaling
        In computer graphics and digital imaging, image scaling refers to the resizing of a digital image. ... When scaling a raster graphics image, a new image with a higher or lower number of pixels must be generated. In the case of decreasing the pixel number (scaling down) this usually results in a visible quality loss.
## b)Rotation 
      Image rotation is a common image processing routine with applications in matching, alignment, and other image-based algorithms. ... An image rotated by 45°. The output is the same size as the input, and the out of edge values are dropped.
# code1
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

#output:

![image](https://user-images.githubusercontent.com/77378707/104428156-20525500-55aa-11eb-907b-32511e3626c7.png)

#code2
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
#output:

![image](https://user-images.githubusercontent.com/77378707/104428496-9060db00-55aa-11eb-95dd-d0d589ee3bdf.png)

##b).Rotation

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

#output:


![image](https://user-images.githubusercontent.com/77378707/104429179-53e1af00-55ab-11eb-9c3f-508c7d5d76ce.png)
     

