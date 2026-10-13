import cv2
import numpy as np
import matplotlib.pyplot as plt

def cropping(image,x_start,y_start,crop_width,crop_height):
    height,width=image.shape[:2]
    if(y_start<0 or x_start<0):
        print("Coordinates must be positive")
        return None
    if(y_start+crop_height>height or x_start+crop_width>width):
        print("Coordinates must not exceed image dimensions")
        return None
    cropped=image[y_start:y_start+crop_height,x_start:x_start+crop_width]
    return cropped



image=cv2.imread("lena.tif")

cropped_image=cropping(image,0,0,300,300)
cropped_image2=cropping(image,-10,0,300,300)

cv2.imshow('Cropped',cropped_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
