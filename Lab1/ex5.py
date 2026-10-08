import cv2
import numpy as np
import matplotlib.pyplot as plt

image=cv2.imread("lena.tif")

height,width=image.shape[:2]

center=(width//2,height//2)

angle1=45
angle2=-45

scale=1.0

rotation_matrix1=cv2.getRotationMatrix2D(center,angle1,scale)
rotation_matrix2=cv2.getRotationMatrix2D(center,angle2,scale)

rotated_image1=cv2.warpAffine(image,rotation_matrix1,(width,height))
rotated_image2=cv2.warpAffine(image,rotation_matrix2,(width,height))

cv2.imshow('Rotated 45',rotated_image1)
cv2.imshow('Rotated -45',rotated_image2)
cv2.waitKey(0)
cv2.destroyAllWindows()