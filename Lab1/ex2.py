import cv2
import numpy as np
import matplotlib.pyplot as plt

image=cv2.imread("lena.tif")
cv2.imshow("Ex2",image)
height,width,channel=image.shape
print(height,width,channel)
cv2.waitKey(0)
cv2.destroyAllWindows()

image_rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
plt.imshow(image_rgb)
plt.title("Plot Ex2")
plt.axis('on')
plt.show()
