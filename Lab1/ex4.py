import cv2
import numpy as np
import matplotlib.pyplot as plt

image=cv2.imread("lena.tif")

w=np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
a=np.array([0,1,2])

filters={}
for i in a:
    w[1][1]+=i
    filters[i]=cv2.filter2D(image,ddepth=-1,kernel=w)
    filters[i]=cv2.cvtColor(filters[i],cv2.COLOR_BGR2RGB)
    w[1][1]-=i

plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.imshow(filters[0])
plt.title("filter 0")
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(filters[1])
plt.title("filter 1")
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(filters[2])
plt.title("filter 2")
plt.axis('off')

plt.tight_layout() 
plt.show()

