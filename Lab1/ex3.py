import cv2
import numpy as np
import matplotlib.pyplot as plt

image=cv2.imread("lena.tif")

image_rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

kernel1 = np.array([[ 0, -1,  0], [-1,  5, -1], [ 0, -1,  0]])
kernel2 = np.array([[-1, -1, -1], [-1,  9, -1], [-1, -1, -1]])
sharp1=cv2.filter2D(image,ddepth=-1,kernel=kernel1)
sharp2=cv2.filter2D(image,ddepth=-1,kernel=kernel2)
blur1=cv2.blur(image,(5,5))
blur2=cv2.blur(image,(7,7))

sharp1_rgb = cv2.cvtColor(sharp1, cv2.COLOR_BGR2RGB)
sharp2_rgb = cv2.cvtColor(sharp2, cv2.COLOR_BGR2RGB)
blur1_rgb = cv2.cvtColor(blur1, cv2.COLOR_BGR2RGB)
blur2_rgb = cv2.cvtColor(blur2, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.imshow(image_rgb)
plt.title("Original")
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(sharp1_rgb)
plt.title("Sharpen 1")
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(sharp2_rgb)
plt.title("Sharpen 2")
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(blur1_rgb)
plt.title("Blur 1")
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(blur2_rgb)
plt.title("Blur 2")
plt.axis('off')


plt.tight_layout() 
plt.show()


