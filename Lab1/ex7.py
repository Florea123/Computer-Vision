import cv2
import numpy as np

image = np.ones((500, 500, 3), dtype=np.uint8) * 255

center_x, center_y = 250, 250

pct_ureche_stanga = np.array([[50, 150], [90, 20], [190, 110]], np.int32)
cv2.fillPoly(image, [pct_ureche_stanga], (207, 198, 251))
cv2.polylines(image, [pct_ureche_stanga], True, (0, 0, 0), 5)

pct_ureche_dreapta = np.array([[450, 150], [410, 20], [310, 110]], np.int32)
cv2.fillPoly(image, [pct_ureche_dreapta], (207, 198, 251))
cv2.polylines(image, [pct_ureche_dreapta], True, (0, 0, 0), 5)

cv2.circle(image, (center_x, center_y), 200,(207, 198, 251), -1)
cv2.circle(image, (center_x, center_y), 200, (0, 0, 0), 5)

ochiul_stang = (170, 180)
ochiul_drept = (330, 180)
raza_ochi = 25
cv2.circle(image, ochiul_stang, raza_ochi, (0, 0, 0), -1)
cv2.circle(image, ochiul_drept, raza_ochi, (0, 0, 0), -1)


cv2.ellipse(image, (250, 250), (60, 45), 0, 0, 360, (180, 150, 220), -1)
cv2.ellipse(image, (250, 250), (60, 45), 0, 0, 360, (0, 0, 0), 4) 
cv2.circle(image, (230, 250), 10, (0, 0, 0), -1)
cv2.circle(image, (270, 250), 10, (0, 0, 0), -1)



centru_gura = (250, 330)
axe_elipsa = (60, 40) 
cv2.ellipse(image, centru_gura, axe_elipsa, 0, 0, 180, (0, 0, 0), 8)





cv2.imshow("Emoji", image)
cv2.imwrite("Florea_Robert-Andrei.jpg", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
