import cv2
import os

class Scan:
    def __init__(self):
        self.path = os.getcwd() + r"\cv" + r"\test_imgs" + r"\angled.jpg"

        self.img = cv2.imread(self.path)
        self.img = cv2.resize(self.img, None, fx=0.5, fy=0.4)

        self.grayscale()
        self.blur()
        self.edge()
        self.contours()
        self.show_image()

    def show_image(self):
        cv2.imshow("Image", self.img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def grayscale(self):
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

    def blur(self):
        #ig (5, 5) is standard
        self.img = cv2.GaussianBlur(self.img, (5, 5), 0)
    
    def edge(self):
        self.img = cv2.Canny(self.img, 50, 150)

    def contours(self):
        #find the contours
        contours, _ = cv2.findContours(self.img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #apply contours
        self.img = cv2.cvtColor(self.img, cv2.COLOR_GRAY2BGR)

        max_area = 0
        biggest = None

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > max_area:
                max_area = area
                biggest = cnt

        cv2.drawContours(self.img, [biggest], -1, (0, 255, 0), 2)

    
    #def warp(self, biggest):




    




test = Scan()