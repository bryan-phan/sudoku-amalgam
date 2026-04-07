import cv2
import os
import numpy as np

class Scan: 
    def __init__(self):
        self.path = os.getcwd() + r"\cv" + r"\test_imgs" + r"\angled.jpg"

        #apparently i need to store the orignal for later and make a copy of it for now
        self.original = cv2.imread(self.path)
        self.original = cv2.resize(self.original, None, fx=0.5, fy=0.4)

        self.img = self.original.copy()
        #for da 4 corner contour used to warp image to straight
        self.biggest = None
        self.warped = None

        self.grayscale()
        self.blur()
        self.edge()
        self.contours()
        self.show_image()
        self.cells = self.splice()
        self.export_cells()

    def show_image(self):
        cv2.imshow("Image", self.img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def export_cells(self):
        output_dir = os.path.join(os.getcwd(), "ml", "cell_export")
        os.makedirs(output_dir, exist_ok=True)
        for r in range(9):
            for c in range(9):
                cell = self.cells[r][c]
                gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
                small = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)

                file_name = f"cell_{r}_{c}.png"
                save_path = os.path.join(output_dir, file_name)
                cv2.imwrite(save_path, small)

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
        #find the biggest contour since the image mixed hella contours together

        for cnt in contours:
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)

            #approximates the shape with verticies
            approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)

            if area > max_area and len(approx) == 4:
                max_area = area
                biggest = approx

        self.biggest = biggest

        if biggest is not None:
            cv2.drawContours(self.img, [biggest], -1, (0, 255, 0), 2)
            self.warped = self.warp(biggest)

    def order_points(self, points):
        points = points.reshape(4, 2).astype(np.float32)
        ordered = np.zeros((4, 2), dtype=np.float32)

        sums = points.sum(axis=1)
        diffs = np.diff(points, axis=1)

        ordered[0] = points[np.argmin(sums)]   # top-left
        ordered[2] = points[np.argmax(sums)]   # bottom-right
        ordered[1] = points[np.argmin(diffs)]  # top-right
        ordered[3] = points[np.argmax(diffs)]  # bottom-left

        return ordered

    def warp(self, biggest):
        src = self.order_points(biggest)

        width_top = np.linalg.norm(src[1] - src[0])
        width_bottom = np.linalg.norm(src[2] - src[3])
        height_left = np.linalg.norm(src[3] - src[0])
        height_right = np.linalg.norm(src[2] - src[1])

        width = int(max(width_top, width_bottom))
        height = int(max(height_left, height_right))

        dst = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
        ], dtype=np.float32)

        matrix = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(self.original, matrix, (width, height))
    
    def splice(self):
        if self.warped is None:
            return None
        
        height, width = self.warped.shape[:2]

        #divides the image into a 9x9 grid
        cell_h = height // 9
        cell_w = width // 9

        cells = []

        for r in range(9):
            row_cells = []
            for c in range(9):
                y1 = r * cell_h
                y2 = (r + 1) * cell_h
                x1 = c * cell_w
                x2 = (c + 1) * cell_w

                cell = self.warped[y1:y2, x1:x2]
                #inside row_cells are column cells
                row_cells.append(cell)
            #makes the entire grid
            cells.append(row_cells)

        return cells
            
test = Scan()
