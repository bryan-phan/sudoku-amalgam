from pathlib import Path

import cv2
import numpy as np


class Scan:
    def __init__(self, image_path=None, grid_size=9, max_detection_height=960):
        root = Path(__file__).resolve().parent.parent
        default_path = root / "cv" / "test_imgs" / "angled.jpg"

        self.image_path = Path(image_path) if image_path is not None else default_path
        self.grid_size = grid_size
        self.max_detection_height = max_detection_height

        self.original = None
        self.warped = None
        self.cells = None

    def load_image(self):
        image = cv2.imread(str(self.image_path))
        if image is None:
            raise FileNotFoundError(f"Could not read image: {self.image_path}")

        self.original = image
        return image

    def _resize_for_detection(self, image):
        height = image.shape[0]
        if height <= self.max_detection_height:
            return image, 1.0

        scale = self.max_detection_height / float(height)
        resized = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        return resized, scale

    def _preprocess_for_contours(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        return edges

    def find_grid_contour(self, image=None):
        if image is None:
            image = self.original if self.original is not None else self.load_image()

        resized, scale = self._resize_for_detection(image)
        edges = self._preprocess_for_contours(resized)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_contour = None
        best_area = 0.0

        for contour in contours:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue

            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            if len(approx) == 4 and area > best_area:
                best_area = area
                best_contour = approx

        if best_contour is None:
            raise ValueError("Could not find a 4-corner Sudoku grid contour.")

        return (best_contour.reshape(4, 2).astype(np.float32)) / scale

    def order_points(self, points):
        points = points.reshape(4, 2).astype(np.float32)
        ordered = np.zeros((4, 2), dtype=np.float32)

        sums = points.sum(axis=1)
        diffs = np.diff(points, axis=1)

        ordered[0] = points[np.argmin(sums)]
        ordered[2] = points[np.argmax(sums)]
        ordered[1] = points[np.argmin(diffs)]
        ordered[3] = points[np.argmax(diffs)]

        return ordered

    def warp(self, image=None, contour=None):
        if image is None:
            image = self.original if self.original is not None else self.load_image()

        if contour is None:
            contour = self.find_grid_contour(image)

        src = self.order_points(contour)

        width_top = np.linalg.norm(src[1] - src[0])
        width_bottom = np.linalg.norm(src[2] - src[3])
        height_left = np.linalg.norm(src[3] - src[0])
        height_right = np.linalg.norm(src[2] - src[1])

        width = int(max(width_top, width_bottom))
        height = int(max(height_left, height_right))

        dst = np.array(
            [
                [0, 0],
                [width - 1, 0],
                [width - 1, height - 1],
                [0, height - 1],
            ],
            dtype=np.float32,
        )

        matrix = cv2.getPerspectiveTransform(src, dst)
        self.warped = cv2.warpPerspective(image, matrix, (width, height))
        return self.warped

    def splice(self, warped=None):
        if warped is None:
            warped = self.warped if self.warped is not None else self.warp()

        height, width = warped.shape[:2]
        cell_h = height // self.grid_size
        cell_w = width // self.grid_size

        cells = np.empty((self.grid_size, self.grid_size), dtype=object)

        for row in range(self.grid_size):
            for col in range(self.grid_size):
                y1 = row * cell_h
                y2 = (row + 1) * cell_h if row < self.grid_size - 1 else height
                x1 = col * cell_w
                x2 = (col + 1) * cell_w if col < self.grid_size - 1 else width
                cells[row, col] = warped[y1:y2, x1:x2].copy()

        self.cells = cells
        return cells

    def extract_cells(self):
        image = self.original if self.original is not None else self.load_image()
        warped = self.warped if self.warped is not None else self.warp(image=image)
        return self.splice(warped)

    def export_cells(self, output_dir):
        cells = self.cells if self.cells is not None else self.extract_cells()
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for row in range(self.grid_size):
            for col in range(self.grid_size):
                save_path = output_dir / f"cell_{row}_{col}.png"
                cv2.imwrite(str(save_path), cells[row, col])

        return output_dir
