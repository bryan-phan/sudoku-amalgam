from pathlib import Path

import cv2
import numpy as np

HEIC_EXTENSIONS = {".heic", ".heif"}

try:
    from PIL import Image, ImageOps
except ImportError:
    Image = None
    ImageOps = None

try:
    from pillow_heif import register_heif_opener
except ImportError:
    register_heif_opener = None
else:
    register_heif_opener()


class Scan:
    def __init__(self, image_path=None, grid_size=9, max_detection_height=960):
        root = Path(__file__).resolve().parent.parent
        default_path = root / "data" / "images" / "angled.jpg"

        self.image_path = Path(image_path) if image_path is not None else default_path
        self.grid_size = grid_size
        self.max_detection_height = max_detection_height

        self.original = None
        self.warped = None
        self.cells = None

    def load_image(self):
        image = cv2.imread(str(self.image_path))
        if image is None:
            image = self._load_with_pillow()
        if image is None:
            if self.image_path.suffix.lower() in HEIC_EXTENSIONS and register_heif_opener is None:
                raise RuntimeError(
                    "HEIC/HEIF support requires pillow-heif. Install it with: "
                    "python -m pip install pillow-heif"
                )
            raise FileNotFoundError(f"Could not read image: {self.image_path}")

        self.original = image
        return image

    def _load_with_pillow(self):
        if Image is None:
            return None

        try:
            image = Image.open(self.image_path)
        except Exception:
            return None

        if ImageOps is not None:
            image = ImageOps.exif_transpose(image)

        if image.mode not in {"RGB", "RGBA"}:
            image = image.convert("RGB")

        array = np.array(image)
        if image.mode == "RGBA":
            return cv2.cvtColor(array, cv2.COLOR_RGBA2BGR)
        return cv2.cvtColor(array, cv2.COLOR_RGB2BGR)

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

    def _find_grid_contour_from_lines(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        inverted = cv2.bitwise_not(gray)
        binary = cv2.adaptiveThreshold(
            inverted,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            31,
            -8,
        )

        height, width = binary.shape
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, max(20, height // 18)))
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, width // 18), 3))
        vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)
        horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
        line_mask = cv2.bitwise_or(vertical, horizontal)
        line_mask = cv2.dilate(
            line_mask,
            cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)),
            iterations=1,
        )

        contours, _ = cv2.findContours(line_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = height * width * 0.05
        best_contour = None
        best_score = 0.0

        for contour in contours:
            x, y, component_width, component_height = cv2.boundingRect(contour)
            box_area = float(component_width * component_height)
            aspect_ratio = component_width / float(max(component_height, 1))
            if box_area < min_area or not 0.65 <= aspect_ratio <= 1.35:
                continue

            score = box_area
            if score > best_score:
                best_score = score
                best_contour = contour

        if best_contour is None:
            return None, 0.0

        return cv2.boxPoints(cv2.minAreaRect(best_contour)).astype(np.float32), best_score

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

        image_area = float(resized.shape[0] * resized.shape[1])
        line_contour, line_area = self._find_grid_contour_from_lines(resized)
        contour_is_too_small = best_area / image_area < 0.10 if image_area else False
        line_is_full_digital_grid = (
            image_area
            and line_area / image_area > 0.90
            and (best_contour is None or line_area > best_area * 3.0)
        )
        if line_contour is not None and (
            best_contour is None
            or line_is_full_digital_grid
            or (contour_is_too_small and line_area > best_area * 1.35)
        ):
            best_contour = line_contour

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

    def _grid_line_binary(self, warped):
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        inverted = cv2.bitwise_not(gray)
        return cv2.adaptiveThreshold(
            inverted,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            31,
            -10,
        )

    def _line_components(self, mask):
        component_mask = (mask > 0).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(component_mask, connectivity=8)
        return count, labels, stats

    def _extract_vertical_line_centers(self, vertical_mask):
        height, width = vertical_mask.shape
        count, labels, stats = self._line_components(vertical_mask)
        centers = []

        for label in range(1, count):
            x, y, component_width, component_height, area = stats[label]
            if area < max(1000, height * 4) or component_height < height * 0.75:
                continue

            points = np.column_stack(np.where(labels == label))
            centers.append(float(np.mean(points[:, 1])))

        centers.sort()

        if len(centers) == 9:
            if centers[0] > width * 0.08:
                centers = [0.0] + centers
            else:
                centers.append(width - 1.0)
        elif len(centers) == 8:
            centers = [0.0] + centers + [width - 1.0]

        return centers if len(centers) == 10 else None

    def _extract_horizontal_line_models(self, horizontal_mask):
        height, width = horizontal_mask.shape
        count, labels, stats = self._line_components(horizontal_mask)
        models = []

        for label in range(1, count):
            x, y, component_width, component_height, area = stats[label]
            if area < max(1000, width // 2) or component_width < width * 0.15:
                continue

            points = np.column_stack(np.where(labels == label))
            y_coords = points[:, 0].astype(np.float32)
            x_coords = points[:, 1].astype(np.float32)
            degree = 2 if np.unique(x_coords).size >= 3 else 1
            coeffs = np.polyfit(x_coords, y_coords, degree)
            models.append((float(np.mean(y_coords)), coeffs.astype(np.float32)))

        models.sort(key=lambda item: item[0])

        line_models = []
        if not models or models[0][0] > height * 0.05:
            line_models.append(np.array([0.0], dtype=np.float32))

        line_models.extend(coeffs for _, coeffs in models)

        if len(line_models) == 9:
            line_models.append(np.array([height - 1.0], dtype=np.float32))

        return line_models if len(line_models) == 10 else None

    def _evaluate_horizontal_line(self, coeffs, x):
        if len(coeffs) == 1:
            return float(coeffs[0])
        return float(np.polyval(coeffs, x))

    def _extract_line_aware_cell(self, warped, x_lines, y_models, row, col):
        height, width = warped.shape[:2]
        x_left = float(x_lines[col])
        x_right = float(x_lines[col + 1])

        y_top_left = self._evaluate_horizontal_line(y_models[row], x_left)
        y_top_right = self._evaluate_horizontal_line(y_models[row], x_right)
        y_bottom_left = self._evaluate_horizontal_line(y_models[row + 1], x_left)
        y_bottom_right = self._evaluate_horizontal_line(y_models[row + 1], x_right)

        src = np.array(
            [
                [x_left, y_top_left],
                [x_right, y_top_right],
                [x_right, y_bottom_right],
                [x_left, y_bottom_left],
            ],
            dtype=np.float32,
        )
        src[:, 0] = np.clip(src[:, 0], 0.0, width - 1.0)
        src[:, 1] = np.clip(src[:, 1], 0.0, height - 1.0)

        target_width = max(24, int(round(max(x_right - x_left, 1.0))))
        target_height = max(
            24,
            int(round(max(y_bottom_left - y_top_left, y_bottom_right - y_top_right, 1.0))),
        )
        dst = np.array(
            [
                [0, 0],
                [target_width - 1, 0],
                [target_width - 1, target_height - 1],
                [0, target_height - 1],
            ],
            dtype=np.float32,
        )

        matrix = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(warped, matrix, (target_width, target_height))

    def _splice_equal(self, warped):
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

        return cells

    def _splice_with_grid_lines(self, warped):
        binary = self._grid_line_binary(warped)
        height, width = binary.shape
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, max(20, height // 12)))
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, width // 12), 3))
        vertical_mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)
        horizontal_mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)

        x_lines = self._extract_vertical_line_centers(vertical_mask)
        y_models = self._extract_horizontal_line_models(horizontal_mask)
        if x_lines is None or y_models is None:
            return None

        cells = np.empty((self.grid_size, self.grid_size), dtype=object)
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                cells[row, col] = self._extract_line_aware_cell(warped, x_lines, y_models, row, col)

        return cells

    def splice(self, warped=None):
        if warped is None:
            warped = self.warped if self.warped is not None else self.warp()

        cells = self._splice_with_grid_lines(warped)
        if cells is None:
            cells = self._splice_equal(warped)

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
