from pathlib import Path

import cv2
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    Image = None
    ImageDraw = None
    ImageFont = None


class DigitRecognizer:
    def __init__(self, template_size=32):
        self.template_size = template_size
        self.crop_ratio = 0.12
        self.min_component_area = 35
        self.min_component_darkness = 95.0
        self.min_blackhat_response = 20.0
        self.max_match_score = 0.10
        self.min_score_gap = 0.04
        self.template_bank = self._build_template_bank()

    def _load_gray(self, image_or_path):
        if isinstance(image_or_path, (str, Path)):
            image = cv2.imread(str(image_or_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise FileNotFoundError(f"Could not read image: {image_or_path}")
            return image

        if image_or_path.ndim == 3:
            return cv2.cvtColor(image_or_path, cv2.COLOR_BGR2GRAY)

        return image_or_path.copy()

    def _crop_inner(self, gray, crop_ratio=None):
        if crop_ratio is None:
            crop_ratio = self.crop_ratio

        height, width = gray.shape
        margin_x = max(2, int(width * crop_ratio))
        margin_y = max(2, int(height * crop_ratio))

        cropped = gray[margin_y : height - margin_y, margin_x : width - margin_x]
        if cropped.size == 0:
            return gray

        return cropped

    def _normalize_mask(self, mask):
        points = cv2.findNonZero(mask)
        if points is None:
            return None

        x, y, width, height = cv2.boundingRect(points)
        crop = mask[y : y + height, x : x + width]

        side = max(width, height) + 8
        canvas = np.zeros((side, side), dtype=np.uint8)
        y_offset = (side - height) // 2
        x_offset = (side - width) // 2
        canvas[y_offset : y_offset + height, x_offset : x_offset + width] = crop

        return cv2.resize(canvas, (self.template_size, self.template_size), interpolation=cv2.INTER_AREA)

    def _extract_digit_mask(self, image_or_path, crop_ratio=None):
        gray = self._load_gray(image_or_path)
        roi = self._crop_inner(gray, crop_ratio=crop_ratio)
        roi_height, roi_width = roi.shape

        blur = cv2.GaussianBlur(roi, (3, 3), 0)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        blackhat = cv2.morphologyEx(blur, cv2.MORPH_BLACKHAT, kernel)

        _, binary = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((2, 2), dtype=np.uint8))

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

        best_mask = None
        best_darkness = 0.0
        best_response = 0.0
        best_score = float("-inf")

        for label in range(1, num_labels):
            x, y, width, height, area = stats[label]
            if area < self.min_component_area:
                continue

            if width < 3 or height < int(roi_height * 0.25):
                continue

            if width > int(roi_width * 0.85) or height > int(roi_height * 0.95):
                continue

            fill_ratio = area / float(max(width * height, 1))
            if fill_ratio < 0.12 or fill_ratio > 0.82:
                continue

            component = labels == label
            darkness = 255.0 - float(np.mean(roi[component]))
            response = float(np.mean(blackhat[component]))

            touches_border = x <= 1 or y <= 1 or (x + width) >= roi_width - 1 or (y + height) >= roi_height - 1
            border_penalty = 18.0 if touches_border and (width <= 5 or height >= roi_height - 2) else 0.0
            score = response * 6.0 + darkness + area * 0.05 - border_penalty

            if score > best_score:
                mask = np.zeros_like(roi, dtype=np.uint8)
                mask[component] = 255
                best_mask = mask
                best_darkness = darkness
                best_response = response
                best_score = score

        if best_mask is None:
            return None

        if best_darkness < self.min_component_darkness or best_response < self.min_blackhat_response:
            return None

        return self._normalize_mask(best_mask)

    def _count_holes(self, mask):
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is None:
            return 0

        mask_area = max(cv2.countNonZero(mask), 1)
        min_hole_area = max(6.0, mask_area * 0.04)

        hole_count = 0
        for index, contour_info in enumerate(hierarchy[0]):
            if contour_info[3] == -1:
                continue

            if cv2.contourArea(contours[index]) >= min_hole_area:
                hole_count += 1

        return hole_count

    def _bottom_band_ratio(self, mask, band_height=4):
        ink_pixels = cv2.countNonZero(mask)
        if ink_pixels == 0:
            return 0.0

        band = mask[-band_height:, :]
        return float(cv2.countNonZero(band)) / float(ink_pixels)

    def _score(self, candidate, template):
        candidate_small = cv2.resize(candidate, (24, 24), interpolation=cv2.INTER_AREA)
        template_small = cv2.resize(template, (24, 24), interpolation=cv2.INTER_AREA)

        overlap = np.logical_and(candidate_small > 0, template_small > 0).sum()
        union = np.logical_or(candidate_small > 0, template_small > 0).sum()
        iou = overlap / union if union else 0.0

        candidate_contours, _ = cv2.findContours(candidate_small, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        template_contours, _ = cv2.findContours(template_small, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        shape_score = 10.0
        if candidate_contours and template_contours:
            shape_score = cv2.matchShapes(
                max(candidate_contours, key=cv2.contourArea),
                max(template_contours, key=cv2.contourArea),
                cv2.CONTOURS_MATCH_I1,
                0.0,
            )

        pixel_score = cv2.norm(candidate_small, template_small, cv2.NORM_L1) / (24 * 24 * 255.0)
        return float(shape_score + 0.55 * pixel_score - 0.95 * iou)

    def _render_opencv_digit(self, digit, font_face, font_scale, thickness):
        canvas = np.full((96, 96), 255, dtype=np.uint8)
        text = str(digit)
        text_size, baseline = cv2.getTextSize(text, font_face, font_scale, thickness)
        x = (canvas.shape[1] - text_size[0]) // 2
        y = (canvas.shape[0] + text_size[1]) // 2
        cv2.putText(canvas, text, (x, y), font_face, font_scale, 0, thickness, cv2.LINE_AA)
        return self._extract_digit_mask(canvas, crop_ratio=0.05)

    def _font_paths(self):
        font_dir = Path(r"C:\Windows\Fonts")
        preferred_fonts = [
            "arial.ttf",
            "arialbd.ttf",
            "calibri.ttf",
            "calibrib.ttf",
            "cambria.ttc",
            "cambriab.ttf",
            "Candara.ttf",
            "Candarab.ttf",
            "consola.ttf",
            "consolab.ttf",
            "constan.ttf",
            "constanb.ttf",
            "corbel.ttf",
            "corbelb.ttf",
            "georgia.ttf",
            "georgiab.ttf",
            "segoeui.ttf",
            "segoeuib.ttf",
            "tahoma.ttf",
            "tahomabd.ttf",
            "times.ttf",
            "timesbd.ttf",
            "trebuc.ttf",
            "trebucbd.ttf",
            "verdana.ttf",
            "verdanab.ttf",
        ]

        return [font_dir / name for name in preferred_fonts if (font_dir / name).exists()]

    def _render_pil_digit(self, digit, font_path, font_size):
        if Image is None or ImageDraw is None or ImageFont is None:
            return None

        canvas = Image.new("L", (96, 96), 255)
        draw = ImageDraw.Draw(canvas)
        font = ImageFont.truetype(str(font_path), size=font_size)
        bbox = draw.textbbox((0, 0), str(digit), font=font)
        x = (96 - (bbox[2] - bbox[0])) // 2 - bbox[0]
        y = (96 - (bbox[3] - bbox[1])) // 2 - bbox[1]
        draw.text((x, y), str(digit), font=font, fill=0)
        return self._extract_digit_mask(np.array(canvas), crop_ratio=0.05)

    def _build_template_bank(self):
        templates = {digit: [] for digit in range(1, 10)}

        if Image is not None and ImageDraw is not None and ImageFont is not None:
            font_sizes = [42, 48, 54, 60]
            for font_path in self._font_paths():
                for digit in range(1, 10):
                    for font_size in font_sizes:
                        try:
                            template = self._render_pil_digit(digit, font_path, font_size)
                        except OSError:
                            continue

                        if template is not None:
                            templates[digit].append(template)

        if all(templates[digit] for digit in templates):
            return templates

        fonts = [
            cv2.FONT_HERSHEY_SIMPLEX,
            cv2.FONT_HERSHEY_DUPLEX,
            cv2.FONT_HERSHEY_TRIPLEX,
            cv2.FONT_HERSHEY_COMPLEX,
        ]
        font_scales = [1.4, 1.8, 2.2]
        thicknesses = [2, 3, 4]

        for digit in range(1, 10):
            if templates[digit]:
                continue

            for font_face in fonts:
                for font_scale in font_scales:
                    for thickness in thicknesses:
                        template = self._render_opencv_digit(digit, font_face, font_scale, thickness)
                        if template is not None:
                            templates[digit].append(template)

        return templates

    def recognize_digit(self, image_or_path):
        prepared = self._extract_digit_mask(image_or_path)
        if prepared is None:
            return 0, 0.0

        if self._count_holes(prepared) >= 2:
            best_eight = min(self._score(prepared, template) for template in self.template_bank[8])
            return 8, best_eight

        digit_scores = {}
        for digit, templates in self.template_bank.items():
            digit_scores[digit] = min(self._score(prepared, template) for template in templates)

        ranked = sorted(digit_scores.items(), key=lambda item: item[1])
        best_digit, best_score = ranked[0]
        second_score = ranked[1][1]

        if best_digit == 2:
            seven_score = digit_scores[7]
            if self._bottom_band_ratio(prepared) < 0.09 and (seven_score - best_score) < 0.75:
                return 7, seven_score

        if best_score > self.max_match_score:
            return 0, best_score

        if (second_score - best_score) < self.min_score_gap:
            return 0, best_score

        return best_digit, best_score

    def recognize_grid(self, image_grid_9x9):
        grid_inputs = np.asarray(image_grid_9x9, dtype=object)
        grid = np.zeros((9, 9), dtype=int)
        scores = np.zeros((9, 9), dtype=float)

        for row in range(9):
            for col in range(9):
                digit, score = self.recognize_digit(grid_inputs[row, col])
                grid[row, col] = digit
                scores[row, col] = score

        return grid, scores
