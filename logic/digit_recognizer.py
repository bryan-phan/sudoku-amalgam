from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    Image = None
    ImageDraw = None
    ImageFont = None


SCORE_SIZE = 24
TEMPLATE_RENDER_SIZE = 96
PREFERRED_FONTS = (
    "arial.ttf", "arialbd.ttf", "calibri.ttf", "calibrib.ttf", "cambria.ttc",
    "cambriab.ttf", "Candara.ttf", "Candarab.ttf", "consola.ttf", "consolab.ttf",
    "constan.ttf", "constanb.ttf", "corbel.ttf", "corbelb.ttf", "georgia.ttf",
    "georgiab.ttf", "segoeui.ttf", "segoeuib.ttf", "tahoma.ttf", "tahomabd.ttf",
    "times.ttf", "timesbd.ttf", "trebuc.ttf", "trebucbd.ttf", "verdana.ttf",
    "verdanab.ttf",
)
PIL_FONT_SIZES = (42, 48, 54, 60)
OPENCV_FONTS = (
    cv2.FONT_HERSHEY_SIMPLEX,
    cv2.FONT_HERSHEY_DUPLEX,
    cv2.FONT_HERSHEY_TRIPLEX,
    cv2.FONT_HERSHEY_COMPLEX,
)
OPENCV_FONT_SCALES = (1.4, 1.8, 2.2)
OPENCV_THICKNESSES = (2, 3, 4)


@dataclass(frozen=True, slots=True)
class ComponentGeometry:
    x: int
    y: int
    width: int
    height: int
    area: int
    fill_ratio: float
    center_dist: float
    touches_border: bool


@dataclass(frozen=True, slots=True)
class MatchFeatures:
    small_mask: np.ndarray
    binary_mask: np.ndarray
    contour: np.ndarray | None


class DigitRecognizer:
    _template_cache = {}
    crop_ratio = 0.12
    fallback_crop_ratios = (0.12, 0.08, 0.05)
    min_component_area = 35
    min_component_darkness = 95.0
    min_blackhat_response = 20.0
    max_match_score = 0.10
    min_score_gap = 0.04
    fallback_min_component_area = 20
    fallback_min_component_darkness = 180.0
    fallback_border_darkness_bonus = 20.0
    fallback_max_match_score = 0.0
    fallback_min_score_gap = 0.02
    _blackhat_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    _open_kernel = np.ones((2, 2), dtype=np.uint8)

    def __init__(self, template_size=32):
        self.template_size = template_size
        self.template_features = self._load_template_cache()

    def _load_template_cache(self):
        cache_key = (type(self), self.template_size)
        cached = self._template_cache.get(cache_key)
        if cached is None:
            cached = self._build_template_features()
            self._template_cache[cache_key] = cached

        return cached

    def _load_gray(self, image_or_path):
        if isinstance(image_or_path, (str, Path)):
            image = cv2.imread(str(image_or_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise FileNotFoundError(f"Could not read image: {image_or_path}")
            return image

        if image_or_path.ndim == 3:
            return cv2.cvtColor(image_or_path, cv2.COLOR_BGR2GRAY)

        return image_or_path

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

    def _prepare_match_features(self, mask):
        small_mask = cv2.resize(mask, (SCORE_SIZE, SCORE_SIZE), interpolation=cv2.INTER_AREA)
        binary_mask = small_mask > 0
        contours, _ = cv2.findContours(small_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = max(contours, key=cv2.contourArea) if contours else None
        return MatchFeatures(small_mask=small_mask, binary_mask=binary_mask, contour=contour)

    def _rank_digit_scores(self, prepared):
        candidate_features = self._prepare_match_features(prepared)
        digit_scores = {
            digit: min(self._score(candidate_features, template) for template in templates)
            for digit, templates in self.template_features.items()
        }
        ranked = sorted(digit_scores.items(), key=lambda item: item[1])
        return digit_scores, ranked

    def _next_ranked_score(self, ranked, excluded_digits):
        return next(score for digit, score in ranked if digit not in excluded_digits)

    def _classify_prepared_mask(self, prepared):
        digit_scores, ranked = self._rank_digit_scores(prepared)

        if self._count_holes(prepared) >= 2:
            best_digit = 8
            best_score = digit_scores[8]
            second_score = self._next_ranked_score(ranked, {8})
        else:
            best_digit, best_score = ranked[0]
            second_score = self._next_ranked_score(ranked, {best_digit})

        if best_digit == 2:
            seven_score = digit_scores[7]
            if self._bottom_band_ratio(prepared) < 0.09 and (seven_score - best_score) < 0.75:
                best_digit = 7
                best_score = seven_score
                second_score = self._next_ranked_score(ranked, {7, 2})

        return best_digit, best_score, second_score

    def _is_confident_primary_match(self, best_score, second_score):
        return best_score <= self.max_match_score and (second_score - best_score) >= self.min_score_gap

    def _is_confident_fallback_match(self, best_score, second_score, meta):
        if (
            best_score > self.fallback_max_match_score
            or (second_score - best_score) < self.fallback_min_score_gap
            or meta["darkness"] < self.fallback_min_component_darkness
        ):
            return False

        if meta["touches_border"]:
            return meta["darkness"] >= (
                self.fallback_min_component_darkness + self.fallback_border_darkness_bonus
            )

        return True

    def _component_geometry(self, stats_row, roi_height, roi_width):
        x, y, width, height, area = (int(value) for value in stats_row)
        return ComponentGeometry(
            x=x,
            y=y,
            width=width,
            height=height,
            area=area,
            fill_ratio=area / float(max(width * height, 1)),
            center_dist=(
                abs((x + width / 2.0) - roi_width / 2.0) / float(max(roi_width, 1))
                + abs((y + height / 2.0) - roi_height / 2.0) / float(max(roi_height, 1))
            ),
            touches_border=(
                x <= 1
                or y <= 1
                or (x + width) >= roi_width - 1
                or (y + height) >= roi_height - 1
            ),
        )

    def _passes_component_bounds(
        self,
        geometry,
        *,
        min_area,
        min_height,
        max_width,
        max_height,
        fill_min,
        fill_max,
    ):
        return (
            geometry.area >= min_area
            and geometry.width >= 3
            and geometry.height >= min_height
            and geometry.width <= max_width
            and geometry.height <= max_height
            and fill_min <= geometry.fill_ratio <= fill_max
        )

    def _select_best_component(self, labels, stats, roi_shape, evaluate_component):
        roi_height, roi_width = roi_shape
        best_mask = None
        best_meta = None
        best_score = float("-inf")

        for label in range(1, len(stats)):
            geometry = self._component_geometry(stats[label], roi_height, roi_width)
            score, meta = evaluate_component(label, geometry)
            if score is None:
                continue

            if score > best_score:
                mask = np.zeros(roi_shape, dtype=np.uint8)
                mask[labels == label] = 255
                best_mask = mask
                best_meta = meta
                best_score = score

        return best_mask, best_meta, best_score

    def _extract_digit_mask(self, gray, crop_ratio=None):
        roi = self._crop_inner(gray, crop_ratio=crop_ratio)
        roi_height, roi_width = roi.shape
        min_height = int(roi_height * 0.25)
        max_width = int(roi_width * 0.85)
        max_height = int(roi_height * 0.95)

        blur = cv2.GaussianBlur(roi, (3, 3), 0)
        blackhat = cv2.morphologyEx(blur, cv2.MORPH_BLACKHAT, self._blackhat_kernel)

        _, binary = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, self._open_kernel)

        _, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

        def evaluate_component(label, geometry):
            if not self._passes_component_bounds(
                geometry,
                min_area=self.min_component_area,
                min_height=min_height,
                max_width=max_width,
                max_height=max_height,
                fill_min=0.12,
                fill_max=0.82,
            ):
                return None, None

            component = labels == label
            darkness = 255.0 - float(np.mean(roi[component]))
            response = float(np.mean(blackhat[component]))
            border_penalty = 18.0 if geometry.touches_border and (
                geometry.width <= 5 or geometry.height >= roi_height - 2
            ) else 0.0
            score = response * 6.0 + darkness + geometry.area * 0.05 - border_penalty
            return score, {"darkness": darkness, "response": response}

        best_mask, best_meta, _ = self._select_best_component(labels, stats, roi.shape, evaluate_component)
        if best_mask is None:
            return None

        if best_meta["darkness"] < self.min_component_darkness or best_meta["response"] < self.min_blackhat_response:
            return None

        return self._normalize_mask(best_mask)

    def _extract_threshold_mask(self, gray, crop_ratio):
        roi = self._crop_inner(gray, crop_ratio=crop_ratio)
        roi_height, roi_width = roi.shape
        min_height = max(6, int(roi_height * 0.18))
        max_width = int(roi_width * 0.82)
        max_height = int(roi_height * 0.92)

        normalized = cv2.normalize(roi, None, 0, 255, cv2.NORM_MINMAX)
        _, binary = cv2.threshold(normalized, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        _, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

        def evaluate_component(label, geometry):
            if not self._passes_component_bounds(
                geometry,
                min_area=self.fallback_min_component_area,
                min_height=min_height,
                max_width=max_width,
                max_height=max_height,
                fill_min=0.05,
                fill_max=0.85,
            ):
                return None, None

            component = labels == label
            darkness = 255.0 - float(np.mean(normalized[component]))
            score = darkness + geometry.area * 0.03 - geometry.center_dist * 45.0 - (
                15.0 if geometry.touches_border else 0.0
            )
            return score, {
                "crop_ratio": crop_ratio,
                "darkness": darkness,
                "touches_border": geometry.touches_border,
            }

        best_mask, best_meta, _ = self._select_best_component(labels, stats, roi.shape, evaluate_component)
        if best_mask is None:
            return None, None

        normalized_mask = self._normalize_mask(best_mask)
        if normalized_mask is None:
            return None, None

        return normalized_mask, best_meta

    def _fallback_digit_match(self, gray):
        best_result = None

        for crop_ratio in self.fallback_crop_ratios:
            prepared, meta = self._extract_threshold_mask(gray, crop_ratio)
            if prepared is None:
                continue

            best_digit, best_score, second_score = self._classify_prepared_mask(prepared)
            if not self._is_confident_fallback_match(best_score, second_score, meta):
                continue

            candidate = (best_digit, best_score, meta["touches_border"], -meta["darkness"])
            if best_result is None or candidate[1:] < best_result[1:]:
                best_result = candidate

        if best_result is None:
            return None

        return best_result[0], best_result[1]

    def _count_holes(self, mask):
        contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
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

    def _score(self, candidate_features, template_features):
        overlap = np.logical_and(candidate_features.binary_mask, template_features.binary_mask).sum()
        union = np.logical_or(candidate_features.binary_mask, template_features.binary_mask).sum()
        iou = overlap / union if union else 0.0

        shape_score = 10.0
        if candidate_features.contour is not None and template_features.contour is not None:
            shape_score = cv2.matchShapes(
                candidate_features.contour,
                template_features.contour,
                cv2.CONTOURS_MATCH_I1,
                0.0,
            )

        pixel_score = cv2.norm(candidate_features.small_mask, template_features.small_mask, cv2.NORM_L1)
        pixel_score /= SCORE_SIZE * SCORE_SIZE * 255.0
        return float(shape_score + 0.55 * pixel_score - 0.95 * iou)

    def _render_opencv_digit(self, digit, font_face, font_scale, thickness):
        canvas = np.full((TEMPLATE_RENDER_SIZE, TEMPLATE_RENDER_SIZE), 255, dtype=np.uint8)
        text = str(digit)
        text_size, _ = cv2.getTextSize(text, font_face, font_scale, thickness)
        x = (canvas.shape[1] - text_size[0]) // 2
        y = (canvas.shape[0] + text_size[1]) // 2
        cv2.putText(canvas, text, (x, y), font_face, font_scale, 0, thickness, cv2.LINE_AA)
        return self._extract_digit_mask(canvas, crop_ratio=0.05)

    def _font_paths(self):
        font_dir = Path(r"C:\Windows\Fonts")
        return [font_dir / name for name in PREFERRED_FONTS if (font_dir / name).exists()]

    def _render_pil_digit(self, digit, font_path, font_size):
        if Image is None or ImageDraw is None or ImageFont is None:
            return None

        canvas = Image.new("L", (TEMPLATE_RENDER_SIZE, TEMPLATE_RENDER_SIZE), 255)
        draw = ImageDraw.Draw(canvas)
        font = ImageFont.truetype(str(font_path), size=font_size)
        bbox = draw.textbbox((0, 0), str(digit), font=font)
        x = (TEMPLATE_RENDER_SIZE - (bbox[2] - bbox[0])) // 2 - bbox[0]
        y = (TEMPLATE_RENDER_SIZE - (bbox[3] - bbox[1])) // 2 - bbox[1]
        draw.text((x, y), str(digit), font=font, fill=0)
        return self._extract_digit_mask(np.array(canvas), crop_ratio=0.05)

    def _append_template(self, templates, digit, template):
        if template is not None:
            templates[digit].append(self._prepare_match_features(template))

    def _build_template_features(self):
        templates = {digit: [] for digit in range(1, 10)}

        if Image is not None and ImageDraw is not None and ImageFont is not None:
            for font_path in self._font_paths():
                for digit in range(1, 10):
                    for font_size in PIL_FONT_SIZES:
                        try:
                            template = self._render_pil_digit(digit, font_path, font_size)
                        except OSError:
                            continue

                        self._append_template(templates, digit, template)

        if all(templates.values()):
            return templates

        for digit in range(1, 10):
            if templates[digit]:
                continue

            for font_face in OPENCV_FONTS:
                for font_scale in OPENCV_FONT_SCALES:
                    for thickness in OPENCV_THICKNESSES:
                        template = self._render_opencv_digit(digit, font_face, font_scale, thickness)
                        self._append_template(templates, digit, template)

        return templates

    def recognize_digit(self, image_or_path):
        gray = self._load_gray(image_or_path)
        prepared = self._extract_digit_mask(gray)
        if prepared is not None:
            best_digit, best_score, second_score = self._classify_prepared_mask(prepared)
            if self._is_confident_primary_match(best_score, second_score):
                return best_digit, best_score
        else:
            best_score = 0.0

        fallback = self._fallback_digit_match(gray)
        if fallback is not None:
            return fallback

        if prepared is None:
            return 0, 0.0

        return 0, best_score

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
