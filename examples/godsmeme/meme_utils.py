import os
import re
import random
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont

_REQUIRED_SECTION_PATTERNS = [
    r"\[Comprehensive Description Section\]",
    r"\[Usage Scenarios Section\]",
    r"\[Text Analysis Section\]",
    r"\[Specific Analysis with User Input\]",
    r"Text on the Meme:",
]


@dataclass
class MemeRenderConfig:
    font_name: str = "DejaVuSans.ttf"
    min_font_size: int = 14
    max_font_size: int = 72
    line_spacing: int = 4
    outline_width: int = 2
    margin: int = 6
    default_padding_ratio: float = 0.06


@dataclass
class PairwisePreference:
    index_a: int
    index_b: int
    score_a: float
    score_b: float


def load_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read().strip()


def extract_assistant_response(text: str) -> str:
    if "<|im_start|>assistant" in text:
        return text.split("<|im_start|>assistant")[-1].strip()
    if "assistant\n" in text:
        return text.split("assistant\n")[-1].strip()
    return text.strip()


def extract_text_on_meme_section(response: str) -> str:
    response = extract_assistant_response(response)
    match = re.search(r"Text on the Meme:\s*(.*)$", response, re.IGNORECASE | re.DOTALL)
    if not match:
        return ""
    return match.group(1).strip()


def extract_box_texts(response: str) -> List[str]:
    section = extract_text_on_meme_section(response)
    if not section:
        return []

    matches = list(re.finditer(
        r"(?im)^\s*box\s*(\d+)\s*:\s*(.*?)(?=^\s*box\s*\d+\s*:|\Z)",
        section,
        re.DOTALL,
    ))
    if matches:
        items = []
        for match in matches:
            payload = match.group(2).strip().strip('"').strip()
            if payload:
                items.append(payload)
        if items:
            return items

    lines = []
    for raw_line in section.splitlines():
        line = raw_line.strip().strip('"').strip()
        if line:
            lines.append(line)
    return lines


def compute_meme_format_reward(response: str, expected_boxes: Optional[int] = None) -> float:
    response = extract_assistant_response(response)
    checks = 0
    passed = 0

    for pattern in _REQUIRED_SECTION_PATTERNS:
        checks += 1
        if re.search(pattern, response, re.IGNORECASE):
            passed += 1

    checks += 2
    if re.search(r"(?im)^\s*Step\s*1\s*:", response):
        passed += 1
    if re.search(r"(?im)^\s*Step\s*2\s*:", response):
        passed += 1

    fragments = extract_box_texts(response)
    checks += 2
    if fragments:
        passed += 1
    if fragments and all(fragment.strip() for fragment in fragments):
        passed += 1

    if expected_boxes is not None and expected_boxes > 0:
        checks += 1
        if len(fragments) == expected_boxes:
            passed += 1

    return float(passed) / float(checks) if checks else 0.0


def normalize_bbox(bbox: Any) -> Optional[Tuple[float, float, float, float]]:
    if bbox is None:
        return None
    if isinstance(bbox, dict):
        for key in ("bbox", "box", "loc", "coordinates"):
            if key in bbox:
                bbox = bbox[key]
                break
    if isinstance(bbox, tuple):
        bbox = list(bbox)
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None

    values: List[float] = []
    for item in bbox:
        try:
            values.append(float(item))
        except (TypeError, ValueError):
            return None
    return tuple(values)


def normalize_detections(reference: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not isinstance(reference, dict):
        return []

    raw_detections = reference.get("detections")
    if raw_detections is None and isinstance(reference.get("text_loc_info"), dict):
        text_loc_info = reference["text_loc_info"]
        locs = text_loc_info.get("loc", [])
        texts = text_loc_info.get("text", [])
        if isinstance(texts, str):
            texts = extract_box_texts(f"Text on the Meme:\n{texts}")
        raw_detections = [{"bbox": loc, "text": texts[idx] if idx < len(texts) else ""} for idx, loc in enumerate(locs)]
    if raw_detections is None and isinstance(reference.get("loc"), list):
        raw_detections = [{"bbox": loc} for loc in reference.get("loc", [])]

    detections: List[Dict[str, Any]] = []
    for item in raw_detections or []:
        bbox = normalize_bbox(item)
        if bbox is None:
            continue
        payload = item if isinstance(item, dict) else {"bbox": item}
        detections.append({"bbox": bbox, **payload})
    return detections


def resolve_expected_box_count(reference: Optional[Dict[str, Any]]) -> Optional[int]:
    detections = normalize_detections(reference)
    if detections:
        return len(detections)
    if isinstance(reference, dict) and isinstance(reference.get("expected_box_count"), int):
        return reference["expected_box_count"]
    return None


def get_user_request(reference: Optional[Dict[str, Any]], fallback_prompt: Optional[str] = None) -> str:
    if isinstance(reference, dict):
        for key in ("user_input_text", "user_request", "input_params", "prompt_summary"):
            value = reference.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return (fallback_prompt or "").strip()


def get_reference_output(reference: Optional[Dict[str, Any]]) -> str:
    if not isinstance(reference, dict):
        return ""
    for key in ("reference_output", "reference", "answer", "label_text"):
        value = reference.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def get_reference_group_id(reference: Optional[Dict[str, Any]], fallback_index: int) -> str:
    if isinstance(reference, dict):
        for key in ("group_id", "sample_id", "id"):
            value = reference.get(key)
            if value is not None:
                return str(value)
    return str(fallback_index)


def get_first_image(raw_image: Any) -> Optional[Image.Image]:
    if raw_image is None:
        return None
    if isinstance(raw_image, Image.Image):
        return raw_image
    if isinstance(raw_image, list) and raw_image:
        return get_first_image(raw_image[0])
    return None


def _resolve_font_path(font_name: str) -> Optional[str]:
    search_roots = []
    env_root = os.getenv("MEMEGENERATOR_FONT_DIR")
    if env_root:
        search_roots.append(os.path.expanduser(env_root))
    search_roots.extend([
        "/usr/share/fonts/truetype/dejavu",
        "/usr/share/fonts/truetype/liberation2",
        "/System/Library/Fonts",
        "/Library/Fonts",
    ])
    for root in search_roots:
        candidate = os.path.join(root, font_name)
        if os.path.exists(candidate):
            return candidate
    return font_name


def _load_font(font_name: str, font_size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(_resolve_font_path(font_name), font_size)
    except OSError:
        return ImageFont.load_default()


def _measure_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> Tuple[int, int]:
    if hasattr(draw, "textbbox"):
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        return right - left, bottom - top
    return draw.textsize(text, font=font)


def _wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> List[str]:
    words = text.split()
    if not words:
        return [""]

    lines: List[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}".strip()
        if _measure_text(draw, candidate, font)[0] <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def _fit_font(draw: ImageDraw.ImageDraw, text: str, box: Tuple[int, int, int, int],
              config: MemeRenderConfig) -> Tuple[ImageFont.ImageFont, List[str]]:
    x1, y1, x2, y2 = box
    max_width = max(1, x2 - x1 - (2 * config.margin))
    max_height = max(1, y2 - y1 - (2 * config.margin))

    best_font: ImageFont.ImageFont = _load_font(config.font_name, config.min_font_size)
    best_lines = _wrap_text(draw, text, best_font, max_width)

    for size in range(config.min_font_size, config.max_font_size + 1):
        font = _load_font(config.font_name, size)
        lines = _wrap_text(draw, text, font, max_width)
        line_heights = [_measure_text(draw, line, font)[1] for line in lines]
        total_height = sum(line_heights) + config.line_spacing * max(0, len(lines) - 1)
        line_width = max(_measure_text(draw, line, font)[0] for line in lines)
        if line_width <= max_width and total_height <= max_height:
            best_font = font
            best_lines = lines
        else:
            break

    return best_font, best_lines


def _choose_colors(image: Image.Image, box: Tuple[int, int, int,
                                                  int]) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    crop = image.crop(box)
    mean_pixel = crop.resize((1, 1)).getpixel((0, 0)) if crop.size[0] > 0 and crop.size[1] > 0 else (255, 255, 255)
    if isinstance(mean_pixel, int):
        mean_pixel = (mean_pixel, mean_pixel, mean_pixel)
    luminance = (0.299 * mean_pixel[0]) + (0.587 * mean_pixel[1]) + (0.114 * mean_pixel[2])
    if luminance > 186:
        return (0, 0, 0), (255, 255, 255)
    return (255, 255, 255), (0, 0, 0)


def _scale_box(
    bbox: Tuple[float, float, float, float],
    image: Image.Image,
    reference: Optional[Dict[str, Any]] = None,
) -> Tuple[int, int, int, int]:
    width, height = image.size
    bbox_scale = None
    normalized = False
    if isinstance(reference, dict):
        bbox_scale = reference.get("bbox_scale")
        normalized = bool(reference.get("bbox_normalized", False))

    x1, y1, x2, y2 = bbox
    values = [x1, y1, x2, y2]
    if bbox_scale is not None:
        scale = float(bbox_scale)
        x1, x2 = (x1 / scale) * width, (x2 / scale) * width
        y1, y2 = (y1 / scale) * height, (y2 / scale) * height
    elif normalized or max(values) <= 1.0:
        x1, x2 = x1 * width, x2 * width
        y1, y2 = y1 * height, y2 * height

    left = max(0, min(int(round(x1)), width - 1))
    top = max(0, min(int(round(y1)), height - 1))
    right = max(left + 1, min(int(round(x2)), width))
    bottom = max(top + 1, min(int(round(y2)), height))
    return left, top, right, bottom


def default_render_boxes(image: Image.Image, count: int, config: MemeRenderConfig) -> List[Tuple[int, int, int, int]]:
    width, height = image.size
    pad_x = int(width * config.default_padding_ratio)
    pad_y = int(height * config.default_padding_ratio)
    if count <= 1:
        return [(pad_x, pad_y, width - pad_x, max(pad_y + 1, int(height * 0.2)))]
    top_box = (pad_x, pad_y, width - pad_x, max(pad_y + 1, int(height * 0.2)))
    bottom_box = (pad_x, max(int(height * 0.78), pad_y), width - pad_x, height - pad_y)
    boxes = [top_box, bottom_box]
    if count > 2:
        middle_height = max(1, int((height * 0.58) / max(1, count - 2)))
        start_y = int(height * 0.24)
        for idx in range(count - 2):
            y1 = start_y + (idx * middle_height)
            y2 = min(height - pad_y, y1 + middle_height)
            boxes.insert(-1, (pad_x, y1, width - pad_x, max(y1 + 1, y2)))
    return boxes[:count]


def render_meme_image(
    image: Image.Image,
    texts: Sequence[str],
    detections: Optional[Sequence[Dict[str, Any]]] = None,
    reference: Optional[Dict[str, Any]] = None,
    config: Optional[MemeRenderConfig] = None,
) -> Image.Image:
    render_config = config or MemeRenderConfig()
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)

    cleaned_texts = [text.strip() for text in texts if text and text.strip()]
    if not cleaned_texts:
        return canvas

    boxes: List[Tuple[int, int, int, int]] = []
    for detection in detections or []:
        bbox = normalize_bbox(detection)
        if bbox is None:
            continue
        boxes.append(_scale_box(bbox, canvas, reference=reference))

    if not boxes:
        boxes = default_render_boxes(canvas, len(cleaned_texts), render_config)

    if len(cleaned_texts) > len(boxes):
        merged = list(cleaned_texts[:len(boxes)])
        merged[-1] = "\n".join([merged[-1], *cleaned_texts[len(boxes):]]).strip()
        cleaned_texts = merged
    else:
        cleaned_texts = cleaned_texts[:len(boxes)]

    for box, text in zip(boxes, cleaned_texts):
        font, wrapped_lines = _fit_font(draw, text, box, render_config)
        fill_color, outline_color = _choose_colors(canvas, box)
        line_sizes = [_measure_text(draw, line, font) for line in wrapped_lines]
        total_height = sum(height
                           for _, height in line_sizes) + render_config.line_spacing * max(0,
                                                                                           len(wrapped_lines) - 1)
        x1, y1, x2, y2 = box
        current_y = y1 + max(render_config.margin, (y2 - y1 - total_height) // 2)

        for line, (line_width, line_height) in zip(wrapped_lines, line_sizes):
            current_x = x1 + max(render_config.margin, (x2 - x1 - line_width) // 2)
            for dx in range(-render_config.outline_width, render_config.outline_width + 1):
                for dy in range(-render_config.outline_width, render_config.outline_width + 1):
                    if dx == 0 and dy == 0:
                        continue
                    draw.text((current_x + dx, current_y + dy), line, font=font, fill=outline_color)
            draw.text((current_x, current_y), line, font=font, fill=fill_color)
            current_y += line_height + render_config.line_spacing

    return canvas


def sample_group_pairs(group_size: int, max_pairs: int = 0, seed: Optional[int] = None) -> List[Tuple[int, int]]:
    pairs = [(left, right) for left in range(group_size) for right in range(left + 1, group_size)]
    if max_pairs <= 0 or len(pairs) <= max_pairs:
        return pairs
    rng = random.Random(seed)
    rng.shuffle(pairs)
    return pairs[:max_pairs]


def aggregate_pairwise_preferences(
    batch_size: int,
    preferences: Iterable[PairwisePreference],
) -> List[float]:
    totals = [0.0 for _ in range(batch_size)]
    counts = [0 for _ in range(batch_size)]

    for pref in preferences:
        denom = pref.score_a + pref.score_b
        if denom <= 0:
            norm_a = 0.5
            norm_b = 0.5
        else:
            norm_a = pref.score_a / denom
            norm_b = pref.score_b / denom
        totals[pref.index_a] += norm_a
        totals[pref.index_b] += norm_b
        counts[pref.index_a] += 1
        counts[pref.index_b] += 1

    rewards = []
    for total, count in zip(totals, counts):
        rewards.append(total / count if count > 0 else 0.0)
    return rewards
