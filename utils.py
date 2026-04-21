"""UI rendering utilities for the FaceTrust realtime dashboard."""

from typing import Dict, Optional, Tuple

import cv2
import numpy as np

# UI color palette (BGR)
COLOR_REAL = (150, 206, 124)
COLOR_SPOOF = (94, 108, 240)
COLOR_UNKNOWN = (128, 178, 238)
COLOR_NO_FACE = (150, 150, 150)
COLOR_TEXT = (236, 230, 224)
COLOR_ACCENT = (235, 176, 64)
COLOR_MUTED = (120, 128, 140)
COLOR_CYAN = (186, 222, 80)

PANEL_DARK = (16, 18, 22)
PANEL_BORDER = (88, 94, 102)
PANEL_ACCENT = (235, 176, 64)
PANEL_SOFT = (38, 42, 50)
BG_TOP = (8, 10, 14)
BG_BOTTOM = (6, 7, 10)

MARGIN = 20
FONT_LARGE = 0.90
FONT_MEDIUM = 0.62
FONT_SMALL = 0.54

_GRADIENT_CACHE: Dict[Tuple[int, int, Tuple[int, int, int], Tuple[int, int, int]], np.ndarray] = {}
_CONFIDENCE_DISPLAY_VALUE = 0.0


def _clamp(value: float, minimum: float, maximum: float) -> float:
    """Clamp a scalar value to a closed range.

    Args:
        value: Input value.
        minimum: Lower bound.
        maximum: Upper bound.

    Returns:
        The clamped value.
    """
    return max(minimum, min(maximum, value))


def _confidence_color(confidence: float) -> Tuple[int, int, int]:
    """Return confidence-bar color for low, medium, and high recognition confidence.

    Args:
        confidence: Recognition confidence in [0.0, 1.0].

    Returns:
        BGR color tuple for the confidence bar.
    """
    if confidence >= 0.55:
        return COLOR_REAL
    if confidence >= 0.30:
        return COLOR_ACCENT
    return COLOR_SPOOF


def _draw_vertical_gradient(frame: np.ndarray, top_color: Tuple[int, int, int], bottom_color: Tuple[int, int, int]) -> None:
    """Overlay a cached vertical gradient over the frame.

    Args:
        frame: Destination frame.
        top_color: BGR color at the top edge.
        bottom_color: BGR color at the bottom edge.
    """
    h, w = frame.shape[:2]
    key = (h, w, top_color, bottom_color)
    gradient = _GRADIENT_CACHE.get(key)

    if gradient is None:
        # Cache per-size gradient to avoid rebuilding every frame.
        t = np.linspace(0.0, 1.0, h, dtype=np.float32).reshape(h, 1, 1)
        top = np.array(top_color, dtype=np.float32).reshape(1, 1, 3)
        bottom = np.array(bottom_color, dtype=np.float32).reshape(1, 1, 3)
        row_gradient = ((1.0 - t) * top + t * bottom).astype(np.uint8)
        gradient = np.repeat(row_gradient, w, axis=1)
        _GRADIENT_CACHE[key] = gradient

    cv2.addWeighted(gradient, 0.14, frame, 0.86, 0.0, frame)


def _draw_panel_shadow(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int, offset: int = 8) -> None:
    """Draw a soft rectangular shadow behind a UI panel.

    Args:
        frame: Destination frame.
        x1: Left bound.
        y1: Top bound.
        x2: Right bound.
        y2: Bottom bound.
        offset: Shadow offset in pixels.
    """
    _draw_alpha_rect(frame, x1 + offset, y1 + offset, x2 + offset, y2 + offset, (0, 0, 0), 0.30)


def _draw_rounded_rect(
    frame: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    color: Tuple[int, int, int],
    radius: int = 12,
    thickness: int = -1,
) -> None:
    """Draw a filled or stroked rounded rectangle.

    Args:
        frame: Destination frame.
        x1: Left bound.
        y1: Top bound.
        x2: Right bound.
        y2: Bottom bound.
        color: BGR fill or stroke color.
        radius: Corner radius in pixels.
        thickness: Border thickness, `-1` for filled.
    """
    radius = max(1, radius)
    if thickness < 0:
        cv2.rectangle(frame, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
        cv2.rectangle(frame, (x1, y1 + radius), (x2, y2 - radius), color, thickness)
        cv2.circle(frame, (x1 + radius, y1 + radius), radius, color, thickness)
        cv2.circle(frame, (x2 - radius, y1 + radius), radius, color, thickness)
        cv2.circle(frame, (x1 + radius, y2 - radius), radius, color, thickness)
        cv2.circle(frame, (x2 - radius, y2 - radius), radius, color, thickness)
        return

    # Use a clean rectangular stroke to avoid renderer-specific corner artifacts.
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)


def _draw_corner_accents(frame: np.ndarray) -> None:
    """Draw subtle corner accents for a dashboard-style frame.

    Args:
        frame: Destination frame.
    """
    h, w = frame.shape[:2]
    c = PANEL_BORDER
    l = 20
    t = 2

    cv2.line(frame, (MARGIN, MARGIN), (MARGIN + l, MARGIN), c, t)
    cv2.line(frame, (MARGIN, MARGIN), (MARGIN, MARGIN + l), c, t)

    cv2.line(frame, (w - MARGIN, MARGIN), (w - MARGIN - l, MARGIN), c, t)
    cv2.line(frame, (w - MARGIN, MARGIN), (w - MARGIN, MARGIN + l), c, t)

    cv2.line(frame, (MARGIN, h - MARGIN), (MARGIN + l, h - MARGIN), c, t)
    cv2.line(frame, (MARGIN, h - MARGIN), (MARGIN, h - MARGIN - l), c, t)

    cv2.line(frame, (w - MARGIN, h - MARGIN), (w - MARGIN - l, h - MARGIN), c, t)
    cv2.line(frame, (w - MARGIN, h - MARGIN), (w - MARGIN, h - MARGIN - l), c, t)


def _draw_alpha_rect(
    frame: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    color: Tuple[int, int, int],
    alpha: float,
) -> None:
    """Draws a semi-transparent rectangle."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def _draw_check_mark(frame: np.ndarray, x: int, y: int, size: int, color: Tuple[int, int, int], thickness: int = 2) -> None:
    """Render a check icon using two anti-aliased line segments.

    Args:
        frame: Destination frame.
        x: Left x-coordinate.
        y: Top y-coordinate.
        size: Icon size in pixels.
        color: BGR line color.
        thickness: Line thickness.
    """
    start = (x, y + size // 2)
    mid = (x + size // 3, y + size)
    end = (x + size, y)
    cv2.line(frame, start, mid, color, thickness, cv2.LINE_AA)
    cv2.line(frame, mid, end, color, thickness, cv2.LINE_AA)


def _draw_cross_mark(frame: np.ndarray, x: int, y: int, size: int, color: Tuple[int, int, int], thickness: int = 2) -> None:
    """Render a cross icon using two anti-aliased diagonal lines.

    Args:
        frame: Destination frame.
        x: Left x-coordinate.
        y: Top y-coordinate.
        size: Icon size in pixels.
        color: BGR line color.
        thickness: Line thickness.
    """
    cv2.line(frame, (x, y), (x + size, y + size), color, thickness, cv2.LINE_AA)
    cv2.line(frame, (x + size, y), (x, y + size), color, thickness, cv2.LINE_AA)


def _draw_keycap(frame: np.ndarray, x: int, y: int, key: str, label: str) -> None:
    """Draw a keyboard-style control hint with a keycap and action label."""
    cap_w, cap_h = 30, 26
    _draw_rounded_rect(frame, x, y - cap_h + 4, x + cap_w, y + 4, (20, 24, 30), radius=6, thickness=-1)
    cv2.rectangle(frame, (x, y - cap_h + 4), (x + cap_w, y + 4), (40, 48, 60), 1)
    cv2.putText(frame, key, (x + 10, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (202, 210, 220), 1, cv2.LINE_AA)
    cv2.putText(frame, label, (x + cap_w + 8, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (138, 146, 158), 1, cv2.LINE_AA)


def draw_status_bar(frame: np.ndarray, status_text: str, status_color: Tuple[int, int, int]) -> None:
    """Render the top status bar.

    Args:
        frame: Destination frame.
        status_text: Center status label text.
        status_color: BGR color for status text.
    """
    h, w = frame.shape[:2]
    _draw_vertical_gradient(frame, BG_TOP, BG_BOTTOM)
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (38, 96, 190), 1)

    bar_h = 54
    x1, y1, x2, y2 = 0, 0, w - 1, bar_h
    cv2.rectangle(frame, (x1, y1), (x2, y2), (10, 13, 18), -1)
    cv2.line(frame, (0, bar_h), (w - 1, bar_h), (30, 38, 50), 1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, "Face", (58, 61), font, 0.80, (222, 228, 236), 1, cv2.LINE_AA)
    cv2.putText(frame, "Trust", (122, 61), font, 0.80, (74, 170, 230), 1, cv2.LINE_AA)
    cv2.line(frame, (330, 45), (330, 86), (36, 42, 52), 1)
    cv2.putText(frame, "LIVE", (356, 57), font, 0.56, (124, 132, 144), 1, cv2.LINE_AA)
    cv2.putText(frame, "SESSION", (356, 79), font, 0.56, (92, 100, 110), 1, cv2.LINE_AA)

    chip_w, chip_h = 370, 38
    chip_x1 = (w - chip_w) // 2
    chip_y1 = 9
    chip_x2 = chip_x1 + chip_w
    chip_y2 = chip_y1 + chip_h
    _draw_rounded_rect(frame, chip_x1, chip_y1, chip_x2, chip_y2, (14, 70, 46), radius=18, thickness=-1)
    cv2.rectangle(frame, (chip_x1, chip_y1), (chip_x2, chip_y2), (28, 126, 88), 1)
    (tw, _), _ = cv2.getTextSize(status_text, font, 0.76, 1)
    cv2.putText(frame, status_text, (chip_x1 + (chip_w - tw) // 2, 35), font, 0.82, status_color, 1, cv2.LINE_AA)

    dot_x = w - 86
    cv2.circle(frame, (dot_x, 24), 4, COLOR_ACCENT, -1)
    cv2.circle(frame, (dot_x + 18, 24), 4, (70, 78, 90), -1)
    cv2.circle(frame, (dot_x + 36, 24), 4, (48, 56, 66), -1)


def draw_face_box(frame: np.ndarray, det: Dict, label: str, color: Tuple[int, int, int]) -> None:
    """Render face bounding box and identity tag.

    Args:
        frame: Destination frame.
        det: Detection payload containing `bbox`.
        label: Identity label text.
        color: BGR color for the box border.
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = det["bbox"]

    pad = 8
    x1 = max(pad, x1)
    y1 = max(pad, y1)
    x2 = min(w - pad, x2)
    y2 = min(h - pad, y2)

    cv2.rectangle(frame, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), (16, 16, 18), 2)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.34
    thick = 1
    (tw, th), baseline = cv2.getTextSize(label, font, scale, thick)
    lx = x1
    ly = y1 - 10
    if ly - th < 6:
        ly = y1 + th + 10

    bx1, by1 = lx - 8, ly - th - 8
    bx2, by2 = lx + tw + 8, ly + baseline + 8
    _draw_rounded_rect(frame, bx1, by1, bx2, by2, (16, 18, 22), radius=8, thickness=-1)
    cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, 1)
    cv2.putText(frame, label, (lx, ly), font, scale, COLOR_TEXT, thick, cv2.LINE_AA)


def draw_info_panel(
    frame: np.ndarray,
    face_count: int,
    fps: float,
    confidence: float = 0.0,
    identity_name: str = "Unknown",
    blink_count: int = 0,
    ear_value: float = 0.0,
    ear_threshold: float = 0.22,
    session_events: Optional[list[Tuple[str, str]]] = None,
) -> None:
    """Draw top-right telemetry card with face count, FPS, and confidence meter.

    Args:
        frame: Destination frame.
        face_count: Number of currently detected faces.
        fps: Current smoothed FPS value.
        confidence: Recognition confidence in [0.0, 1.0].
    """
    global _CONFIDENCE_DISPLAY_VALUE

    h, w = frame.shape[:2]
    panel_w = min(615, max(540, int(w * 0.45)))
    x1 = w - panel_w - 58
    card_gap = 12
    card_y = 170

    identity_h = 118
    telemetry_h = 150
    ear_h = 96
    log_h = 122

    def draw_card(y1: int, h: int, title: str) -> Tuple[int, int, int, int]:
        x2_local = x1 + panel_w
        y2_local = y1 + h
        _draw_panel_shadow(frame, x1, y1, x2_local, y2_local, offset=3)
        _draw_rounded_rect(frame, x1, y1, x2_local, y2_local, (17, 12, 10), radius=12, thickness=-1)
        _draw_rounded_rect(frame, x1, y1, x2_local, y2_local, (56, 44, 36), radius=12, thickness=1)
        cv2.putText(frame, title, (x1 + 22, y1 + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (104, 92, 84), 1, cv2.LINE_AA)
        return x1, y1, x2_local, y2_local

    id_x1, id_y1, id_x2, _ = draw_card(card_y, identity_h, "IDENTITY")
    tele_y = card_y + identity_h + card_gap
    tele_x1, tele_y1, tele_x2, _ = draw_card(tele_y, telemetry_h, "TELEMETRY")
    ear_y = tele_y + telemetry_h + card_gap
    ear_x1, ear_y1, ear_x2, _ = draw_card(ear_y, ear_h, "EYE ASPECT RATIO")
    log_y = ear_y + ear_h + card_gap
    log_x1, log_y1, log_x2, _ = draw_card(log_y, log_h, "SESSION LOG")

    confidence = _clamp(float(confidence), 0.0, 1.0)
    _CONFIDENCE_DISPLAY_VALUE = 0.72 * _CONFIDENCE_DISPLAY_VALUE + 0.28 * confidence

    font = cv2.FONT_HERSHEY_SIMPLEX
    label_color = (166, 174, 186)

    # Identity card
    avatar_center = (id_x1 + 46, id_y1 + 70)
    cv2.circle(frame, avatar_center, 26, (16, 68, 50), -1)
    cv2.circle(frame, avatar_center, 26, (34, 122, 92), 1)
    initials = (identity_name[:2].upper() if identity_name and identity_name != "Unknown" else "--")
    cv2.putText(frame, initials, (avatar_center[0] - 13, avatar_center[1] + 7), font, 0.74, (110, 226, 170), 1, cv2.LINE_AA)
    cv2.putText(frame, identity_name, (id_x1 + 84, id_y1 + 60), font, 0.84, (222, 226, 232), 1, cv2.LINE_AA)
    cv2.putText(frame, "Verified - Live", (id_x1 + 84, id_y1 + 89), font, 0.52, (118, 126, 138), 1, cv2.LINE_AA)

    # Telemetry card
    rows = [("Faces", f"{face_count}", COLOR_TEXT), ("FPS", f"{fps:.0f}", COLOR_ACCENT), ("Blinks", f"{blink_count}", COLOR_TEXT)]
    row_y = tele_y1 + 56
    for label, value, val_color in rows:
        cv2.putText(frame, label, (tele_x1 + 22, row_y), font, 0.72, label_color, 1, cv2.LINE_AA)
        (vw, _), _ = cv2.getTextSize(value, font, 0.82, 1)
        cv2.putText(frame, value, (tele_x2 - 18 - vw, row_y), font, 0.76, val_color, 1, cv2.LINE_AA)
        cv2.line(frame, (tele_x1 + 22, row_y + 11), (tele_x2 - 22, row_y + 11), (28, 36, 48), 1)
        row_y += 29

    cv2.putText(frame, "Confidence", (tele_x1 + 22, tele_y1 + 136), font, 0.70, label_color, 1, cv2.LINE_AA)
    conf_pct_text = f"{int(round(_CONFIDENCE_DISPLAY_VALUE * 100.0))}%"
    (conf_w, _), _ = cv2.getTextSize(conf_pct_text, font, 0.76, 1)
    cv2.putText(frame, conf_pct_text, (tele_x2 - 18 - conf_w, tele_y1 + 136), font, 0.76, COLOR_REAL, 1, cv2.LINE_AA)

    bar_x1 = tele_x1 + 22
    bar_y1 = tele_y1 + 152
    bar_x2 = tele_x2 - 22
    bar_y2 = bar_y1 + 12
    _draw_rounded_rect(frame, bar_x1, bar_y1, bar_x2, bar_y2, (34, 38, 44), radius=5, thickness=-1)
    cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x2, bar_y2), (58, 64, 72), 1)

    fill_w = int((bar_x2 - bar_x1) * _CONFIDENCE_DISPLAY_VALUE)
    if fill_w > 0:
        fill_color = _confidence_color(_CONFIDENCE_DISPLAY_VALUE)
        _draw_rounded_rect(frame, bar_x1, bar_y1, bar_x1 + fill_w, bar_y2, fill_color, radius=5, thickness=-1)

    # EAR card
    cv2.putText(frame, f"{ear_value:.2f}", (ear_x2 - 52, ear_y1 + 56), font, 0.68, (110, 178, 250), 1, cv2.LINE_AA)
    ear_bar_x1 = ear_x1 + 22
    ear_bar_x2 = ear_x2 - 70
    ear_bar_y1 = ear_y1 + 46
    ear_bar_y2 = ear_bar_y1 + 10
    _draw_rounded_rect(frame, ear_bar_x1, ear_bar_y1, ear_bar_x2, ear_bar_y2, (24, 30, 38), radius=5, thickness=-1)
    normalized_ear = _clamp((ear_value - 0.12) / 0.24, 0.0, 1.0)
    ear_fill = int((ear_bar_x2 - ear_bar_x1) * normalized_ear)
    if ear_fill > 0:
        _draw_rounded_rect(frame, ear_bar_x1, ear_bar_y1, ear_bar_x1 + ear_fill, ear_bar_y2, (180, 118, 74), radius=5, thickness=-1)
    cv2.putText(frame, f"Threshold  {ear_threshold:.2f}", (ear_x1 + 22, ear_y1 + 78), font, 0.52, (92, 100, 112), 1, cv2.LINE_AA)

    # Session log card
    events = session_events if session_events is not None else []
    y = log_y1 + 54
    for status_text, time_text in events[-4:][::-1]:
        if status_text == "AUTHENTICATED":
            dot_color = COLOR_REAL
            label = "Authenticated"
        elif status_text == "SPOOF_DETECTED":
            dot_color = COLOR_SPOOF
            label = "Spoof attempt"
        else:
            dot_color = COLOR_ACCENT
            label = "Unknown face"

        cv2.circle(frame, (log_x1 + 26, y - 5), 4, dot_color, -1)
        cv2.putText(frame, label, (log_x1 + 40, y), font, 0.62, (146, 154, 168), 1, cv2.LINE_AA)
        (tw, _), _ = cv2.getTextSize(time_text, font, 0.62, 1)
        cv2.putText(frame, time_text, (log_x2 - 18 - tw, y), font, 0.62, (98, 106, 118), 1, cv2.LINE_AA)
        y += 26


def draw_liveness_row(
    frame: np.ndarray,
    anchor_y: int,
    blink_detected: bool,
    movement_detected: bool,
    pose_detected: bool = False,
) -> None:
    """Draw compact liveness status labels under the camera feed.

    Args:
        frame: Destination frame.
        anchor_y: Vertical anchor aligned to the camera panel bottom.
        blink_detected: Whether blink cue is recently valid.
        movement_detected: Whether motion cue is recently valid.
        pose_detected: Whether pose-variation cue is recently valid.
    """
    h, w = frame.shape[:2]
    y1 = min(anchor_y + 12, h - 92)
    y2 = y1 + 38
    left = 58
    right = w - 58

    _draw_rounded_rect(frame, left, y1, right, y2, (12, 16, 22), radius=12, thickness=-1)
    _draw_rounded_rect(frame, left, y1, right, y2, (30, 38, 50), radius=12, thickness=1)

    capsule_gap = 10
    capsule_w = max(84, int((right - left - capsule_gap * 4) / 3))
    cx = left + capsule_gap
    status_items = [("Blink", blink_detected), ("Movement", movement_detected), ("Pose", pose_detected)]
    for label, state in status_items:
        cy1 = y1 + 6
        cy2 = y2 - 6
        fill = (12, 76, 48) if state else (58, 22, 30)
        border = (24, 128, 84) if state else (132, 52, 70)
        text_color = (82, 230, 160) if state else (236, 136, 150)

        _draw_rounded_rect(frame, cx, cy1, cx + capsule_w, cy2, fill, radius=14, thickness=-1)
        cv2.rectangle(frame, (cx, cy1), (cx + capsule_w, cy2), border, 1)
        cv2.circle(frame, (cx + 20, (cy1 + cy2) // 2), 5, text_color, -1)
        cv2.putText(frame, label, (cx + 36, (cy1 + cy2) // 2 + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.66, text_color, 1, cv2.LINE_AA)
        cx += capsule_w + capsule_gap


def draw_bottom_bar(frame: np.ndarray) -> None:
    """Render the bottom control hint bar.

    Args:
        frame: Destination frame.
    """
    h, w = frame.shape[:2]
    y = h - 16
    cv2.line(frame, (58, h - 42), (w - 58, h - 42), (24, 30, 40), 1)

    items = [("N", "Register"), ("C", "Capture"), ("R", "Rebuild"), ("F", "Fullscreen"), ("Q", "Quit")]
    spacing = min(146, (w - 140) // max(1, len(items)))
    start_x = 58
    for idx, (key, label) in enumerate(items):
        x = start_x + idx * spacing
        _draw_keycap(frame, x, y, key, label)


def draw_name_entry_prompt(
    frame: np.ndarray,
    typed_name: str,
    last_captured_face: Optional[np.ndarray] = None,
    blink_detected: bool = False,
    movement_detected: bool = False,
) -> None:
    """Render the name-entry modal for user registration.

    Args:
        frame: Destination frame.
        typed_name: Current text entered by the user.
        last_captured_face: Optional latest captured face crop preview.
        blink_detected: Blink cue status for the modal footer.
        movement_detected: Motion cue status for the modal footer.
    """
    h, w = frame.shape[:2]
    pad = 20

    box_w = min(860, w - (MARGIN * 2))
    box_h = min(360, h - 164)
    x1 = max(MARGIN, (w - box_w) // 2)
    y1 = max(88, (h - box_h) // 2 + 4)
    x2, y2 = x1 + box_w, y1 + box_h

    # Dim outside area before drawing the modal shell.
    _draw_alpha_rect(frame, 0, 0, w, h, (4, 6, 8), 0.34)
    _draw_panel_shadow(frame, x1, y1, x2, y2, offset=10)
    _draw_rounded_rect(frame, x1, y1, x2, y2, PANEL_DARK, radius=14, thickness=-1)
    _draw_rounded_rect(frame, x1, y1, x2, y2, (40, 44, 50), radius=14, thickness=1)
    cv2.line(frame, (x1, y1 + 58), (x2, y1 + 58), PANEL_SOFT, 1)

    title = "REGISTER NEW IDENTITY"
    (tw, th), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 1.12, 2)
    cv2.putText(
        frame,
        title,
        (x1 + (box_w - tw) // 2, y1 + 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.12,
        (206, 176, 126),
        2,
        cv2.LINE_AA,
    )

    inner_x1 = x1 + pad
    inner_y1 = y1 + 74
    inner_x2 = x2 - pad
    inner_y2 = y2 - pad
    inner_w = inner_x2 - inner_x1

    left_w = int(inner_w * 0.56)
    right_w = inner_w - left_w

    left_x1 = inner_x1
    left_x2 = left_x1 + left_w
    right_x1 = left_x2
    right_x2 = inner_x2

    cv2.line(frame, (left_x2 + 2, inner_y1), (left_x2 + 2, inner_y2), PANEL_SOFT, 1)

    # Text block (left side).
    text_x = left_x1
    cv2.putText(frame, "Enter Name", (text_x + 2, inner_y1 + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.84, (198, 188, 174), 1, cv2.LINE_AA)

    input_x1 = text_x
    input_y1 = inner_y1 + 52
    input_x2 = left_x2 - 16
    input_y2 = input_y1 + 56
    _draw_alpha_rect(frame, input_x1, input_y1, input_x2, input_y2, (24, 26, 30), 0.80)
    cv2.rectangle(frame, (input_x1, input_y1), (input_x2, input_y2), (88, 80, 66), 1)
    cv2.putText(frame, f"{typed_name}_", (input_x1 + 12, input_y1 + 38), cv2.FONT_HERSHEY_SIMPLEX, 1.00, PANEL_ACCENT, 2, cv2.LINE_AA)

    help_y = input_y2 + 44
    cv2.putText(frame, "ENTER -> Save", (text_x + 4, help_y), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (198, 188, 174), 1, cv2.LINE_AA)
    cv2.putText(frame, "ESC -> Cancel", (text_x + 190, help_y), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (198, 188, 174), 1, cv2.LINE_AA)

    liveness_y = inner_y2 - 16
    blink_text = "Blink" if blink_detected else "Blink"
    move_text = "Movement" if movement_detected else "Movement"
    blink_color = COLOR_REAL if blink_detected else (176, 160, 142)
    move_color = COLOR_REAL if movement_detected else (176, 160, 142)
    blink_status = "OK" if blink_detected else "WAIT"
    move_status = "OK" if movement_detected else "WAIT"
    cv2.putText(frame, blink_text, (text_x + 18, liveness_y), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (210, 196, 178), 1, cv2.LINE_AA)
    cv2.putText(frame, blink_status, (text_x + 86, liveness_y), cv2.FONT_HERSHEY_SIMPLEX, 0.72, blink_color, 1, cv2.LINE_AA)
    cv2.putText(frame, move_text, (text_x + 172, liveness_y), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (210, 196, 178), 1, cv2.LINE_AA)
    cv2.putText(frame, move_status, (text_x + 286, liveness_y), cv2.FONT_HERSHEY_SIMPLEX, 0.72, move_color, 1, cv2.LINE_AA)

    # Preview block (right side).
    preview_size = 220
    preview_x1 = right_x1 + max(0, (right_w - preview_size) // 2)
    preview_y1 = inner_y1 + max(0, ((inner_y2 - inner_y1) - preview_size) // 2) + 6
    preview_x2 = preview_x1 + preview_size
    preview_y2 = preview_y1 + preview_size

    # Keep preview strictly inside modal inner area.
    if preview_x2 > inner_x2:
        shift = preview_x2 - inner_x2
        preview_x1 -= shift
        preview_x2 -= shift
    if preview_y2 > inner_y2:
        shift = preview_y2 - inner_y2
        preview_y1 -= shift
        preview_y2 -= shift

    cv2.putText(frame, "Preview", (preview_x1 + 52, preview_y1 - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (198, 188, 174), 1, cv2.LINE_AA)
    _draw_alpha_rect(frame, preview_x1, preview_y1, preview_x2, preview_y2, (12, 14, 16), 0.88)
    cv2.rectangle(frame, (preview_x1, preview_y1), (preview_x2, preview_y2), PANEL_ACCENT, 2)

    if last_captured_face is None or last_captured_face.size == 0:
        (tw, th), _ = cv2.getTextSize("No Image", cv2.FONT_HERSHEY_SIMPLEX, 0.76, 2)
        tx = preview_x1 + (preview_size - tw) // 2
        ty = preview_y1 + (preview_size + th) // 2
        cv2.putText(frame, "No Image", (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.76, COLOR_TEXT, 1, cv2.LINE_AA)
    else:
        crop_h, crop_w = last_captured_face.shape[:2]
        scale = min(preview_size / max(1, crop_w), preview_size / max(1, crop_h))
        resized_w = max(1, int(crop_w * scale))
        resized_h = max(1, int(crop_h * scale))
        resized = cv2.resize(last_captured_face, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)

        canvas = np.zeros((preview_size, preview_size, 3), dtype=np.uint8)
        off_x = (preview_size - resized_w) // 2
        off_y = (preview_size - resized_h) // 2
        canvas[off_y:off_y + resized_h, off_x:off_x + resized_w] = resized
        frame[preview_y1:preview_y2, preview_x1:preview_x2] = canvas
        cv2.rectangle(frame, (preview_x1, preview_y1), (preview_x2, preview_y2), PANEL_ACCENT, 2)
