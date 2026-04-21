"""Realtime application entrypoint for FaceTrust face authentication and liveness."""

import os
import cv2
import time
import warnings
import numpy as np
from typing import Optional
from datetime import datetime

# Reduce TensorFlow/MediaPipe C++ log noise for cleaner demo output.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GLOG_minloglevel"] = "2"

# Silence non-critical third-party warnings that clutter the demo terminal.
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from anti_spoofing import AntiSpoofing
from face_recognition import FaceRecognizer
from session_logger import SessionLogger

from utils import (
    COLOR_NO_FACE,
    COLOR_REAL,
    COLOR_SPOOF,
    COLOR_UNKNOWN,
    _draw_panel_shadow,
    _draw_rounded_rect,
    draw_face_box,
    draw_bottom_bar,
    draw_info_panel,
    draw_liveness_row,
    draw_name_entry_prompt,
    draw_status_bar,
)


def _show_startup_splash(window_name: str, subtitle: str, phase: int = 0) -> None:
    """Render startup splash while subsystems initialize.

    Args:
        window_name: OpenCV window title.
        subtitle: Current startup phase text.
        phase: Startup phase index used for progress animation.
    """
    splash_h, splash_w = 720, 1280
    frame = np.zeros((splash_h, splash_w, 3), dtype=np.uint8)

    top_color = np.array([44, 66, 92], dtype=np.float32)
    bottom_color = np.array([16, 26, 38], dtype=np.float32)
    for y in range(splash_h):
        t = y / max(1, splash_h - 1)
        frame[y, :, :] = ((1.0 - t) * top_color + t * bottom_color).astype(np.uint8)

    # Soft spotlight panel.
    cv2.rectangle(frame, (110, 140), (splash_w - 110, 520), (26, 32, 40), -1)
    cv2.rectangle(frame, (110, 140), (splash_w - 110, 520), (96, 116, 138), 1)

    bar_y = int(splash_h * 0.70)
    cv2.rectangle(frame, (130, bar_y), (splash_w - 130, bar_y + 46), (32, 36, 42), -1)
    fill_ratio = max(0.1, min(1.0, 0.25 + 0.25 * phase))
    fill_w = int((splash_w - 260) * fill_ratio)
    cv2.rectangle(frame, (130, bar_y), (130 + fill_w, bar_y + 46), (78, 186, 242), -1)
    cv2.rectangle(frame, (130, bar_y), (splash_w - 130, bar_y + 46), (190, 206, 224), 1)

    spinner = ["|", "/", "-", "\\"]
    spin = spinner[(int(time.time() * 8) + phase) % len(spinner)]

    cv2.putText(frame, "FACETRUST SYSTEM", (160, 248), cv2.FONT_HERSHEY_SIMPLEX, 1.9, (242, 246, 250), 4, cv2.LINE_AA)
    cv2.putText(frame, "Realtime Face Security Console", (160, 304), cv2.FONT_HERSHEY_SIMPLEX, 0.93, (202, 222, 238), 2, cv2.LINE_AA)
    cv2.putText(frame, f"{spin} {subtitle}", (160, 374), cv2.FONT_HERSHEY_SIMPLEX, 0.90, (112, 210, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "Initializing modules and camera pipeline", (160, 416), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (178, 196, 214), 1, cv2.LINE_AA)

    cv2.imshow(window_name, frame)
    cv2.waitKey(1)


def _draw_capture_overlay(
    frame: np.ndarray,
    title: str,
    subtitle: str,
    color: tuple[int, int, int],
) -> np.ndarray:
    """Draw centered registration overlay text on a frame.

    Args:
        frame: Source frame.
        title: Main overlay text.
        subtitle: Secondary overlay text.
        color: BGR title color.

    Returns:
        Frame copy with overlay graphics.
    """
    overlay = frame.copy()
    h, w = overlay.shape[:2]

    cv2.rectangle(overlay, (0, h - 120), (w, h), (10, 12, 16), -1)
    cv2.addWeighted(overlay, 0.68, frame, 0.32, 0.0, overlay)

    (title_w, _), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 1.05, 2)
    title_x = max(20, (w - title_w) // 2)
    cv2.putText(overlay, title, (title_x, h - 72), cv2.FONT_HERSHEY_SIMPLEX, 1.05, color, 2, cv2.LINE_AA)
    cv2.putText(overlay, subtitle, (34, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (214, 220, 228), 1, cv2.LINE_AA)

    return overlay


def _capture_registration_burst(
    cap: cv2.VideoCapture,
    recognizer: FaceRecognizer,
    window_name: str,
    person_name: str,
    sample_target: int = 5,
) -> tuple[int, Optional[np.ndarray]]:
    """Capture multiple registration samples with visual countdown between captures.

    Args:
        cap: Active webcam capture object.
        recognizer: Face recognizer instance used for sample capture.
        window_name: OpenCV window title.
        person_name: Active identity label for saving samples.
        sample_target: Total samples to save.

    Returns:
        Tuple of `(saved_count, last_captured_face)`.
    """
    saved_count = 0
    last_captured_face: Optional[np.ndarray] = None

    for index in range(sample_target):
        countdown_steps = ["3", "2", "1", "CAPTURE"]
        for step in countdown_steps:
            ok, frame = cap.read()
            if not ok:
                return saved_count, last_captured_face

            stage_frame = _draw_capture_overlay(
                frame,
                step,
                f"Preparing sample {index + 1}/{sample_target} for {person_name}",
                (112, 210, 255) if step != "CAPTURE" else (150, 206, 124),
            )
            cv2.imshow(window_name, stage_frame)
            key = cv2.waitKey(200) & 0xFF
            if key == ord("q"):
                return saved_count, last_captured_face

        ok, frame = cap.read()
        if not ok:
            break

        success, message, captured_crop = recognizer.capture_current_face_sample(frame, person_name)
        if success:
            saved_count += 1
            last_captured_face = captured_crop
            status_title = f"Sample {saved_count}/{sample_target} saved"
            status_color = (150, 206, 124)
            print(message)
        else:
            status_title = f"Sample {index + 1}/{sample_target} failed"
            status_color = (94, 108, 240)
            print(message)

        result_frame = _draw_capture_overlay(
            frame,
            status_title,
            "Hold steady and face the camera naturally",
            status_color,
        )
        cv2.imshow(window_name, result_frame)
        key = cv2.waitKey(250) & 0xFF
        if key == ord("q"):
            break

    return saved_count, last_captured_face


def main() -> None:
    """
    Main realtime pipeline:
    1) Read webcam frame
    2) Detect + recognize face using InsightFace
    3) Run anti-spoofing checks (blink + head motion)
    4) Show final decision on screen
    """
    cap = None
    anti_spoof = None
    session_logger = None
    window_name = "FaceTrust System"

    try:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)
        _show_startup_splash(window_name, "Warming up AI engines...", phase=0)

        recognizer = FaceRecognizer(
            dataset_dir="dataset",
            embeddings_file="embeddings/face_embeddings.npz",
            recognition_threshold=0.55,
        )

        if recognizer.has_embeddings():
            print(f"Loaded {len(recognizer.known_names)} users from embeddings")
        else:
            print("No embeddings found")

        _show_startup_splash(window_name, "Calibrating blink detector...", phase=1)

        anti_spoof = AntiSpoofing(
            ear_threshold=0.22,
            blink_consec_frames=1,
            motion_threshold_px=7.0,
            validity_window_frames=75,
            process_scale=0.65,
            required_blinks_recent=1,
            blink_cooldown_frames=12,
            open_eye_margin=0.015,
            blink_min_drop=0.03,
            reopen_consec_frames=1,
            min_open_frames_before_blink=1,
        )
        session_logger = SessionLogger(logs_dir="logs")

        # Bootstrap embeddings from dataset only when no saved embeddings are available.
        if not recognizer.has_embeddings() and recognizer.dataset_has_images():
            print("No saved embeddings found. Building from dataset folder...")
            _show_startup_splash(window_name, "Building embeddings from dataset...", phase=2)
            added = recognizer.rebuild_embeddings_from_dataset()
            print(f"Embeddings built: {added} samples")

        _show_startup_splash(window_name, "Opening camera and turbo mode...", phase=3)

        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Could not open webcam. Please check camera permissions.")

        is_fullscreen = False

        # Performance-oriented capture settings for smoother realtime display.
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
        cap.set(cv2.CAP_PROP_FPS, 60)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

        active_identity = ""
        entering_name = False
        name_buffer = ""
        last_captured_face = None
        live_preview_face = None
        prev_time = time.time()
        fps = 0.0

        # Ultra-performance mode knobs (maximum smoothness on CPU).
        processing_scale = 0.45
        recognition_every_n = 6
        antispoof_every_n = 2
        frame_index = 0

        cached_detections = []
        cached_spoof_state = {
            "is_real": False,
            "blink_detected": False,
            "motion_detected": False,
            "pose_detected": False,
            "ear": 0.0,
            "blink_count": 0,
            "message": "SPOOF DETECTED",
        }
        last_logged_result: Optional[str] = None
        recent_session_events: list[tuple[str, str]] = []

        print("\nControls:")
        print("  n - set active person name")
        print("  c - capture current face sample for active person")
        print("  r - rebuild all embeddings from dataset")
        print("  q - quit\n")
        print(
            f"Ultra performance mode: downscale={processing_scale:.2f}, "
            f"detection every {recognition_every_n} frames, "
            f"liveness every {antispoof_every_n} frames\n"
        )

        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read frame from webcam.")
                break

            # Keep an untouched frame for detection/capture logic.
            raw_frame = frame.copy()

            frame_index += 1

            # 1) Recognition branch
            if frame_index % recognition_every_n == 0:
                if processing_scale != 1.0:
                    small_frame = cv2.resize(
                        raw_frame,
                        None,
                        fx=processing_scale,
                        fy=processing_scale,
                        interpolation=cv2.INTER_LINEAR,
                    )
                else:
                    small_frame = raw_frame

                small_detections = recognizer.detect_and_recognize(small_frame)

                # Scale boxes back to display frame size.
                if processing_scale != 1.0:
                    scale_back = 1.0 / processing_scale
                    cached_detections = []
                    for det in small_detections:
                        scaled_bbox = (det["bbox"].astype(float) * scale_back).astype(int)
                        cached_detections.append(
                            {
                                "bbox": scaled_bbox,
                                "name": det["name"],
                                "score": det["score"],
                            }
                        )
                else:
                    cached_detections = small_detections

            detections = cached_detections
            face_count = len(detections)
            any_known_face = any(det["name"] != "Unknown" for det in detections)
            best_detection = max(
                detections,
                key=lambda d: max(0, int(d["bbox"][2] - d["bbox"][0])) * max(0, int(d["bbox"][3] - d["bbox"][1])),
            ) if detections else None
            confidence_score = float(best_detection["score"]) if best_detection is not None else 0.0
            confidence_score = max(0.0, min(1.0, confidence_score))

            # 2) Anti-spoofing branch
            if frame_index % antispoof_every_n == 0:
                cached_spoof_state = anti_spoof.process(raw_frame)
            spoof_state = cached_spoof_state

            # 3) Decide global status text and color.
            if face_count == 0:
                system_status = "NO FACE"
                system_color = COLOR_NO_FACE
                last_logged_result = None
            elif spoof_state.get("is_real", False) and any_known_face:
                system_status = "AUTHENTICATED"
                system_color = COLOR_REAL
            elif not spoof_state.get("is_real", False):
                system_status = "SPOOF DETECTED"
                system_color = COLOR_SPOOF
            elif not any_known_face:
                system_status = "UNKNOWN FACE"
                system_color = COLOR_UNKNOWN
            else:
                system_status = "RECOGNIZED"
                system_color = COLOR_UNKNOWN

            ui_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            draw_status_bar(ui_frame, system_status, system_color)

            frame_h, frame_w = ui_frame.shape[:2]
            left_margin = 58
            right_margin = 58
            column_gap = 12
            right_panel_w = min(615, max(540, int(frame_w * 0.45)))
            video_x1 = left_margin
            video_y1 = 154
            video_w = frame_w - left_margin - right_margin - column_gap - right_panel_w
            video_h = min(438, frame_h - video_y1 - 148)
            video_x2 = video_x1 + video_w
            video_y2 = video_y1 + video_h
            inner_pad = 10
            inner_x1 = video_x1 + inner_pad
            inner_y1 = video_y1 + inner_pad
            inner_x2 = video_x2 - inner_pad
            inner_y2 = video_y2 - inner_pad
            inner_w = inner_x2 - inner_x1
            inner_h = inner_y2 - inner_y1

            _draw_panel_shadow(ui_frame, video_x1, video_y1, video_x2, video_y2, offset=8)
            _draw_rounded_rect(ui_frame, video_x1, video_y1, video_x2, video_y2, (8, 8, 10), radius=14, thickness=-1)
            _draw_rounded_rect(ui_frame, video_x1, video_y1, video_x2, video_y2, (26, 28, 32), radius=14, thickness=1)
            _draw_rounded_rect(ui_frame, inner_x1, inner_y1, inner_x2, inner_y2, (42, 42, 46), radius=10, thickness=-1)

            viewport = cv2.resize(raw_frame, (inner_w, inner_h), interpolation=cv2.INTER_LINEAR)

            # Draw boxes on the resized viewport so the framing matches the inset panel.
            raw_h, raw_w = raw_frame.shape[:2]
            scale_x = inner_w / max(1, raw_w)
            scale_y = inner_h / max(1, raw_h)
            for det in detections:
                scaled_bbox = det["bbox"].astype(np.float32).copy()
                scaled_bbox[[0, 2]] *= scale_x
                scaled_bbox[[1, 3]] *= scale_y
                scaled_det = {"bbox": scaled_bbox.astype(int)}
                label = f"{det['name']} ({det['score']:.2f})"
                if det["name"] == "Unknown":
                    box_color = COLOR_UNKNOWN
                elif spoof_state.get("is_real", False):
                    box_color = COLOR_REAL
                else:
                    box_color = COLOR_SPOOF
                draw_face_box(viewport, scaled_det, label=label, color=box_color)

            ui_frame[inner_y1:inner_y2, inner_x1:inner_x2] = viewport

            # Minimal camera accents kept close to the reference screenshot.
            corner_color = (70, 76, 54)
            corner_len = 18
            cv2.line(ui_frame, (inner_x1 + 8, inner_y1 + 8), (inner_x1 + 8 + corner_len, inner_y1 + 8), corner_color, 1)
            cv2.line(ui_frame, (inner_x1 + 8, inner_y1 + 8), (inner_x1 + 8, inner_y1 + 8 + corner_len), corner_color, 1)
            cv2.line(ui_frame, (inner_x2 - 8, inner_y1 + 8), (inner_x2 - 8 - corner_len, inner_y1 + 8), corner_color, 1)
            cv2.line(ui_frame, (inner_x2 - 8, inner_y1 + 8), (inner_x2 - 8, inner_y1 + 8 + corner_len), corner_color, 1)
            cv2.line(ui_frame, (inner_x1 + 8, inner_y2 - 8), (inner_x1 + 8 + corner_len, inner_y2 - 8), corner_color, 1)
            cv2.line(ui_frame, (inner_x1 + 8, inner_y2 - 8), (inner_x1 + 8, inner_y2 - 8 - corner_len), corner_color, 1)
            cv2.line(ui_frame, (inner_x2 - 8, inner_y2 - 8), (inner_x2 - 8 - corner_len, inner_y2 - 8), corner_color, 1)
            cv2.line(ui_frame, (inner_x2 - 8, inner_y2 - 8), (inner_x2 - 8, inner_y2 - 8 - corner_len), corner_color, 1)

            draw_liveness_row(
                ui_frame,
                video_y2,
                blink_detected=spoof_state.get("blink_detected", False),
                movement_detected=spoof_state.get("motion_detected", False),
                pose_detected=spoof_state.get("pose_detected", False),
            )

            if face_count > 0:
                if spoof_state.get("is_real", False) and any_known_face:
                    audit_result = "AUTHENTICATED"
                elif not spoof_state.get("is_real", False):
                    audit_result = "SPOOF_DETECTED"
                else:
                    audit_result = "UNKNOWN"

                if audit_result != last_logged_result and session_logger is not None:
                    best_name = "Unknown"
                    if best_detection is not None and best_detection.get("name") != "Unknown":
                        best_name = str(best_detection["name"])
                    session_logger.log_event(
                        person_name=best_name,
                        result=audit_result,
                        confidence=confidence_score,
                        blink_detected=bool(spoof_state.get("blink_detected", False)),
                        motion_detected=bool(spoof_state.get("motion_detected", False)),
                    )
                    recent_session_events.append((audit_result, datetime.now().strftime("%H:%M")))
                    if len(recent_session_events) > 6:
                        recent_session_events = recent_session_events[-6:]
                    last_logged_result = audit_result

            # Update live preview crop from the largest current detection.
            live_preview_face = None
            if detections:
                largest = max(detections, key=lambda d: max(0, int(d["bbox"][2] - d["bbox"][0])) * max(0, int(d["bbox"][3] - d["bbox"][1])))
                x1, y1, x2, y2 = largest["bbox"].astype(int)
                rh, rw = raw_frame.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(rw, x2), min(rh, y2)
                if x2 > x1 and y2 > y1:
                    live_preview_face = raw_frame[y1:y2, x1:x2].copy()

            # 5) Clean UI blocks.
            draw_info_panel(
                ui_frame,
                face_count=face_count,
                fps=fps,
                confidence=confidence_score,
                identity_name=(str(best_detection["name"]) if best_detection is not None else "Unknown"),
                blink_count=int(spoof_state.get("blink_count", 0)),
                ear_value=float(spoof_state.get("ear", 0.0)),
                ear_threshold=0.22,
                session_events=recent_session_events,
            )
            draw_bottom_bar(ui_frame)

            # 7) Name registration overlay (interaction UI).
            if entering_name:
                draw_name_entry_prompt(
                    ui_frame,
                    name_buffer,
                    last_captured_face if last_captured_face is not None else live_preview_face,
                    blink_detected=spoof_state.get("blink_detected", False),
                    movement_detected=spoof_state.get("motion_detected", False),
                )

            # 8) FPS counter value update.
            now = time.time()
            delta = max(1e-6, now - prev_time)
            instant_fps = 1.0 / delta
            fps = 0.9 * fps + 0.1 * instant_fps if fps > 0 else instant_fps
            prev_time = now

            cv2.imshow(window_name, ui_frame)

            # If user clicks window close button (X), exit cleanly.
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break

            key_code = cv2.waitKeyEx(1)
            key = key_code & 0xFF if key_code != -1 else -1

            if key == ord("q") or key_code == 27:
                break

            # Some backends only update window visibility after processing UI events.
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break

            if key == ord("f"):
                is_fullscreen = not is_fullscreen
                cv2.setWindowProperty(
                    window_name,
                    cv2.WND_PROP_FULLSCREEN,
                    cv2.WINDOW_FULLSCREEN if is_fullscreen else cv2.WINDOW_NORMAL,
                )

            if key == ord("n"):
                entering_name = True
                name_buffer = ""
                print("Name entry mode ON. Type in window, press ENTER to save.")
                continue

            if entering_name:
                # ENTER
                if key in (13, 10):
                    entered = name_buffer.strip()
                    if entered:
                        active_identity = entered
                        print(f"Active identity set to: {active_identity}")
                        # Auto-save one sample immediately for future recognition.
                        success, message, captured_crop = recognizer.capture_current_face_sample(raw_frame, active_identity)
                        print(message)
                        if success:
                            last_captured_face = captured_crop
                            print("Auto-saved initial sample and embedding for this person.")
                        else:
                            print("Could not auto-save sample. Make sure your face is visible and press C.")
                    else:
                        print("Name was empty. Active identity unchanged.")
                    entering_name = False
                    name_buffer = ""
                    continue

                # ESC
                if key == 27:
                    entering_name = False
                    name_buffer = ""
                    print("Name entry cancelled.")
                    continue

                # Backspace
                if key in (8, 127):
                    name_buffer = name_buffer[:-1]
                    continue

                # Printable ASCII characters
                if 32 <= key <= 126:
                    if len(name_buffer) < 32:
                        name_buffer += chr(key)
                    continue

            if key == ord("c"):
                if not active_identity:
                    print("Set a name first using key 'n'.")
                else:
                    saved_count, last_crop = _capture_registration_burst(
                        cap,
                        recognizer,
                        window_name,
                        active_identity,
                        sample_target=5,
                    )
                    if last_crop is not None:
                        last_captured_face = last_crop
                    print(f"Registration burst complete: {saved_count}/5 samples saved.")

            if key == ord("r"):
                count = recognizer.rebuild_embeddings_from_dataset()
                print(f"Embeddings rebuilt from dataset: {count} samples")
    except KeyboardInterrupt:
        print("\nInterrupted by user. Closing application...")
    finally:
        if anti_spoof is not None:
            anti_spoof.close()
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
