"""Liveness evaluation module using blink, motion, and pose-variation cues."""

from collections import deque
from typing import Any, Deque, Dict, Tuple

import cv2
import mediapipe as mp
import numpy as np


class AntiSpoofing:
    """
    Lightweight anti-spoofing using two liveness cues:
    1) Eye blink detection via EAR from MediaPipe landmarks
    2) Head movement detection via nose landmark motion

    Decision rule:
    - REAL if we observed blink + head movement + pose variation recently
    - Otherwise mark as potential spoof
    """

    LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]
    NOSE_TIP_IDX = 1

    def __init__(
        self,
        ear_threshold: float = 0.21,
        blink_consec_frames: int = 1,
        motion_threshold_px: float = 4.0,
        validity_window_frames: int = 75,
        process_scale: float = 0.65,
        required_blinks_recent: int = 1,
        blink_cooldown_frames: int = 8,
        open_eye_margin: float = 0.015,
        blink_min_drop: float = 0.03,
        reopen_consec_frames: int = 1,
        min_open_frames_before_blink: int = 1,
        pose_variation_threshold: float = 0.03,
    ) -> None:
        """Initialize anti-spoofing pipeline and liveness thresholds.

        Args:
            ear_threshold: Eye Aspect Ratio threshold for closed-eye detection.
            blink_consec_frames: Minimum consecutive closed-eye frames for blink candidate.
            motion_threshold_px: Nose-tip motion threshold in pixels.
            validity_window_frames: Window where cues remain valid.
            process_scale: Downscale factor for faster processing.
            required_blinks_recent: Required recent blinks for liveness.
            blink_cooldown_frames: Minimum frame gap between accepted blinks.
            open_eye_margin: Margin above `ear_threshold` to classify eyes as open.
            blink_min_drop: Minimum EAR drop for a valid blink.
            reopen_consec_frames: Open-eye frames required to close a blink cycle.
            min_open_frames_before_blink: Open-eye warmup before first blink can be accepted.
            pose_variation_threshold: Normalized pose signature change threshold.
        """
        self.ear_threshold = ear_threshold
        self.blink_consec_frames = blink_consec_frames
        self.motion_threshold_px = motion_threshold_px
        self.validity_window_frames = validity_window_frames
        self.process_scale = max(0.35, min(1.0, process_scale))
        self.required_blinks_recent = max(1, required_blinks_recent)
        self.blink_cooldown_frames = max(1, blink_cooldown_frames)
        self.open_eye_threshold = self.ear_threshold + max(0.01, open_eye_margin)
        self.blink_min_drop = max(0.02, blink_min_drop)
        self.reopen_consec_frames = max(1, reopen_consec_frames)
        self.min_open_frames_before_blink = max(1, min_open_frames_before_blink)
        self.pose_variation_threshold = max(0.01, pose_variation_threshold)

        self.frame_count = 0
        self.closed_eye_frames = 0
        self.open_eye_frames = 0
        self.blink_count = 0
        self.in_closed_state = False
        self.last_blink_frame = -10_000
        self.blink_frames: Deque[int] = deque()
        self.preclose_ear_peak = 0.0
        self.closed_state_min_ear = 1.0
        self.recent_open_ear = self.open_eye_threshold
        self.open_streak_frames = 0

        self.last_motion_frame = -10_000
        self.last_pose_change_frame = -10_000

        self.prev_nose_xy = None
        self.prev_pose_signature = None

        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def _landmark_to_xy(self, landmark: Any, width: int, height: int) -> Tuple[float, float]:
        """Convert normalized landmark to pixel-space coordinates.

        Args:
            landmark: MediaPipe landmark object with x/y attributes.
            width: Frame width in pixels.
            height: Frame height in pixels.

        Returns:
            Landmark (x, y) coordinates in pixel space.
        """
        return landmark.x * width, landmark.y * height

    def _ear(self, eye_points: np.ndarray) -> float:
        """
        EAR formula:
            (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
        """
        a = np.linalg.norm(eye_points[1] - eye_points[5])
        b = np.linalg.norm(eye_points[2] - eye_points[4])
        c = np.linalg.norm(eye_points[0] - eye_points[3])
        if c < 1e-6:
            return 0.0
        return float((a + b) / (2.0 * c))

    def _extract_eye_points(self, landmarks: Any, indices: list[int], width: int, height: int) -> np.ndarray:
        """Extract eye landmarks as a float32 array in pixel coordinates.

        Args:
            landmarks: MediaPipe landmarks array.
            indices: Landmark indices for one eye contour.
            width: Frame width in pixels.
            height: Frame height in pixels.

        Returns:
            A `(6, 2)` array of eye contour points.
        """
        points = [self._landmark_to_xy(landmarks[i], width, height) for i in indices]
        return np.array(points, dtype=np.float32)

    def process(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Runs anti-spoofing checks on each frame.

        Args:
            frame: Current BGR frame.

        Returns:
            State dictionary including cue flags and final decision:
            `is_real`, `blink_detected`, `motion_detected`, `pose_detected`,
            `ear`, `blink_count`, `message`, `face_present`.
        """
        self.frame_count += 1

        if self.process_scale < 1.0:
            scaled = cv2.resize(
                frame,
                None,
                fx=self.process_scale,
                fy=self.process_scale,
                interpolation=cv2.INTER_LINEAR,
            )
            h, w = scaled.shape[:2]
            frame_rgb = cv2.cvtColor(scaled, cv2.COLOR_BGR2RGB)
        else:
            h, w = frame.shape[:2]
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        result = self.face_mesh.process(frame_rgb)

        state: Dict[str, Any] = {
            "is_real": False,
            "blink_detected": False,
            "motion_detected": False,
            "pose_detected": False,
            "ear": 0.0,
            "blink_count": self.blink_count,
            "message": "SPOOF DETECTED",
            "face_present": False,
        }

        if not result.multi_face_landmarks:
            state["message"] = "SPOOF DETECTED (No stable face landmarks)"
            self.prev_nose_xy = None
            self.prev_pose_signature = None
            self.open_eye_frames = 0
            self.closed_eye_frames = 0
            self.in_closed_state = False
            self.preclose_ear_peak = 0.0
            self.closed_state_min_ear = 1.0
            self.recent_open_ear = self.open_eye_threshold
            self.open_streak_frames = 0
            return state

        landmarks = result.multi_face_landmarks[0].landmark

        left_eye = self._extract_eye_points(landmarks, self.LEFT_EYE_IDX, w, h)
        right_eye = self._extract_eye_points(landmarks, self.RIGHT_EYE_IDX, w, h)
        left_ear = self._ear(left_eye)
        right_ear = self._ear(right_eye)
        ear = (left_ear + right_ear) / 2.0
        both_eyes_closed = (
            left_ear < (self.ear_threshold + 0.005)
            and right_ear < (self.ear_threshold + 0.005)
        )
        average_eye_closed = ear < (self.ear_threshold - 0.005)
        eyes_look_closed = both_eyes_closed or average_eye_closed

        # Blink state machine with hysteresis + cooldown to reduce photo jitter false positives.
        if ear > self.open_eye_threshold:
            self.recent_open_ear = max(self.recent_open_ear * 0.92, ear)
            self.open_streak_frames += 1

        can_start_closed_state = not (
            eyes_look_closed
            and ear < self.ear_threshold
            and self.closed_eye_frames == 0
            and self.open_streak_frames < self.min_open_frames_before_blink
        )

        if eyes_look_closed and ear < self.ear_threshold and can_start_closed_state:
            if self.closed_eye_frames == 0:
                self.preclose_ear_peak = max(self.open_eye_threshold, self.recent_open_ear)
                self.closed_state_min_ear = ear
            else:
                self.closed_state_min_ear = min(self.closed_state_min_ear, ear)

            self.closed_eye_frames += 1
            self.open_eye_frames = 0
            if self.closed_eye_frames >= self.blink_consec_frames:
                self.in_closed_state = True
        else:
            if ear > self.open_eye_threshold:
                self.open_eye_frames += 1

            if self.in_closed_state and self.open_eye_frames >= self.reopen_consec_frames:
                ear_drop = self.preclose_ear_peak - self.closed_state_min_ear
                blink_is_valid = (
                    self.closed_eye_frames >= self.blink_consec_frames
                    and (
                        ear_drop >= self.blink_min_drop
                        or self.closed_state_min_ear < (self.ear_threshold - 0.015)
                    )
                )

                if blink_is_valid and (self.frame_count - self.last_blink_frame) >= self.blink_cooldown_frames:
                    self.blink_count += 1
                    self.last_blink_frame = self.frame_count
                    self.blink_frames.append(self.frame_count)
                self.in_closed_state = False
                self.closed_eye_frames = 0
                self.preclose_ear_peak = 0.0
                self.closed_state_min_ear = 1.0
                self.recent_open_ear = max(self.open_eye_threshold, ear)
                self.open_streak_frames = 1

            if ear > self.open_eye_threshold:
                self.closed_eye_frames = 0

        # Head movement check using nose tip motion
        nose_xy = np.array(self._landmark_to_xy(landmarks[self.NOSE_TIP_IDX], w, h), dtype=np.float32)
        if self.prev_nose_xy is not None:
            motion = float(np.linalg.norm(nose_xy - self.prev_nose_xy))
            if motion > self.motion_threshold_px:
                self.last_motion_frame = self.frame_count
        self.prev_nose_xy = nose_xy

        # Pose-variation cue: normalized nose offset from eye-center is robust to translation
        # (moving a photo), but changes during true 3D head yaw/roll movement.
        left_eye_center = left_eye.mean(axis=0)
        right_eye_center = right_eye.mean(axis=0)
        eye_mid = (left_eye_center + right_eye_center) / 2.0
        inter_eye = float(np.linalg.norm(right_eye_center - left_eye_center))
        if inter_eye > 1e-3:
            pose_signature = float((nose_xy[0] - eye_mid[0]) / inter_eye)
            if self.prev_pose_signature is not None:
                if abs(pose_signature - self.prev_pose_signature) > self.pose_variation_threshold:
                    self.last_pose_change_frame = self.frame_count
            self.prev_pose_signature = pose_signature

        while self.blink_frames and (self.frame_count - self.blink_frames[0]) > self.validity_window_frames:
            self.blink_frames.popleft()

        blink_recent = len(self.blink_frames) >= self.required_blinks_recent
        motion_recent = (self.frame_count - self.last_motion_frame) <= self.validity_window_frames
        pose_recent = (self.frame_count - self.last_pose_change_frame) <= self.validity_window_frames

        # Final decision: all three cues should be recent.
        is_real = blink_recent and motion_recent and pose_recent

        state.update(
            {
                "is_real": is_real,
                "blink_detected": blink_recent,
                "motion_detected": motion_recent,
                "pose_detected": pose_recent,
                "ear": ear,
                "blink_count": self.blink_count,
                "message": "REAL USER" if is_real else "SPOOF DETECTED",
                "face_present": True,
            }
        )

        return state

    def close(self) -> None:
        """Release MediaPipe resources."""
        self.face_mesh.close()
