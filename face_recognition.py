"""Face detection, embedding management, and identity matching for FaceTrust."""

import os
import time
from typing import Any, Dict, List, Optional, Tuple, TypedDict

import cv2
import numpy as np
from insightface.app import FaceAnalysis


class DetectionResult(TypedDict):
    """Structured face recognition output for one detected face."""

    bbox: np.ndarray
    name: str
    score: float


class FaceRecognizer:
    """
    Handles:
    - Face detection (RetinaFace via InsightFace FaceAnalysis)
    - Face embedding extraction (ArcFace)
    - Embedding matching
    - Dataset capture and local embedding storage
    """

    def __init__(
        self,
        dataset_dir: str,
        embeddings_file: str,
        recognition_threshold: float = 0.45,
    ) -> None:
        """Initialize face analysis pipeline and local embedding store.

        Args:
            dataset_dir: Root directory holding captured user face images.
            embeddings_file: Path to NPZ file storing names and embeddings.
            recognition_threshold: Cosine-similarity threshold for known identity.
        """
        self.dataset_dir = dataset_dir
        self.embeddings_file = embeddings_file
        self.recognition_threshold = recognition_threshold

        os.makedirs(self.dataset_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.embeddings_file), exist_ok=True)

        self.app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        self.app.prepare(ctx_id=-1, det_size=(320, 320))

        self.known_names: List[str] = []
        self.known_embeddings: np.ndarray = np.empty((0, 512), dtype=np.float32)
        self._load_embeddings()

    def has_embeddings(self) -> bool:
        """Return whether a non-empty embedding database is available."""
        return len(self.known_names) > 0 and self.known_embeddings.shape[0] > 0

    def dataset_has_images(self) -> bool:
        """Check whether dataset directory contains any supported image files."""
        for root, _, files in os.walk(self.dataset_dir):
            for file_name in files:
                if file_name.lower().endswith((".jpg", ".jpeg", ".png")):
                    return True
        return False

    def _load_embeddings(self) -> None:
        """Load and normalize embeddings from NPZ if present."""
        if not os.path.exists(self.embeddings_file):
            return

        try:
            data = np.load(self.embeddings_file, allow_pickle=True)
            names = data["names"].tolist()
            embeddings = data["embeddings"].astype(np.float32)

            if embeddings.ndim == 1:
                embeddings = embeddings.reshape(1, -1)

            valid_count = min(len(names), embeddings.shape[0])
            self.known_names = names[:valid_count]

            if valid_count == 0:
                self.known_embeddings = np.empty((0, 512), dtype=np.float32)
            else:
                normalized_embeddings = [self._normalize_embedding(embeddings[i]) for i in range(valid_count)]
                self.known_embeddings = np.array(normalized_embeddings, dtype=np.float32)

        except Exception:
            self.known_names = []
            self.known_embeddings = np.empty((0, 512), dtype=np.float32)

    def _save_embeddings(self) -> None:
        """Persist current in-memory names and embeddings to NPZ."""
        np.savez(
            self.embeddings_file,
            names=np.array(self.known_names, dtype=object),
            embeddings=self.known_embeddings.astype(np.float32),
        )

    def _normalize_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """Normalize embedding vector to unit length.

        Args:
            embedding: Raw embedding vector.

        Returns:
            L2-normalized embedding vector.
        """
        norm = np.linalg.norm(embedding)
        if norm < 1e-8:
            return embedding.astype(np.float32)
        return (embedding / norm).astype(np.float32)

    @staticmethod
    def _face_area(face: Any) -> float:
        """Compute area for a detected face bounding box.

        Args:
            face: InsightFace face object with `bbox` coordinates.

        Returns:
            Bounding box area as float.
        """
        return float((face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1]))

    def _match_identity(self, embedding: np.ndarray) -> Tuple[str, float]:
        """Match an embedding to the closest known identity.

        Args:
            embedding: Candidate face embedding.

        Returns:
            Tuple of `(name, cosine_similarity_score)`.
        """
        if self.known_embeddings.shape[0] == 0:
            return "Unknown", 0.0

        emb = self._normalize_embedding(embedding)
        best_name = "Unknown"
        best_score = -1.0

        # Compare against every stored sample, but choose the best score per person.
        unique_names = set(self.known_names)
        for person_name in unique_names:
            person_indices = [i for i, name in enumerate(self.known_names) if name == person_name]
            person_embeddings = self.known_embeddings[person_indices]
            person_scores = np.dot(person_embeddings, emb)
            person_best_score = float(np.max(person_scores))

            if person_best_score > best_score:
                best_score = person_best_score
                best_name = person_name

        if best_score >= self.recognition_threshold:
            return best_name, best_score
        return "Unknown", best_score

    def detect_and_recognize(self, frame: np.ndarray) -> List[DetectionResult]:
        """
        Detect faces and recognize identities with ArcFace cosine matching.

        Args:
            frame: BGR frame.

        Returns:
            List of detection results with bbox, identity name, and confidence score.
        """
        faces = self.app.get(frame)
        results: List[DetectionResult] = []

        for face in faces:
            bbox = face.bbox.astype(int)
            name, score = self._match_identity(face.normed_embedding)

            results.append(
                {
                    "bbox": bbox,
                    "name": name,
                    "score": score,
                }
            )

        return results

    def draw_detection(
        self,
        frame: np.ndarray,
        det: DetectionResult,
        label: str,
        color: Tuple[int, int, int],
    ) -> None:
        """Draw a simple bounding box and label for a detection.

        Args:
            frame: Destination frame.
            det: Detection result with `bbox`.
            label: Display label text.
            color: BGR color for rectangle and text.
        """
        x1, y1, x2, y2 = det["bbox"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            label,
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

    def capture_current_face_sample(self, frame: np.ndarray, person_name: str) -> Tuple[bool, str, Optional[np.ndarray]]:
        """
        Captures the most visible face in current frame and stores:
        1) Image in dataset/<person_name>/
        2) Embedding in local NPZ file
        """
        faces = self.app.get(frame)
        if len(faces) == 0:
            return False, "No face detected. Move into camera view and try again.", None

        selected = max(faces, key=self._face_area)
        x1, y1, x2, y2 = selected.bbox.astype(int)

        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return False, "Detected face box is invalid. Please try again.", None

        face_crop = frame[y1:y2, x1:x2]
        person_dir = os.path.join(self.dataset_dir, person_name)
        os.makedirs(person_dir, exist_ok=True)

        file_name = f"{person_name}_{int(time.time())}.jpg"
        image_path = os.path.join(person_dir, file_name)
        cv2.imwrite(image_path, face_crop)

        embedding = self._normalize_embedding(selected.normed_embedding)
        self.known_names.append(person_name)

        if self.known_embeddings.shape[0] == 0:
            self.known_embeddings = embedding.reshape(1, -1)
        else:
            self.known_embeddings = np.vstack([self.known_embeddings, embedding])

        self._save_embeddings()
        return True, f"Saved sample to {image_path}", face_crop.copy()

    def capture_multiple_angle_samples(self, frame: np.ndarray, person_name: str) -> Tuple[int, str]:
        """
        Captures all visible faces in the frame for the same identity.
        This is useful when the user turns slightly and wants more pose coverage.
        """
        faces = self.app.get(frame)
        if len(faces) == 0:
            return 0, "No face detected. Move into camera view and try again."

        person_dir = os.path.join(self.dataset_dir, person_name)
        os.makedirs(person_dir, exist_ok=True)

        saved_count = 0
        for face in faces:
            x1, y1, x2, y2 = face.bbox.astype(int)
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                continue

            face_crop = frame[y1:y2, x1:x2]
            file_name = f"{person_name}_{int(time.time())}_{saved_count}.jpg"
            image_path = os.path.join(person_dir, file_name)
            cv2.imwrite(image_path, face_crop)

            embedding = self._normalize_embedding(face.normed_embedding)
            self.known_names.append(person_name)
            if self.known_embeddings.shape[0] == 0:
                self.known_embeddings = embedding.reshape(1, -1)
            else:
                self.known_embeddings = np.vstack([self.known_embeddings, embedding])

            saved_count += 1

        if saved_count > 0:
            self._save_embeddings()
            return saved_count, f"Saved {saved_count} sample(s) for {person_name}"

        return 0, "No valid face crops were found. Try again."

    def rebuild_embeddings_from_dataset(self) -> int:
        """
        Recreates embedding database by scanning dataset folder.
        Expected layout:
            dataset/
              person1/*.jpg
              person2/*.png
        """
        new_names: List[str] = []
        new_embeddings: List[np.ndarray] = []

        for person_name in sorted(os.listdir(self.dataset_dir)):
            person_dir = os.path.join(self.dataset_dir, person_name)
            if not os.path.isdir(person_dir):
                continue

            for file_name in sorted(os.listdir(person_dir)):
                if not file_name.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue

                image_path = os.path.join(person_dir, file_name)
                image = cv2.imread(image_path)
                if image is None:
                    continue

                faces = self.app.get(image)
                if len(faces) == 0:
                    continue

                selected = max(faces, key=self._face_area)
                new_names.append(person_name)
                new_embeddings.append(self._normalize_embedding(selected.normed_embedding))

        if len(new_embeddings) == 0:
            self.known_names = []
            self.known_embeddings = np.empty((0, 512), dtype=np.float32)
        else:
            self.known_names = new_names
            self.known_embeddings = np.array(new_embeddings, dtype=np.float32)

        self._save_embeddings()
        return len(self.known_names)
