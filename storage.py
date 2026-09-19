import json
import os
import re
from datetime import datetime
from pathlib import Path

import cv2


def _safe_name(value):
    value = re.sub(r'[<>:"/\\|?*]+', "_", str(value)).strip().strip(".")
    return value or "untitled"


class AnnotationSession:
    """Persist one video's annotations under a date-scoped session directory."""

    def __init__(self, root="captured"):
        self.root = Path(root)
        self.clear()

    def clear(self):
        self.video_path = None
        self.video_title = ""
        self.video_stem = ""
        self.labeling_date = ""
        self.session_dir = None
        self.images_dir = None
        self.metadata_path = None

    def set_video(self, video_path):
        path = Path(video_path)
        self.video_path = path
        self.video_title = path.name
        self.video_stem = _safe_name(path.stem)
        self.labeling_date = datetime.now().strftime("%Y-%m-%d")
        self.session_dir = self.root / f"{self.video_stem}_{self.labeling_date}"
        self.images_dir = self.session_dir / "images"
        self.metadata_path = self.session_dir / "metadata.json"

    def _require_video(self):
        if self.session_dir is None:
            raise RuntimeError("No video is selected.")

    def _ensure_output(self):
        self._require_video()
        self.images_dir.mkdir(parents=True, exist_ok=True)

    def _read_records(self):
        if self.metadata_path is None or not self.metadata_path.is_file():
            return []

        with self.metadata_path.open("r", encoding="utf-8") as handle:
            records = json.load(handle)

        if not isinstance(records, list):
            raise ValueError("metadata.json must contain a list of annotation records.")
        return records

    def _write_records(self, records):
        self._ensure_output()
        temp_path = self.metadata_path.with_suffix(".json.tmp")
        with temp_path.open("w", encoding="utf-8") as handle:
            json.dump(records, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temp_path, self.metadata_path)

    def image_path(self, object_name, frame_number):
        self._require_video()
        filename = f"{_safe_name(object_name)}_{int(frame_number):08d}.jpg"
        return str(self.images_dir / filename)

    def annotations_for_object(self, object_name):
        if self.metadata_path is None:
            return []

        records = [
            record
            for record in self._read_records()
            if record.get("video_title") == self.video_title
            and record.get("object_name") == object_name
        ]
        records.sort(key=lambda record: int(record["frame_number"]))
        return records

    def save_annotation(self, object_name, action, frame_number, image):
        self._ensure_output()
        frame_number = int(frame_number)
        image_path = self.image_path(object_name, frame_number)

        if not cv2.imwrite(image_path, image):
            raise OSError(f"Could not save annotation image: {image_path}")

        record = {
            "video_title": self.video_title,
            "object_name": object_name,
            "action": action,
            "frame_number": frame_number,
        }

        records = self._read_records()
        records = [
            existing
            for existing in records
            if not (
                existing.get("video_title") == self.video_title
                and existing.get("object_name") == object_name
                and int(existing.get("frame_number", -1)) == frame_number
            )
        ]
        records.append(record)
        records.sort(
            key=lambda item: (
                item.get("object_name", ""),
                int(item.get("frame_number", -1)),
            )
        )
        self._write_records(records)
        return image_path

    def delete_annotation(self, object_name, frame_number):
        self._require_video()
        frame_number = int(frame_number)

        records = self._read_records()
        remaining = [
            record
            for record in records
            if not (
                record.get("video_title") == self.video_title
                and record.get("object_name") == object_name
                and int(record.get("frame_number", -1)) == frame_number
            )
        ]

        image_path = Path(self.image_path(object_name, frame_number))
        if image_path.is_file():
            image_path.unlink()

        if remaining:
            self._write_records(remaining)
            return

        if self.metadata_path and self.metadata_path.is_file():
            self.metadata_path.unlink()
        if self.images_dir and self.images_dir.is_dir() and not any(self.images_dir.iterdir()):
            self.images_dir.rmdir()
        if self.session_dir and self.session_dir.is_dir() and not any(self.session_dir.iterdir()):
            self.session_dir.rmdir()

    def output_path(self):
        if self.session_dir is not None and self.session_dir.is_dir():
            return str(self.session_dir)

        self.root.mkdir(parents=True, exist_ok=True)
        return str(self.root)
