from __future__ import annotations

import copy
import json
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .types import EpisodeTelemetry, Lifecycle


class RuntimeStore:
    def __init__(self, telemetry: EpisodeTelemetry, run_dir: Path) -> None:
        self._lock = threading.RLock()
        self.telemetry = telemetry
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._frames: dict[str, bytes] = {}
        self._commands: list[str] = []
        self._stop = threading.Event()
        self._reset = threading.Event()

    @property
    def stop_requested(self) -> bool:
        return self._stop.is_set()

    def request_stop(self) -> None:
        self._stop.set()

    @property
    def reset_requested(self) -> bool:
        return self._reset.is_set()

    def request_reset(self) -> None:
        """Request a fresh simulator process, never an in-place object reset."""

        self._reset.set()
        self._stop.set()

    def state(self) -> dict[str, Any]:
        with self._lock:
            payload = asdict(copy.deepcopy(self.telemetry))
        payload["lifecycle"] = self.telemetry.lifecycle.value
        payload["stage"] = self.telemetry.stage.value
        payload["substate"] = self.telemetry.substate.value
        return payload

    def update(self, **changes: Any) -> None:
        with self._lock:
            for key, value in changes.items():
                setattr(self.telemetry, key, value)

    def queue_command(self, command: str) -> None:
        with self._lock:
            self._commands.append(command)

    def drain_commands(self) -> list[str]:
        with self._lock:
            commands = self._commands[:]
            self._commands.clear()
        return commands

    def set_frame(self, camera: str, image_bgr: np.ndarray) -> None:
        ok, encoded = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 82])
        if ok:
            with self._lock:
                self._frames[camera] = encoded.tobytes()

    def get_frame(self, camera: str) -> bytes | None:
        with self._lock:
            return self._frames.get(camera) or self._frames.get("overview")

    def event(self, event: str, **details: Any) -> None:
        record = {
            "wall_time": time.time(),
            "event": event,
            "state": self.state(),
            "details": details,
        }
        with (self.run_dir / "episode.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    def write_summary(self) -> None:
        payload = self.state()
        payload["written_at"] = time.time()
        (self.run_dir / "summary.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    def apply_command(self, command: str) -> None:
        lifecycle = self.telemetry.lifecycle
        if command == "pause" and lifecycle in {Lifecycle.RUNNING, Lifecycle.RECOVERY}:
            self.update(lifecycle=Lifecycle.PAUSED, message="Paused by operator")
        elif command == "resume" and lifecycle is Lifecycle.PAUSED:
            self.update(lifecycle=Lifecycle.RUNNING, message="Resumed by operator")
        elif command == "start" and lifecycle is Lifecycle.READY:
            self.update(lifecycle=Lifecycle.RUNNING, message="Autonomous policy running")
        self.event("control", command=command)
