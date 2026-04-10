"""
Online Pipeline — Step 8
Memory Store — persist high-quality generated outputs for future context.
Simple JSON-backed in-memory + disk store.
"""

import json, os, uuid
from datetime import datetime


class MemoryStore:
    def __init__(self, path: str = "memory/good_cases.json"):
        self.path    = path
        self._memory = []
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path) as f:
                    self._memory = json.load(f)
            except Exception:
                self._memory = []

    def _save(self):
        with open(self.path, "w") as f:
            json.dump(self._memory, f, indent=2)

    def store(self, record: dict) -> str:
        mem_id = str(uuid.uuid4())[:8]
        entry  = {
            "id":        mem_id,
            "timestamp": datetime.now().isoformat(),
            **record,
        }
        self._memory.insert(0, entry)
        # Keep latest 200 in memory
        self._memory = self._memory[:200]
        self._save()
        return mem_id

    def get_all(self) -> list:
        return self._memory

    def count(self) -> int:
        return len(self._memory)

    def clear(self):
        self._memory = []
        self._save()
