"""
Stage 1: Data Curation
Load → Data Understanding → Text Normalization →
Semantic Annotation → Data Quality Check → Filter (Valid / Invalid)
"""

import re, json, os, hashlib
from collections import Counter


class DataCurationPipeline:
    def __init__(self):
        self.raw_data        = []
        self.normalized_data = []
        self.annotated_data  = []
        self.valid_data      = []
        self.invalid_data    = []
        self.stats           = {}

    def load_and_understand(self, csv_path: str, limit: int = 500) -> dict:
        import csv
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV not found: {csv_path}. Place emails.csv in the data/ folder.")

        self.raw_data = []
        with open(csv_path, encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if i >= limit:
                    break
                self.raw_data.append({
                    "id":       hashlib.md5(f"{i}_{row.get('file','')[:50]}".encode()).hexdigest()[:10],
                    "filename": row.get("file", f"email_{i}"),
                    "message":  row.get("message", ""),
                })

        lengths   = [len(r["message"]) for r in self.raw_data]
        avg_len   = sum(lengths) / max(len(lengths), 1)
        short_pct = sum(1 for l in lengths if l < 100) / max(len(lengths), 1) * 100

        self.stats["load"] = {
            "total_loaded":     len(self.raw_data),
            "avg_email_length": round(avg_len, 1),
            "short_email_pct":  round(short_pct, 1),
            "max_length":       max(lengths) if lengths else 0,
            "min_length":       min(lengths) if lengths else 0,
        }
        print(f"[stage1/load] loaded={self.stats['load']['total_loaded']} "
              f"avg_len={self.stats['load']['avg_email_length']} "
              f"short_pct={self.stats['load']['short_email_pct']}")
        if self.raw_data:
            first = self.raw_data[0]
            print(f"[stage1/load:first] id={first.get('id')} file={first.get('filename')} "
                  f"message={first.get('message','')[:120]}")
        return self.stats["load"]

    def normalize_text(self) -> dict:
        if not self.raw_data:
            raise RuntimeError("Run load_and_understand first.")

        self.normalized_data = []
        noise_removed = 0

        for item in self.raw_data:
            text = item["message"]
            if "\n\n" in text:
                text = text.split("\n\n", 1)[1]
            orig_len = len(text)
            text = re.sub(r"http\S+|www\.\S+", " ", text)
            text = re.sub(r"\S+@\S+", " ", text)
            text = re.sub(r"[^a-zA-Z0-9\s.,!?'\"-]", " ", text)
            text = re.sub(r"\s+", " ", text).strip()
            text = text.lower()
            if len(text) < orig_len:
                noise_removed += 1
            self.normalized_data.append({**item, "clean_text": text})

        self.stats["normalize"] = {
            "processed":     len(self.normalized_data),
            "noise_removed": noise_removed,
        }
        print(f"[stage1/normalize] processed={self.stats['normalize']['processed']} "
              f"noise_removed={self.stats['normalize']['noise_removed']}")
        if self.normalized_data:
            first = self.normalized_data[0]
            print(f"[stage1/normalize:first] id={first.get('id')} "
                  f"clean_text={first.get('clean_text','')[:120]}")
        return self.stats["normalize"]

    def semantic_annotation(self) -> dict:
        if not self.normalized_data:
            raise RuntimeError("Run normalize_text first.")

        INTENT_KW = {
            "request":     ["please","could you","can you","would you","request","need"],
            "information": ["fyi","update","inform","let you know","attached","report"],
            "meeting":     ["meeting","call","schedule","calendar","conference","discuss"],
            "approval":    ["approve","approval","sign off","confirm","authorize"],
            "complaint":   ["issue","problem","concern","disappointed","wrong","error"],
        }
        TONE_KW = {
            "formal":   ["dear","sincerely","regards","pursuant","hereby"],
            "informal": ["hey","hi","thanks","yeah","ok","sure"],
            "urgent":   ["urgent","asap","immediately","critical","priority"],
        }
        STYLE = {
            "brief":    lambda t: len(t.split()) < 50,
            "detailed": lambda t: len(t.split()) >= 200,
            "moderate": lambda t: True,
        }

        intent_counts = Counter()
        self.annotated_data = []

        for item in self.normalized_data:
            text = item["clean_text"]
            intent = next((l for l, kws in INTENT_KW.items() if any(k in text for k in kws)), "general")
            tone   = next((l for l, kws in TONE_KW.items() if any(k in text for k in kws)), "neutral")
            style  = next((s for s, fn in STYLE.items() if fn(text)), "moderate")
            intent_counts[intent] += 1
            self.annotated_data.append({**item, "intent": intent, "tone": tone, "style": style})

        self.stats["annotate"] = {
            "annotated":   len(self.annotated_data),
            "intent_dist": dict(intent_counts),
        }
        print(f"[stage1/annotate] annotated={self.stats['annotate']['annotated']} "
              f"intents={self.stats['annotate']['intent_dist']}")
        if self.annotated_data:
            first = self.annotated_data[0]
            print(f"[stage1/annotate:first] id={first.get('id')} intent={first.get('intent')} "
                  f"tone={first.get('tone')} style={first.get('style')}")
        return self.stats["annotate"]

    def quality_check_and_filter(self) -> dict:
        if not self.annotated_data:
            raise RuntimeError("Run semantic_annotation first.")

        self.valid_data   = []
        self.invalid_data = []
        seen              = set()
        reasons           = Counter()

        for item in self.annotated_data:
            text   = item["clean_text"]
            reason = None
            if len(text.strip()) < 20:
                reason = "too_short"
            elif (h := hashlib.md5(text[:200].encode()).hexdigest()) in seen:
                reason = "duplicate"
            elif sum(c.isalpha() for c in text) / max(len(text), 1) < 0.5:
                reason = "noisy_content"
            else:
                seen.add(h)

            if reason:
                reasons[reason] += 1
                self.invalid_data.append({**item, "reject_reason": reason})
            else:
                self.valid_data.append(item)

        os.makedirs("logs", exist_ok=True)
        with open("logs/invalid_data.json", "w") as f:
            json.dump(self.invalid_data[:200], f, indent=2)
        # Also capture a sample of valid rows for inspection.
        with open("logs/valid_data.json", "w") as f:
            json.dump(self.valid_data[:200], f, indent=2)

        self.stats["quality"] = {
            "valid":          len(self.valid_data),
            "invalid":        len(self.invalid_data),
            "reject_reasons": dict(reasons),
        }
        print(f"[stage1/quality] valid={self.stats['quality']['valid']} "
              f"invalid={self.stats['quality']['invalid']} "
              f"reasons={self.stats['quality']['reject_reasons']}")
        if self.valid_data:
            first_valid = self.valid_data[0]
            print(f"[stage1/quality:first_valid] id={first_valid.get('id')} "
                  f"snippet={first_valid.get('clean_text','')[:120]}")
        if self.invalid_data:
            first_invalid = self.invalid_data[0]
            print(f"[stage1/quality:first_invalid] id={first_invalid.get('id')} "
                  f"reason={first_invalid.get('reject_reason')} "
                  f"snippet={first_invalid.get('clean_text','')[:120]}")
        return self.stats["quality"]

    def get_status(self) -> dict:
        return {
            "raw_count":        len(self.raw_data),
            "normalized_count": len(self.normalized_data),
            "annotated_count":  len(self.annotated_data),
            "valid_count":      len(self.valid_data),
            "invalid_count":    len(self.invalid_data),
            "stats":            self.stats,
        }
