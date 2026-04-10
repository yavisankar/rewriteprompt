"""
Stage 2: Representation Learning
Content Filtering → Pre-trained Embedding Model (Sentence-BERT) →
Generate Embeddings → Embedding Validation → Store Failed
"""

import json, os
import numpy as np
from collections import Counter


class RepresentationLearningPipeline:
    def __init__(self):
        self.filtered_data     = []
        self.valid_embeddings  = []
        self.valid_metadata    = []
        self.failed_embeddings = []
        self.model             = None
        self.model_name        = ""
        self.stats             = {}
        self._raw_embeddings   = None

    def content_filtering(self, valid_data: list) -> dict:
        IRRELEVANT = [
            "unsubscribe","click here","advertisement","sponsored",
            "no-reply","do not reply","auto-generated","out of office",
            "automatic reply","delivery failed","bounced","mailer-daemon",
        ]
        MIN_WORDS = 10
        self.filtered_data = []
        removed = Counter()

        for item in valid_data:
            text  = item["clean_text"].lower()
            words = text.split()
            if len(words) < MIN_WORDS:
                removed["too_few_words"] += 1
                continue
            if any(p in text for p in IRRELEVANT):
                removed["irrelevant_content"] += 1
                continue
            self.filtered_data.append(item)

        self.stats["content_filter"] = {
            "input":   len(valid_data),
            "output":  len(self.filtered_data),
            "removed": dict(removed),
        }
        print(f"[stage2/filter] input={self.stats['content_filter']['input']} "
              f"output={self.stats['content_filter']['output']} "
              f"removed={self.stats['content_filter']['removed']}")
        if self.filtered_data:
            first = self.filtered_data[0]
            print(f"[stage2/filter:first] id={first.get('id')} "
                  f"snippet={first.get('clean_text','')[:120]}")
        return self.stats["content_filter"]

    def _load_model(self):
        if self.model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self.model      = SentenceTransformer("all-MiniLM-L6-v2")
                self.model_name = "all-MiniLM-L6-v2 (Sentence-BERT)"
            except ImportError:
                self.model      = "tfidf_fallback"
                self.model_name = "TF-IDF + SVD fallback (install sentence-transformers)"

    def generate_embeddings(self) -> dict:
        if not self.filtered_data:
            raise RuntimeError("Run content_filtering first.")
        self._load_model()
        texts = [item["clean_text"][:512] for item in self.filtered_data]

        if self.model == "tfidf_fallback":
            embeddings = self._tfidf_embeddings(texts)
        else:
            embeddings = self.model.encode(texts, show_progress_bar=False,
                                           batch_size=32, normalize_embeddings=True)

        self._raw_embeddings = embeddings
        self.stats["embeddings"] = {
            "model":      self.model_name,
            "count":      len(embeddings),
            "dimensions": embeddings.shape[1] if hasattr(embeddings, "shape") else len(embeddings[0]),
        }
        print(f"[stage2/embeddings] model={self.stats['embeddings']['model']} "
              f"count={self.stats['embeddings']['count']} "
              f"dim={self.stats['embeddings']['dimensions']}")
        if len(embeddings):
            first_emb = embeddings[0]
            preview = first_emb[:5].tolist() if hasattr(first_emb, "tolist") else first_emb[:5]
            print(f"[stage2/embeddings:first] values={preview}")
        return self.stats["embeddings"]

    def _tfidf_embeddings(self, texts):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD
        vec  = TfidfVectorizer(max_features=5000, stop_words="english")
        X    = vec.fit_transform(texts)
        n    = min(128, X.shape[1]-1, X.shape[0]-1)
        svd  = TruncatedSVD(n_components=n)
        return svd.fit_transform(X)

    def validate_embeddings(self) -> dict:
        if self._raw_embeddings is None:
            raise RuntimeError("Run generate_embeddings first.")

        embeddings = np.array(self._raw_embeddings)
        self.valid_embeddings  = []
        self.valid_metadata    = []
        self.failed_embeddings = []
        reasons = Counter()

        for emb, item in zip(embeddings, self.filtered_data):
            reason = None
            if np.allclose(emb, 0, atol=1e-6):
                reason = "zero_vector"
            elif np.var(emb) < 1e-8:
                reason = "low_variance"
            elif not np.all(np.isfinite(emb)):
                reason = "nan_or_inf"

            if reason:
                reasons[reason] += 1
                self.failed_embeddings.append({
                    "id": item["id"], "reason": reason,
                    "snippet": item["clean_text"][:80],
                })
            else:
                self.valid_embeddings.append(emb.tolist())
                self.valid_metadata.append({
                    "id":       item["id"],
                    "filename": item["filename"],
                    "intent":   item.get("intent","general"),
                    "tone":     item.get("tone","neutral"),
                    "style":    item.get("style","moderate"),
                    "snippet":  item["clean_text"][:200],
                })

        os.makedirs("logs", exist_ok=True)
        with open("logs/failed_embeddings.json","w") as f:
            json.dump(self.failed_embeddings[:100], f, indent=2)

        self.stats["validation"] = {
            "valid":   len(self.valid_embeddings),
            "failed":  len(self.failed_embeddings),
            "reasons": dict(reasons),
        }
        print(f"[stage2/validate] valid={self.stats['validation']['valid']} "
              f"failed={self.stats['validation']['failed']} "
              f"reasons={self.stats['validation']['reasons']}")
        if self.valid_metadata:
            first_valid = self.valid_metadata[0]
            print(f"[stage2/validate:first_valid] id={first_valid.get('id')} "
                  f"intent={first_valid.get('intent')} tone={first_valid.get('tone')} "
                  f"style={first_valid.get('style')}")
        if self.failed_embeddings:
            first_failed = self.failed_embeddings[0]
            print(f"[stage2/validate:first_failed] id={first_failed.get('id')} "
                  f"reason={first_failed.get('reason')}")
        return self.stats["validation"]

    def get_status(self) -> dict:
        return {
            "filtered_count":    len(self.filtered_data),
            "valid_embeddings":  len(self.valid_embeddings),
            "failed_embeddings": len(self.failed_embeddings),
            "stats":             self.stats,
        }
