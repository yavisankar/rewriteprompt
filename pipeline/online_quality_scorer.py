"""
Online Pipeline — Step 7
Quality Scoring Module:
  - Relevance Score     (keyword overlap between prompt and output)
  - Tone Match Score    (tone consistency check)
  - Semantic Similarity (embedding cosine similarity if model available)
"""

import re
from collections import Counter


TONE_LEXICON = {
    "formal":     ["dear","sincerely","regards","pursuant","hereby","enclosed","please find","attached"],
    "informal":   ["hey","hi","thanks","yeah","ok","sure","cheers","no worries","sounds good"],
    "urgent":     ["urgent","asap","immediately","critical","priority","today","right away"],
    "apologetic": ["sorry","apologies","regret","unfortunately","my mistake","I apologize"],
    "assertive":  ["must","require","expect","insist","need","necessary"],
    "professional":["please","thank you","appreciate","look forward","best regards","kind regards"],
}


class QualityScorer:

    def score(self, original_prompt: str, generated_text: str,
              retrieved_examples: list, intent_ctx: dict) -> dict:

        relevance  = self._relevance_score(original_prompt, generated_text)
        tone_match = self._tone_match_score(generated_text, intent_ctx.get("tone","professional"))
        sem_sim    = self._semantic_similarity(generated_text, retrieved_examples)
        length_ok  = self._length_check(generated_text, intent_ctx.get("style","moderate"))

        # Weighted composite
        composite = round(
            0.35 * relevance +
            0.25 * tone_match +
            0.25 * sem_sim +
            0.15 * length_ok,
            4
        )

        return {
            "relevance_score":    round(relevance, 4),
            "tone_match_score":   round(tone_match, 4),
            "semantic_similarity":round(sem_sim, 4),
            "length_score":       round(length_ok, 4),
            "composite":          composite,
            "passed":             composite >= 0.65,
        }

    # ── Sub-scorers ──────────────────────────────────────────────────────────

    def _relevance_score(self, prompt: str, output: str) -> float:
        """Keyword overlap: what fraction of significant prompt words appear in output."""
        STOP = {"the","a","an","is","in","of","to","for","and","or","it","be","with","that","this","on","at"}
        p_words = {w for w in re.findall(r"\b\w{4,}\b", prompt.lower()) if w not in STOP}
        o_words = set(re.findall(r"\b\w{4,}\b", output.lower()))
        if not p_words:
            return 0.8  # nothing to compare
        overlap = len(p_words & o_words) / len(p_words)
        return min(overlap * 1.5, 1.0)  # scale up slightly

    def _tone_match_score(self, output: str, expected_tone: str) -> float:
        """Check how many tone-marker words appear in the output."""
        markers   = TONE_LEXICON.get(expected_tone, TONE_LEXICON["professional"])
        lower_out = output.lower()
        hits      = sum(1 for m in markers if m in lower_out)
        return min(hits / max(len(markers) * 0.3, 1), 1.0)

    def _semantic_similarity(self, output: str, examples: list) -> float:
        """Cosine similarity between output and retrieved examples (embedding-based if possible)."""
        if not examples:
            return 0.7  # no examples = neutral

        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np
            model = SentenceTransformer("all-MiniLM-L6-v2")
            vecs  = model.encode([output] + examples[:2], normalize_embeddings=True)
            sims  = vecs[0] @ vecs[1:].T
            return float(sims.mean())
        except Exception:
            # Fallback: simple word overlap with examples
            out_words = set(re.findall(r"\b\w{4,}\b", output.lower()))
            scores = []
            for ex in examples[:2]:
                ex_words = set(re.findall(r"\b\w{4,}\b", ex.lower()))
                union = out_words | ex_words
                if union:
                    scores.append(len(out_words & ex_words) / len(union))
            return sum(scores)/len(scores) if scores else 0.5

    def _length_check(self, output: str, style: str) -> float:
        """Score output length against expected style."""
        word_count = len(output.split())
        if style == "brief":
            return 1.0 if word_count <= 100 else max(0, 1 - (word_count-100)/200)
        elif style == "detailed":
            return 1.0 if word_count >= 150 else word_count / 150
        else:  # moderate
            return 1.0 if 50 <= word_count <= 300 else 0.6
