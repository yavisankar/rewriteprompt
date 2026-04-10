"""
Online Pipeline — Step 3 & 4
Query Embedding Generation (Sentence Transformer)
Retrieve top-K Similar Emails (Vector DB Search)
Retrieval Quality Check (low similarity / irrelevant results)
"""

from pipeline.stage3_knowledge_base import KnowledgeBasePipeline

# Thresholds
GOOD_CONTEXT_THRESHOLD = 0.35   # min avg similarity for "good context"
MIN_RESULTS_NEEDED     = 2      # at least N results needed


class RetrievalEngine:
    def __init__(self, kb: KnowledgeBasePipeline):
        self.kb = kb

    def retrieve(self, query: str, top_k: int = 5) -> dict:
        """
        Embed query, search VectorDB, run quality check.
        Returns: good_context bool, examples list, scores list, fallback bool
        """
        fallback = False
        examples = []
        scores   = []

        try:
            result   = self.kb.search(query, top_k=top_k)
            hits     = result.get("results", [])
            examples = [h.get("snippet","") for h in hits if h.get("snippet")]
            scores   = [h.get("score", 0.0) for h in hits]

            # Retrieval quality check
            avg_score = sum(scores) / max(len(scores), 1)
            good_context = (
                len(hits) >= MIN_RESULTS_NEEDED and
                avg_score >= GOOD_CONTEXT_THRESHOLD
            )

            if not good_context:
                fallback = True

        except Exception:
            fallback     = True
            good_context = False

        return {
            "good_context": good_context,
            "examples":     examples[:3],   # use top 3 for context
            "scores":       scores,
            "fallback":     fallback,
            "raw_hits":     hits if not fallback else [],
        }
