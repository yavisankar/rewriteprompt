"""
Stage 3: Knowledge Base Construction
Store valid embeddings in Vector DB → Semantic Search
"""

import json, os
import numpy as np
from datetime import datetime


class KnowledgeBasePipeline:
    def __init__(self):
        self.db_type    = None
        self.collection = None
        self._embeddings = np.array([])
        self._metadata   = []
        self._model      = None
        self.stats       = {}
        self._attach_existing_store()

    def _attach_existing_store(self):
        """
        On server restart, re-attach existing persistent KB automatically.
        Priority:
          1) ChromaDB persistent collection
          2) NumPy flat index files
        """
        # Try ChromaDB first (if installed and collection exists).
        try:
            import chromadb
            client = chromadb.PersistentClient(path="vectordb/chroma")
            self.collection = client.get_collection("enron_emails")
            self.db_type = "ChromaDB (persistent, reloaded)"
            print("[stage3/init] attached existing ChromaDB collection")
            return
        except Exception:
            self.collection = None

        # Fallback to NumPy on-disk vectors.
        self._ensure_loaded()
        if len(self._embeddings) > 0:
            print("[stage3/init] loaded existing NumPy vector index")

    def store_embeddings(self, embeddings: list, metadata: list) -> dict:
        if not embeddings:
            raise ValueError("No embeddings to store.")
        try:
            self._store_chromadb(embeddings, metadata)
        except Exception:
            self._store_numpy(embeddings, metadata)

        self.stats["store"] = {
            "db_type":   self.db_type,
            "stored":    len(embeddings),
            "timestamp": datetime.now().isoformat(),
        }
        print(f"[stage3/store] db_type={self.stats['store']['db_type']} "
              f"stored={self.stats['store']['stored']} "
              f"ts={self.stats['store']['timestamp']}")
        if metadata:
            first = metadata[0]
            print(f"[stage3/store:first] id={first.get('id')} file={first.get('filename')} "
                  f"intent={first.get('intent')} snippet={first.get('snippet','')[:120]}")
        return self.stats["store"]

    def _store_chromadb(self, embeddings, metadata):
        import chromadb
        client = chromadb.PersistentClient(path="vectordb/chroma")
        try:
            client.delete_collection("enron_emails")
        except Exception:
            pass
        self.collection = client.create_collection(
            name="enron_emails",
            metadata={"hnsw:space":"cosine"},
        )
        ids   = [m["id"] for m in metadata]
        docs  = [m["snippet"] for m in metadata]
        metas = [{k:v for k,v in m.items() if k!="snippet"} for m in metadata]
        for i in range(0, len(embeddings), 500):
            self.collection.add(
                embeddings=embeddings[i:i+500],
                documents=docs[i:i+500],
                metadatas=metas[i:i+500],
                ids=ids[i:i+500],
            )
        self.db_type = "ChromaDB (persistent)"

    def _store_numpy(self, embeddings, metadata):
        os.makedirs("vectordb", exist_ok=True)
        self._embeddings = np.array(embeddings, dtype=np.float32)
        self._metadata   = metadata
        np.save("vectordb/embeddings.npy", self._embeddings)
        with open("vectordb/metadata.json","w") as f:
            json.dump(metadata, f)
        self.db_type = "NumPy flat index"

    def _ensure_loaded(self):
        # If Chroma collection is already attached, keep it as source of truth.
        if self.collection is not None:
            return
        if len(self._embeddings) == 0 and os.path.exists("vectordb/embeddings.npy"):
            self._embeddings = np.load("vectordb/embeddings.npy")
            with open("vectordb/metadata.json") as f:
                self._metadata = json.load(f)
            self.db_type = "NumPy flat index (from disk)"

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer("all-MiniLM-L6-v2")
            except ImportError:
                self._model = "tfidf"
        return self._model

    def search(self, query: str, top_k: int = 5) -> dict:
        if self.collection:
            return self._search_chroma(query, top_k)
        return self._search_numpy(query, top_k)

    def _search_chroma(self, query, top_k):
        r = self.collection.query(query_texts=[query], n_results=top_k)
        hits = []
        for i in range(len(r["ids"][0])):
            hits.append({
                "rank":     i+1,
                "id":       r["ids"][0][i],
                "score":    round(1 - r["distances"][0][i], 4),
                "snippet":  r["documents"][0][i],
                "metadata": r["metadatas"][0][i],
            })
        if hits:
            first = hits[0]
            print(f"[stage3/search:first] query={query} id={first.get('id')} "
                  f"score={first.get('score')} snippet={first.get('snippet','')[:120]}")
        return {"query": query, "db": self.db_type, "results": hits}

    def _search_numpy(self, query, top_k):
        self._ensure_loaded()
        if len(self._embeddings) == 0:
            raise RuntimeError("Knowledge base empty. Run offline pipeline first.")

        model = self._get_model()
        if model == "tfidf":
            scores = [sum(1 for w in query.lower().split() if w in m.get("snippet","").lower()) for m in self._metadata]
            top_idx = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]
            results = [
                {"rank":r+1,"id":self._metadata[i]["id"],"score":float(scores[i]),
                 "snippet":self._metadata[i].get("snippet",""),"metadata":self._metadata[i]}
                for r,i in enumerate(top_idx)
            ]
            if results:
                first = results[0]
                print(f"[stage3/search:first] query={query} id={first.get('id')} "
                      f"score={first.get('score')} snippet={first.get('snippet','')[:120]}")
            return {"query": query, "db": "keyword", "results": results}

        q_vec  = model.encode([query], normalize_embeddings=True)[0]
        norms  = np.linalg.norm(self._embeddings, axis=1, keepdims=True) + 1e-9
        sims   = (self._embeddings / norms) @ (q_vec / (np.linalg.norm(q_vec)+1e-9))
        top_idx = np.argsort(-sims)[:top_k]
        results = [
            {"rank":r+1,"id":self._metadata[i]["id"],"score":round(float(sims[i]),4),
             "snippet":self._metadata[i].get("snippet",""),"metadata":self._metadata[i]}
            for r,i in enumerate(top_idx)
        ]
        if results:
            first = results[0]
            print(f"[stage3/search:first] query={query} id={first.get('id')} "
                  f"score={first.get('score')} snippet={first.get('snippet','')[:120]}")
        return {"query": query, "db": self.db_type, "results": results}

    def get_stats(self) -> dict:
        self._ensure_loaded()
        n = 0
        if self.collection is not None:
            try:
                n = int(self.collection.count())
            except Exception:
                n = 0
        elif hasattr(self._embeddings, "__len__"):
            n = len(self._embeddings)
        return {
            "db_type":         self.db_type,
            "stored_vectors":  n,
            "collection_ready": self.collection is not None or n > 0,
            "stats":           self.stats,
        }
