"""
Enron Email Intelligence Platform
Offline Pipeline  → Data Curation + Representation Learning + Knowledge Base
Online Pipeline   → RAG-based Email Generation with ChatGPT + Quality Scoring
"""

from flask import Flask, jsonify, request, render_template
import traceback, os, json

from pipeline.stage1_data_curation        import DataCurationPipeline
from pipeline.stage2_representation       import RepresentationLearningPipeline
from pipeline.stage3_knowledge_base       import KnowledgeBasePipeline
from pipeline.online_prompt_processor     import PromptProcessor
from pipeline.online_retrieval            import RetrievalEngine
from pipeline.online_prompt_rewriter      import PromptRewriter
from pipeline.online_llm_generator        import LLMGenerator
from pipeline.online_quality_scorer       import QualityScorer
from pipeline.online_memory               import MemoryStore

app = Flask(__name__)

def load_env_file(env_path=".env"):
    """Minimal .env loader to avoid extra dependency."""
    if not os.path.exists(env_path):
        return
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            # Keep existing environment values if already set by shell/OS.
            os.environ.setdefault(key, value)

load_env_file()

def _preview_text(value, limit=240):
    if value is None:
        return ""
    text = str(value)
    return text[:limit]

def _find_by_id(rows, row_id):
    return next((x for x in rows if x.get("id") == row_id), None)

# ── Instantiate all pipeline stages ─────────────────────────────────────────
stage1   = DataCurationPipeline()
stage2   = RepresentationLearningPipeline()
stage3   = KnowledgeBasePipeline()

# Online components (shared instances)
prompt_proc  = PromptProcessor()
retrieval    = RetrievalEngine(stage3)
rewriter     = PromptRewriter()
generator    = LLMGenerator()
scorer       = QualityScorer()
memory       = MemoryStore()

# ── UI ───────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

# ════════════════════════════════════════════════════════════════════════════
# OFFLINE PIPELINE — Stage 1: Data Curation
# ════════════════════════════════════════════════════════════════════════════
@app.route("/api/offline/stage1/load", methods=["POST"])
def s1_load():
    d = request.get_json()
    try:
        csv_path = d.get("csv_path","data/emails.csv")
        limit    = int(d.get("limit",500))
        print(f"[offline/s1_load] path={csv_path} limit={limit}")
        r = stage1.load_and_understand(csv_path, limit)
        print(f"[offline/s1_load] done stats={r}")
        first = stage1.raw_data[0] if stage1.raw_data else None
        return jsonify({
            "status":"success",
            "data":r,
            "first_record_behavior": {
                "stage": "stage1_load",
                "input": {"csv_path": csv_path, "limit": limit},
                "output_first_record": first,
            }
        })
    except Exception as e:
        return jsonify({"status":"error","message":str(e),"trace":traceback.format_exc()}), 500

@app.route("/api/offline/stage1/normalize", methods=["POST"])
def s1_normalize():
    try:
        print("[offline/s1_normalize] start")
        r = stage1.normalize_text()
        print(f"[offline/s1_normalize] done stats={r}")
        first_after = stage1.normalized_data[0] if stage1.normalized_data else None
        first_before = _find_by_id(stage1.raw_data, first_after.get("id")) if first_after else None
        return jsonify({
            "status":"success",
            "data":r,
            "first_record_behavior": {
                "stage": "stage1_normalize",
                "before": {
                    "id": first_before.get("id") if first_before else None,
                    "message": _preview_text(first_before.get("message")) if first_before else "",
                },
                "after": {
                    "id": first_after.get("id") if first_after else None,
                    "clean_text": _preview_text(first_after.get("clean_text")) if first_after else "",
                },
            }
        })
    except Exception as e:
        return jsonify({"status":"error","message":str(e)}), 500

@app.route("/api/offline/stage1/annotate", methods=["POST"])
def s1_annotate():
    try:
        print("[offline/s1_annotate] start")
        r = stage1.semantic_annotation()
        print(f"[offline/s1_annotate] done stats={r}")
        first_after = stage1.annotated_data[0] if stage1.annotated_data else None
        first_before = _find_by_id(stage1.normalized_data, first_after.get("id")) if first_after else None
        return jsonify({
            "status":"success",
            "data":r,
            "first_record_behavior": {
                "stage": "stage1_annotate",
                "before": {
                    "id": first_before.get("id") if first_before else None,
                    "clean_text": _preview_text(first_before.get("clean_text")) if first_before else "",
                },
                "after": first_after,
            }
        })
    except Exception as e:
        return jsonify({"status":"error","message":str(e)}), 500

@app.route("/api/offline/stage1/quality_check", methods=["POST"])
def s1_quality():
    try:
        print("[offline/s1_quality] start")
        r = stage1.quality_check_and_filter()
        print(f"[offline/s1_quality] done stats={r}")
        return jsonify({
            "status":"success",
            "data":r,
            "first_record_behavior": {
                "stage": "stage1_quality",
                "first_valid": stage1.valid_data[0] if stage1.valid_data else None,
                "first_invalid": stage1.invalid_data[0] if stage1.invalid_data else None,
            }
        })
    except Exception as e:
        return jsonify({"status":"error","message":str(e)}), 500

@app.route("/api/offline/stage1/status", methods=["GET"])
def s1_status():
    status = stage1.get_status()
    first = stage1.valid_data[0] if stage1.valid_data else (stage1.raw_data[0] if stage1.raw_data else None)
    if first:
        print(f"[offline/s1_status:first] id={first.get('id')} file={first.get('filename')}")
    return jsonify(status)

# ════════════════════════════════════════════════════════════════════════════
# OFFLINE PIPELINE — Stage 2: Representation Learning
# ════════════════════════════════════════════════════════════════════════════
@app.route("/api/offline/stage2/content_filter", methods=["POST"])
def s2_filter():
    try:
        print("[offline/s2_filter] start")
        r = stage2.content_filtering(stage1.valid_data)
        print(f"[offline/s2_filter] done stats={r}")
        first_after = stage2.filtered_data[0] if stage2.filtered_data else None
        first_before = _find_by_id(stage1.valid_data, first_after.get("id")) if first_after else None
        return jsonify({
            "status":"success",
            "data":r,
            "first_record_behavior": {
                "stage": "stage2_filter",
                "before": first_before,
                "after": first_after,
            }
        })
    except Exception as e:
        return jsonify({"status":"error","message":str(e)}), 500

@app.route("/api/offline/stage2/generate_embeddings", methods=["POST"])
def s2_embed():
    try:
        print("[offline/s2_embed] start")
        r = stage2.generate_embeddings()
        print(f"[offline/s2_embed] done stats={r}")
        first_item = stage2.filtered_data[0] if stage2.filtered_data else None
        first_vec = stage2._raw_embeddings[0].tolist() if getattr(stage2, "_raw_embeddings", None) is not None and len(stage2._raw_embeddings) else None
        return jsonify({
            "status":"success",
            "data":r,
            "first_record_behavior": {
                "stage": "stage2_embeddings",
                "input_text": _preview_text(first_item.get("clean_text")) if first_item else "",
                "embedding_first_12_values": first_vec[:12] if first_vec else [],
            }
        })
    except Exception as e:
        return jsonify({"status":"error","message":str(e),"trace":traceback.format_exc()}), 500

@app.route("/api/offline/stage2/validate_embeddings", methods=["POST"])
def s2_validate():
    try:
        print("[offline/s2_validate] start")
        r = stage2.validate_embeddings()
        print(f"[offline/s2_validate] done stats={r}")
        return jsonify({
            "status":"success",
            "data":r,
            "first_record_behavior": {
                "stage": "stage2_validate",
                "first_valid_metadata": stage2.valid_metadata[0] if stage2.valid_metadata else None,
                "first_valid_embedding_first_12_values": stage2.valid_embeddings[0][:12] if stage2.valid_embeddings else [],
                "first_failed_embedding": stage2.failed_embeddings[0] if stage2.failed_embeddings else None,
            }
        })
    except Exception as e:
        return jsonify({"status":"error","message":str(e)}), 500

@app.route("/api/offline/stage2/status", methods=["GET"])
def s2_status():
    status = stage2.get_status()
    first = stage2.valid_metadata[0] if stage2.valid_metadata else (stage2.filtered_data[0] if stage2.filtered_data else None)
    if first:
        print(f"[offline/s2_status:first] id={first.get('id')} "
              f"snippet={first.get('snippet', first.get('clean_text', ''))[:120]}")
    return jsonify(status)

# ════════════════════════════════════════════════════════════════════════════
# OFFLINE PIPELINE — Stage 3: Knowledge Base
# ════════════════════════════════════════════════════════════════════════════
@app.route("/api/offline/stage3/store", methods=["POST"])
def s3_store():
    try:
        print("[offline/s3_store] start")
        r = stage3.store_embeddings(stage2.valid_embeddings, stage2.valid_metadata)
        print(f"[offline/s3_store] done stats={r}")
        return jsonify({
            "status":"success",
            "data":r,
            "first_record_behavior": {
                "stage": "stage3_store",
                "first_metadata_stored": stage2.valid_metadata[0] if stage2.valid_metadata else None,
                "first_embedding_first_12_values": stage2.valid_embeddings[0][:12] if stage2.valid_embeddings else [],
            }
        })
    except Exception as e:
        return jsonify({"status":"error","message":str(e)}), 500

@app.route("/api/offline/stage3/search", methods=["POST"])
def s3_search():
    d = request.get_json()
    q = d.get("query","")
    k = int(d.get("top_k",5))
    if not q:
        return jsonify({"status":"error","message":"query required"}), 400
    try:
        result = stage3.search(q,k)
        first = result["results"][0] if result.get("results") else None
        if first:
            print(f"[offline/s3_search:first] id={first.get('id')} score={first.get('score')} "
                  f"snippet={first.get('snippet','')[:120]}")
        return jsonify({
            "status":"success",
            "data":result,
            "first_record_behavior": {
                "stage": "stage3_search",
                "query": q,
                "first_hit": result["results"][0] if result.get("results") else None,
            }
        })
    except Exception as e:
        return jsonify({"status":"error","message":str(e)}), 500

@app.route("/api/offline/stage3/stats", methods=["GET"])
def s3_stats():
    stats = stage3.get_stats()
    if stage3._metadata:
        first = stage3._metadata[0]
        print(f"[offline/s3_stats:first] id={first.get('id')} file={first.get('filename')}")
    return jsonify(stats)

# ════════════════════════════════════════════════════════════════════════════
# OFFLINE PIPELINE — Full run
# ════════════════════════════════════════════════════════════════════════════
@app.route("/api/offline/pipeline/run", methods=["POST"])
def run_offline():
    d     = request.get_json()
    csv   = d.get("csv_path","data/emails.csv")
    limit = int(d.get("limit",200))
    report = {}
    try:
        print(f"[offline/pipeline] start csv={csv} limit={limit}")
        report["stage1_load"]       = stage1.load_and_understand(csv, limit)
        report["stage1_normalize"]  = stage1.normalize_text()
        report["stage1_annotate"]   = stage1.semantic_annotation()
        report["stage1_quality"]    = stage1.quality_check_and_filter()
        report["stage2_filter"]     = stage2.content_filtering(stage1.valid_data)
        report["stage2_embeddings"] = stage2.generate_embeddings()
        report["stage2_validation"] = stage2.validate_embeddings()
        report["stage3_store"]      = stage3.store_embeddings(stage2.valid_embeddings, stage2.valid_metadata)
        print("[offline/pipeline] complete")
        report["knowledge_base_ready"] = True
        return jsonify({"status":"success","report":report})
    except Exception as e:
        return jsonify({"status":"error","message":str(e),"partial_report":report,"trace":traceback.format_exc()}), 500

# ════════════════════════════════════════════════════════════════════════════
# ONLINE PIPELINE — Full RAG Email Generation
# ════════════════════════════════════════════════════════════════════════════
@app.route("/api/online/generate", methods=["POST"])
def online_generate():
    """
    Full online pipeline:
    User Prompt → Preprocess → Intent/Context → Embed Query → Retrieve → 
    Quality Check Context → Prompt Rewrite → LLM Generate → Score → 
    Retry if needed → Store in Memory
    """
    d          = request.get_json()
    user_prompt = d.get("prompt", "")
    api_key    = (d.get("openai_api_key") or os.getenv("OPENAI_API_KEY", "")).strip()
    max_retries = int(d.get("max_retries", 2))
    threshold   = float(d.get("score_threshold", 0.65))

    if not user_prompt:
        return jsonify({"status":"error","message":"prompt is required"}), 400
    if not api_key:
        return jsonify({"status":"error","message":"OpenAI API key is required"}), 400
    if "*" in api_key:
        return jsonify({"status":"error","message":"Invalid API key format. Use a full key from OpenAI dashboard."}), 400

    print(f"[online] prompt_len={len(user_prompt)} "
          f"threshold={threshold} max_retries={max_retries} "
          f"key_source={'body' if d.get('openai_api_key') else 'env'}")
    pipeline_trace = []
    try:
        # Step 1: Prompt preprocessing
        cleaned = prompt_proc.preprocess(user_prompt)
        print(f"[online] cleaned_len={len(cleaned)}")
        pipeline_trace.append({"step":"preprocess","output":cleaned})

        # Step 2: Intent & context extraction
        intent_ctx = prompt_proc.extract_intent_context(cleaned)
        print(f"[online] intent_ctx={intent_ctx}")
        pipeline_trace.append({"step":"intent_extraction","output":intent_ctx})

        # Step 3: Query embedding + retrieval
        retrieved = retrieval.retrieve(cleaned, top_k=10)
        print(f"[online] retrieved examples={len(retrieved['examples'])} "
              f"good_context={retrieved['good_context']} "
              f"fallback={retrieved['fallback']}")
        if retrieved["examples"]:
            print(f"[online/retrieval:first] snippet={retrieved['examples'][0][:140]}")
        pipeline_trace.append({"step":"retrieval","output":{
            "good_context": retrieved["good_context"],
            "similarity_scores": retrieved["scores"][:3],
            "fallback_used": retrieved["fallback"],
            "top_hits_count": len(retrieved.get("raw_hits", [])),
        }})

        # Step 4: Prompt rewriting (personalisation)
        rewritten = rewriter.rewrite(
            original_prompt=cleaned,
            intent_ctx=intent_ctx,
            retrieved_examples=retrieved["examples"],
            use_rag=retrieved["good_context"],
        )
        print(f"[online/rewrite:first] prompt_preview={rewritten['prompt'][:180]}")
        pipeline_trace.append({"step":"prompt_rewrite","output":{"rewritten_prompt": rewritten["prompt"]}})

        # Step 5–7: Generate → Score → Retry loop
        attempt        = 0
        final_output   = None
        final_scores   = None
        retry_history  = []

        while attempt <= max_retries:
            # Generate with ChatGPT
            print(f"[online] attempt={attempt+1} calling LLM")
            generated = generator.generate(rewritten["prompt"], api_key, intent_ctx)
            print(f"[online/generate:first] text_preview={generated['text'][:180]}")

            # Score output
            scores = scorer.score(
                original_prompt=user_prompt,
                generated_text=generated["text"],
                retrieved_examples=retrieved["examples"],
                intent_ctx=intent_ctx,
            )
            composite = scores["composite"]
            print(f"[online/score:first] scores={scores}")

            retry_history.append({"attempt": attempt+1, "score": composite, "text_snippet": generated["text"]})

            if composite >= threshold:
                final_output = generated
                final_scores = scores
                break
            else:
                # Retry: adjust prompt (change tone, add context)
                rewritten = rewriter.adjust_for_retry(rewritten["prompt"], scores, attempt)
                print(f"[online] retrying attempt={attempt+1} score={composite}")
                attempt += 1

        if final_output is None:
            # Use last attempt even if below threshold
            final_output = generated
            final_scores = scores

        pipeline_trace.append({"step":"generate_score_loop","output":{
            "attempts": len(retry_history),
            "final_score": final_scores["composite"],
            "passed_threshold": final_scores["composite"] >= threshold,
            "retry_history": retry_history,
        }})

        # Step 8: Store in memory if good
        mem_id = None
        if final_scores["composite"] >= threshold:
            mem_id = memory.store({
                "prompt":    user_prompt,
                "intent":    intent_ctx.get("intent","general"),
                "tone":      intent_ctx.get("tone","neutral"),
                "output":    final_output["text"],
                "score":     final_scores["composite"],
            })
        pipeline_trace.append({"step":"memory_store","output":{"stored": mem_id is not None, "memory_id": mem_id}})

        print(f"[online] done score={final_scores['composite']} "
              f"stored_in_memory={final_scores['composite'] >= threshold}")
        return jsonify({
            "status":          "success",
            "final_output":    final_output["text"],
            "scores":          final_scores,
            "intent_context":  intent_ctx,
            "pipeline_trace":  pipeline_trace,
            "retrieved_count": len(retrieved.get("raw_hits", [])),
            "fallback_used":   retrieved["fallback"],
            "attempts":        len(retry_history),
            "semantic_matches_top10": retrieved.get("raw_hits", [])[:10],
            "first_record_behavior": {
                "stage": "online_generate",
                "user_prompt": user_prompt,
                "after_preprocess": cleaned,
                "retrieval_first_example": retrieved["examples"][0] if retrieved["examples"] else None,
                "after_rewrite_prompt": rewritten["prompt"],
                "first_generated_text": final_output["text"],
                "scores": final_scores,
            },
        })

    except Exception as e:
        return jsonify({"status":"error","message":str(e),"trace":traceback.format_exc(),"pipeline_trace":pipeline_trace}), 500


@app.route("/api/online/memory", methods=["GET"])
def get_memory():
    data = memory.get_all()
    if data:
        first = data[0]
        print(f"[online/memory:first] id={first.get('id')} score={first.get('score')} "
              f"prompt={first.get('prompt','')[:120]}")
    return jsonify(data)

@app.route("/api/online/memory/clear", methods=["POST"])
def clear_memory():
    before = memory.get_all()
    if before:
        first = before[0]
        print(f"[online/memory_clear:first_before] id={first.get('id')} score={first.get('score')}")
    memory.clear()
    print("[online/memory_clear] done")
    return jsonify({"status":"success","message":"Memory cleared"})

@app.route("/api/online/status", methods=["GET"])
def online_status():
    payload = {
        "kb_ready":     stage3.get_stats()["collection_ready"],
        "memory_count": memory.count(),
        "model":        "gpt-4o-mini (ChatGPT)",
    }
    print(f"[online/status] {payload}")
    return jsonify(payload)

# ════════════════════════════════════════════════════════════════════════════
# Logs
# ════════════════════════════════════════════════════════════════════════════
@app.route("/api/logs/<log_type>", methods=["GET"])
def get_logs(log_type):
    path = f"logs/{log_type}.json"
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
            if data:
                first = data[0]
                print(f"[logs/{log_type}:first] keys={list(first.keys())} preview={str(first)[:160]}")
            return jsonify(data)
    return jsonify([])

if __name__ == "__main__":
    app.run(debug=True, port=5000)
