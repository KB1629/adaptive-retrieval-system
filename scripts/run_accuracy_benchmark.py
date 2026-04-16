#!/usr/bin/env python3
"""
REAL-MM-RAG Accuracy Benchmark for the Adaptive Retrieval System.

Measures retrieval accuracy metrics:
  - Recall@1, Recall@5, Recall@10
  - MRR (Mean Reciprocal Rank)
  - NDCG@10 (Normalized Discounted Cumulative Gain)

Pipeline:
  1. Load REAL-MM-RAG corpus pages + queries + ground-truth labels from HuggingFace
  2. Index corpus pages through adaptive pipeline (router → text/vision embedding → LanceDB)
  3. For each query: encode → search top-10 → collect ranked doc_ids
  4. Compute and report accuracy metrics

Usage (quick smoke test, ~2 min):
  python scripts/run_accuracy_benchmark.py --max-pages 50 --max-queries 10

Usage (full run, ~20-30 min on M1 Pro):
  python scripts/run_accuracy_benchmark.py --max-pages 500 --max-queries 100

Usage (full dataset):
  python scripts/run_accuracy_benchmark.py
"""

import argparse
import json
import logging
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

# ── Project root on sys.path ─────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.router.vision_router import VisionRouter
from src.router.base import RouterConfig
from src.embedding.text_path import TextEmbeddingPath
from src.embedding.vision_path import VisionEmbeddingPath
from src.embedding.text_embedder import TextEmbedder
from src.storage.lancedb_backend import LanceDBBackend
from src.storage.base import DocumentMetadata
from src.benchmark.metrics import evaluate_retrieval
from src.models.data import Page
from src.utils.hardware import detect_device, get_hardware_config

logging.basicConfig(
    level=logging.WARNING,          # suppress library noise
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("accuracy_benchmark")
logger.setLevel(logging.INFO)


# ── Dataset loading ───────────────────────────────────────────────────────────

def load_real_mm_rag(
    variant: str = "techreport",
    max_pages: Optional[int] = None,
    max_queries: Optional[int] = None,
    cache_dir: Optional[str] = None,
):
    """
    Load REAL-MM-RAG dataset from HuggingFace.

    Dataset structure (single `test` split, each row has):
      id, image, image_filename, query, rephrase_level_1/2/3, answer

    We treat:
      - Corpus  = all unique images (indexed by str(id))
      - Queries = all query strings  (query_id = str(id))
      - Qrels   = query_id -> {str(id)}  (each query is relevant to its own page)

    Returns:
        pages   – list of (doc_id: str, page_num: int, PIL.Image)
        queries – list of (query_id: str, query_text: str)
        qrels   – dict  query_id -> set of relevant doc_ids
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise RuntimeError("Install datasets: pip install datasets")

    hf_ids = {
        "techreport": "ibm-research/REAL-MM-RAG_TechReport",
        "techslides":  "ibm-research/REAL-MM-RAG_TechSlides",
    }
    if variant not in hf_ids:
        raise ValueError(f"variant must be 'techreport' or 'techslides', got '{variant}'")

    hf_id = hf_ids[variant]
    print(f"\n📦 Loading dataset: {hf_id}")
    print("   (First run downloads ~1-3 GB; subsequent runs use cache)\n")

    kwargs = {}
    if cache_dir:
        kwargs["cache_dir"] = cache_dir

    ds = load_dataset(hf_id, **kwargs)
    # Dataset has a single 'test' split — each row = one (page, query, answer)
    split_name = "test" if "test" in ds else list(ds.keys())[0]
    rows = ds[split_name]

    pages:   list = []
    queries: list = []
    qrels:   dict[str, set[str]] = {}

    page_limit = max_pages or len(rows)
    q_limit    = max_queries or len(rows)

    for idx, item in enumerate(rows):
        if idx >= page_limit:
            break

        doc_id = str(item.get("id", idx))
        image  = item.get("image")
        if image is None:
            continue

        # Extract page number from filename, e.g. "report_page_15.png"
        fname    = item.get("image_filename", "")
        page_num = 1
        if "_page_" in fname:
            try:
                page_num = int(fname.split("_page_")[-1].split(".")[0])
            except ValueError:
                pass

        pages.append((doc_id, page_num, image))

        # Each row has a query — ground truth = this page (1-to-1)
        if idx < q_limit:
            q_txt = item.get("query", "")
            if q_txt:
                q_id = doc_id   # query_id == doc_id
                queries.append((q_id, q_txt))
                qrels[q_id] = {doc_id}

    print(f"  Corpus pages loaded : {len(pages)}")
    print(f"  Queries loaded      : {len(queries)}")
    print(f"  Qrels               : {len(qrels)} (1-to-1 — each query → its own page)\n")
    return pages, queries, qrels


# ── Pipeline initialisation ───────────────────────────────────────────────────

def init_pipeline():
    """Load all models and return (router, text_path, vision_path, text_query_embedder, vision_embedder)."""
    device = detect_device()
    hw     = get_hardware_config()
    print(f"🖥️  Device: {device} ({hw.device_name})")

    print("⚙️  Loading models (this may take 1-2 minutes on first run)...")

    router_config = RouterConfig(text_threshold=0.5, min_confidence=0.4)
    print("  [1/4] Vision Router (Microsoft DiT)...")
    router = VisionRouter(config=router_config)

    print("  [2/4] Text Embedding Path (PaddleOCR + nomic-embed-text)...")
    text_path = TextEmbeddingPath()

    print("  [3/4] Vision Embedding Path (ColPali)...")
    vision_path = VisionEmbeddingPath()
    vision_path.embedder._load_model()   # pre-load to avoid cold-start

    print("  [4/4] Text Query Embedder (nomic-embed-text for text DB)...")
    text_query_embedder = TextEmbedder()

    # ── Warmup ────────────────────────────────────────────────────────────────
    print("  Warming up models...", end=" ", flush=True)
    from PIL import Image as _Image
    dummy_img = _Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
    try:
        vision_path.embedder.embed(dummy_img)
        text_query_embedder.embed("warmup")
        # Warmup ColPali text query encoder too
        vision_path.embedder.encode_text_query("warmup")
    except Exception:
        pass
    print("✓\n")

    # vision_embedder is exposed separately so the benchmark can use
    # encode_text_query() for cross-modal retrieval against vision_db
    return router, text_path, vision_path, text_query_embedder, vision_path.embedder


# ── Indexing ──────────────────────────────────────────────────────────────────

def index_corpus(
    pages: list,
    router: VisionRouter,
    text_path: TextEmbeddingPath,
    vision_path: VisionEmbeddingPath,
    text_db: LanceDBBackend,
    vision_db: LanceDBBackend,
    batch_size: int = 4,
) -> dict:
    """
    Run corpus pages through the adaptive pipeline and store in two separate
    LanceDB tables (text_db: 768-dim, vision_db: 128-dim).

    Returns stats dict with router_split and timing info.
    """
    total = len(pages)
    print(f"📚 Indexing {total} corpus pages into vector store...")

    # ── Batch classify all pages ──────────────────────────────────────────────
    print("  Step 1/3: Batch classification...", end=" ", flush=True)
    t0 = time.perf_counter()

    page_objs = []
    for (doc_id, page_num, pil_img) in pages:
        if hasattr(pil_img, "convert"):
            arr = np.array(pil_img.convert("RGB"))
        else:
            arr = np.array(pil_img)
        page_objs.append(Page.from_array(arr, page_number=page_num, source_document=doc_id))

    classifications = router.classify_batch(page_objs)
    classify_ms = (time.perf_counter() - t0) * 1000
    print(f"✓  ({classify_ms:.0f} ms total, {classify_ms/total:.1f} ms/page)")

    # ── Split by modality ─────────────────────────────────────────────────────
    text_items   = [(i, page_objs[i], classifications[i]) for i in range(total) if classifications[i].modality == "text-heavy"]
    visual_items = [(i, page_objs[i], classifications[i]) for i in range(total) if classifications[i].modality == "visual-critical"]

    n_text   = len(text_items)
    n_visual = len(visual_items)
    print(f"  Step 2/3: Routing — Text: {n_text} ({n_text/total*100:.0f}%)  Visual: {n_visual} ({n_visual/total*100:.0f}%)")
    print(f"            Text DB (768-dim nomic), Vision DB (128-dim ColPali)")

    # ── Embed & store ─────────────────────────────────────────────────────────
    print(f"  Step 3/3: Embedding + storing (batch_size={batch_size})...")

    indexed = 0
    embed_errors = 0
    t_embed_start = time.perf_counter()

    def _store_result(emb_result, page_obj: Page, classification, target_db):
        nonlocal indexed
        meta = DocumentMetadata(
            doc_id       = page_obj.source_document,
            page_number  = page_obj.page_number,
            modality     = classification.modality,
            source_file  = page_obj.source_document,
            processed_at = datetime.now(),
            model_name   = emb_result.model_name,
            embedding_dim= int(emb_result.vector.shape[0]),
        )
        target_db.insert(emb_result.vector, meta)
        indexed += 1

    # Text path batches → text_db
    for batch_start in range(0, n_text, batch_size):
        batch = text_items[batch_start:batch_start + batch_size]
        images = [item[1].image for item in batch]
        try:
            results = text_path.process_batch(images)
            for j, res in enumerate(results):
                _, page_obj, cls = batch[j]
                _store_result(res, page_obj, cls, text_db)
        except Exception as e:
            for _, page_obj, cls in batch:
                embed_errors += 1
                logger.warning(f"Text embed failed for {page_obj.source_document}: {e}")
        pct = min((batch_start + batch_size) / n_text, 1.0) * 100 if n_text else 100
        print(f"\r    Text  [{pct:5.1f}%] {min(batch_start+batch_size, n_text)}/{n_text}", end="", flush=True)

    if n_text:
        print()

    # Vision path batches → vision_db (128-dim ColPali vectors)
    for batch_start in range(0, n_visual, batch_size):
        batch = visual_items[batch_start:batch_start + batch_size]
        images = [item[1].image for item in batch]
        try:
            results = vision_path.process_batch(images)
            for j, res in enumerate(results):
                _, page_obj, cls = batch[j]
                _store_result(res, page_obj, cls, vision_db)
        except Exception as e:
            for _, page_obj, cls in batch:
                embed_errors += 1
                logger.warning(f"Vision embed failed for {page_obj.source_document}: {e}")
        pct = min((batch_start + batch_size) / n_visual, 1.0) * 100 if n_visual else 100
        print(f"\r    Vision [{pct:5.1f}%] {min(batch_start+batch_size, n_visual)}/{n_visual}", end="", flush=True)

    if n_visual:
        print()

    embed_s = time.perf_counter() - t_embed_start
    print(f"\n  ✓ Indexed {indexed} pages in {embed_s:.1f}s  ({embed_s/total:.2f}s/page avg)")
    if embed_errors:
        print(f"  ⚠ {embed_errors} pages failed to embed (skipped)")

    return {
        "total_pages":   total,
        "indexed_pages": indexed,
        "embed_errors":  embed_errors,
        "text_pages":    n_text,
        "visual_pages":  n_visual,
        "embed_time_s":  embed_s,
    }


# ── Retrieval evaluation ──────────────────────────────────────────────────────

def run_retrieval_eval(
    queries: list,
    qrels: dict,
    text_query_embedder: TextEmbedder,
    vision_query_embedder,          # VisionEmbedder with encode_text_query()
    text_db: LanceDBBackend,
    vision_db: LanceDBBackend,
    top_k: int = 10,
) -> dict:
    """
    For each query: encode → search both DBs → merge → evaluate.

    text_db   holds 768-dim nomic embeddings (text-heavy pages).
    vision_db holds 128-dim ColPali embeddings (visual-critical pages).

    Correct cross-modal retrieval:
      - text_db   → encode query with nomic-embed-text (768-dim)
      - vision_db → encode query with ColPali process_queries() (128-dim)
        This puts the query in the SAME embedding space as the indexed images.
    """
    print(f"\n🔍 Running retrieval for {len(queries)} queries (top_k={top_k})...")

    # Only evaluate queries that have ground-truth labels
    eval_queries = [(q_id, q_txt) for q_id, q_txt in queries if q_id in qrels]
    skipped = len(queries) - len(eval_queries)
    if skipped:
        print(f"  ⚠ Skipping {skipped} queries with no ground-truth labels")

    if not eval_queries:
        print("  ❌ No evaluable queries found — check dataset qrels structure")
        return {}

    has_text_db   = text_db is not None and text_db.count() > 0
    has_vision_db = vision_db is not None and vision_db.count() > 0
    print(f"  Text DB  : {text_db.count() if has_text_db else 0} pages  (nomic 768-dim search)")
    print(f"  Vision DB: {vision_db.count() if has_vision_db else 0} pages  (ColPali 128-dim search)")

    predictions_list = []
    ground_truth_list = []
    per_query = []

    t0 = time.perf_counter()
    for i, (q_id, q_txt) in enumerate(eval_queries):
        print(f"\r    Query {i+1}/{len(eval_queries)}", end="", flush=True)
        try:
            r_text = []
            r_vision = []

            # ── Text DB: search with nomic text embeddings (same space) ────────
            if has_text_db:
                try:
                    q_vec_text = text_query_embedder.embed(q_txt)  # (768,)
                    r_text = text_db.search(q_vec_text, top_k=top_k)
                except Exception as e:
                    logger.debug(f"text_db search failed for query {q_id}: {e}")

            # ── Vision DB: use ColPali text query encoder (SAME 128-dim space) ──
            if has_vision_db:
                try:
                    q_vec_colpali = vision_query_embedder.encode_text_query(q_txt)  # (128,)
                    r_vision = vision_db.search(q_vec_colpali, top_k=top_k)
                except Exception as e:
                    logger.debug(f"vision_db search failed for query {q_id}: {e}")

            # ── Merge results using Reciprocal Rank Fusion (RRF) ───────────────
            # RRF normalizes scores across different DBs purely based on Rank
            rrf_k = 60
            rrf_scores: dict[str, float] = {}

            # Process text results (already sorted from best to worst by LanceDB)
            for rank, r in enumerate(r_text, start=1):
                did = r.metadata.doc_id
                rrf_scores[did] = rrf_scores.get(did, 0.0) + (1.0 / (rrf_k + rank))

            # Process vision results
            for rank, r in enumerate(r_vision, start=1):
                did = r.metadata.doc_id
                rrf_scores[did] = rrf_scores.get(did, 0.0) + (1.0 / (rrf_k + rank))

            # Sort by RRF score descending
            ranked_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
            ranked_doc_ids = [did for did, score in ranked_docs][:top_k]

        except Exception as e:
            logger.warning(f"Query '{q_txt[:40]}' failed: {e}")
            ranked_doc_ids = []

        gt = qrels[q_id]
        predictions_list.append(ranked_doc_ids)
        ground_truth_list.append(gt)
        per_query.append({
            "query_id":     q_id,
            "query":        q_txt[:80],
            "ground_truth": list(gt),
            "retrieved":    ranked_doc_ids,
        })

    retrieval_s = time.perf_counter() - t0
    print(f"\n  ✓ Retrieval complete in {retrieval_s:.1f}s ({retrieval_s/len(eval_queries)*1000:.0f}ms/query avg)")

    # Compute aggregate metrics
    metrics = evaluate_retrieval(
        predictions_list=predictions_list,
        ground_truth_list=ground_truth_list,
        k_values=[1, 5, 10],
    )

    return {
        "num_queries":    len(eval_queries),
        "retrieval_s":    retrieval_s,
        "metrics":        metrics,
        "per_query":      per_query,
    }


# ── Report generation ─────────────────────────────────────────────────────────

def print_report(index_stats: dict, retrieval_stats: dict, args):
    """Print a clean summary report."""
    metrics = retrieval_stats.get("metrics")
    if not metrics:
        print("\n❌ No metrics computed.")
        return

    sep = "=" * 65
    print(f"\n{sep}")
    print("  ADAPTIVE RETRIEVAL SYSTEM — ACCURACY BENCHMARK")
    print(f"  Dataset   : REAL-MM-RAG ({args.variant})")
    print(f"  Pages     : {index_stats['indexed_pages']} / {index_stats['total_pages']} indexed")
    print(f"  Queries   : {retrieval_stats['num_queries']}")
    print(f"{sep}")
    print()
    print("  RETRIEVAL ACCURACY METRICS")
    print(f"  {'Recall@1':<15}: {metrics.recall_at_1:.4f}")
    print(f"  {'Recall@5':<15}: {metrics.recall_at_5:.4f}")
    print(f"  {'Recall@10':<15}: {metrics.recall_at_10:.4f}")
    print(f"  {'MRR':<15}: {metrics.mrr:.4f}")
    print(f"  {'NDCG@10':<15}: {metrics.ndcg:.4f}")
    print()
    print("  ROUTING SPLIT")
    n = index_stats['total_pages']
    nt = index_stats['text_pages']
    nv = index_stats['visual_pages']
    print(f"  Text-heavy     : {nt} pages ({nt/n*100:.1f}%)")
    print(f"  Visual-critical: {nv} pages ({nv/n*100:.1f}%)")
    print()
    print("  TIMING")
    et = index_stats['embed_time_s']
    rt = retrieval_stats['retrieval_s']
    print(f"  Indexing time  : {et:.1f}s  ({et/n:.2f}s/page)")
    print(f"  Retrieval time : {rt:.1f}s  ({rt/retrieval_stats['num_queries']*1000:.0f}ms/query)")
    print(f"{sep}\n")


def save_results(index_stats: dict, retrieval_stats: dict, args, output_dir: Path):
    """Save full JSON results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"accuracy_benchmark_{args.variant}_{ts}.json"

    metrics = retrieval_stats.get("metrics")
    payload = {
        "metadata": {
            "timestamp":  datetime.now().isoformat(),
            "dataset":    f"REAL-MM-RAG_{args.variant}",
            "max_pages":  args.max_pages,
            "max_queries":args.max_queries,
        },
        "index_stats": index_stats,
        "retrieval_stats": {
            "num_queries":  retrieval_stats.get("num_queries"),
            "retrieval_s":  retrieval_stats.get("retrieval_s"),
        },
        "metrics": {
            "recall_at_1":  float(metrics.recall_at_1)  if metrics else None,
            "recall_at_5":  float(metrics.recall_at_5)  if metrics else None,
            "recall_at_10": float(metrics.recall_at_10) if metrics else None,
            "mrr":          float(metrics.mrr)           if metrics else None,
            "ndcg_at_10":   float(metrics.ndcg)          if metrics else None,
        },
        "per_query_results": retrieval_stats.get("per_query", []),
    }

    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"💾 Results saved to: {out_path}")
    return out_path


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="REAL-MM-RAG Accuracy Benchmark for Adaptive Retrieval System"
    )
    parser.add_argument("--variant",      default="techreport",
                        choices=["techreport", "techslides"],
                        help="REAL-MM-RAG dataset variant (default: techreport)")
    parser.add_argument("--max-pages",    type=int, default=None,
                        help="Max corpus pages to index (default: all)")
    parser.add_argument("--max-queries",  type=int, default=None,
                        help="Max queries to evaluate (default: all)")
    parser.add_argument("--top-k",        type=int, default=10,
                        help="Top-K results for retrieval (default: 10)")
    parser.add_argument("--batch-size",   type=int, default=4,
                        help="Embedding batch size (default: 4)")
    parser.add_argument("--output-dir",   default="outputs/benchmark_results",
                        help="Where to save results JSON")
    parser.add_argument("--db-path",      default="./data/accuracy_benchmark_db",
                        help="Temp LanceDB path (wiped each run)")
    parser.add_argument("--cache-dir",    default=None,
                        help="HuggingFace dataset cache dir (optional)")
    args = parser.parse_args()

    print("\n" + "=" * 65)
    print("  REAL-MM-RAG ACCURACY BENCHMARK")
    print(f"  Variant   : {args.variant}")
    print(f"  Max pages : {args.max_pages or 'all'}")
    print(f"  Max queries: {args.max_queries or 'all'}")
    print("=" * 65)

    # ── 1. Load dataset ───────────────────────────────────────────────────────
    pages, queries, qrels = load_real_mm_rag(
        variant=args.variant,
        max_pages=args.max_pages,
        max_queries=args.max_queries,
        cache_dir=args.cache_dir,
    )

    if not pages:
        print("❌ No pages loaded. Exiting.")
        sys.exit(1)
    if not queries:
        print("❌ No queries loaded. Exiting.")
        sys.exit(1)
    if not qrels:
        print("⚠  No qrels loaded — metrics will be zero. Check dataset structure.")

    # ── 2. Init models ────────────────────────────────────────────────────────
    router, text_path, vision_path, text_query_embedder, vision_embedder = init_pipeline()

    # ── 3. Fresh LanceDB instances (wiped each run for clean eval) ────────────
    db_path = Path(args.db_path)
    if db_path.exists():
        shutil.rmtree(db_path)   # clean slate every run

    text_db = LanceDBBackend(
        table_name="text_pages",
        db_path=str(db_path),
        distance_metric="cosine",
    )
    vision_db = LanceDBBackend(
        table_name="vision_pages",
        db_path=str(db_path),
        distance_metric="cosine",
    )

    # ── 4. Index corpus ───────────────────────────────────────────────────────
    index_stats = index_corpus(
        pages=pages,
        router=router,
        text_path=text_path,
        vision_path=vision_path,
        text_db=text_db,
        vision_db=vision_db,
        batch_size=args.batch_size,
    )

    if index_stats["indexed_pages"] == 0:
        print("❌ No pages indexed. Cannot run retrieval. Exiting.")
        sys.exit(1)

    # ── 5. Retrieval evaluation ───────────────────────────────────────────────
    retrieval_stats = run_retrieval_eval(
        queries=queries,
        qrels=qrels,
        text_query_embedder=text_query_embedder,
        vision_query_embedder=vision_embedder,
        text_db=text_db,
        vision_db=vision_db,
        top_k=args.top_k,
    )

    # ── 6. Report + save ──────────────────────────────────────────────────────
    print_report(index_stats, retrieval_stats, args)
    output_dir = Path(args.output_dir)
    save_results(index_stats, retrieval_stats, args, output_dir)

    # Clean up temp DB
    if db_path.exists():
        shutil.rmtree(db_path)

    print("✅ Benchmark complete!\n")


if __name__ == "__main__":
    main()
