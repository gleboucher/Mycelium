#!/usr/bin/env python3
"""
Construit source_to_pdf.json pour un dossier article : mappe les chaînes de
« source » présentes dans articles/<slug>/article.json vers les PDF du même dossier.

Usage : python build_source_mapping.py --article Beauty_algorithm
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

try:
    from sentence_transformers import SentenceTransformer

    HAS_EMBEDDINGS = True
except ImportError:
    HAS_EMBEDDINGS = False

PROJECT_ROOT = Path(__file__).resolve().parent


def collect_sources_from_article(data: dict) -> set[str]:
    sources: set[str] = set()
    for fig in data.get("figures", []):
        g = fig.get("graph")
        if g:
            if g.get("source"):
                sources.add(str(g["source"]).strip())
            for s in g.get("series") or []:
                if s.get("source"):
                    sources.add(str(s["source"]).strip())
        c = fig.get("chart")
        if c and c.get("source"):
            sources.add(str(c["source"]).strip())
    return sources


def list_pdfs(dir_path: Path) -> list[tuple[str, str]]:
    if not dir_path.is_dir():
        return []
    result = []
    for f in sorted(dir_path.iterdir()):
        if f.suffix.lower() == ".pdf":
            searchable = f.stem.replace("_", " ").replace("-", " ")
            result.append((f.name, searchable))
    return result


def _match_with_embeddings(sources: set[str], pdfs: list[tuple[str, str]]) -> dict[str, str]:
    import numpy as np

    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    source_list = list(sources)
    pdf_names = [p[0] for p in pdfs]
    pdf_texts = [p[1] for p in pdfs]
    source_emb = model.encode(source_list, normalize_embeddings=True)
    pdf_emb = model.encode(pdf_texts, normalize_embeddings=True)
    sim = np.dot(source_emb, pdf_emb.T)
    mapping = {}
    for i, src in enumerate(source_list):
        j = int(sim[i].argmax())
        mapping[src] = pdf_names[j]
    return mapping


def _match_with_fallback(sources: set[str], pdfs: list[tuple[str, str]]) -> dict[str, str]:
    mapping = {}
    for src in sources:
        best = None
        best_score = 0
        src_lower = src.lower()
        src_tokens = set(
            t.strip("(),")
            for t in re.split(r"\s+", src_lower.replace("(", " ").replace(")", " "))
            if len(t.strip("(),")) > 2
        )
        for filename, searchable in pdfs:
            search_lower = searchable.lower()
            score = sum(1 for t in src_tokens if t in search_lower)
            if score > best_score:
                best_score = score
                best = filename
        mapping[src] = best or (pdfs[0][0] if pdfs else "")
    return mapping


def main() -> int:
    parser = argparse.ArgumentParser(description="Mappe les sources d'un article vers les PDF locaux.")
    parser.add_argument(
        "--article",
        required=True,
        help="Dossier sous articles/ (ex. Beauty_algorithm).",
    )
    args = parser.parse_args()
    slug = args.article.strip().strip("/")
    article_dir = PROJECT_ROOT / "articles" / slug
    article_json = article_dir / "article.json"
    if not article_json.exists():
        print(f"Fichier manquant : {article_json}")
        return 1
    with article_json.open(encoding="utf-8") as f:
        data = json.load(f)
    sources = collect_sources_from_article(data)
    pdfs = list_pdfs(article_dir)
    if not pdfs:
        print(f"Aucun PDF dans {article_dir}")
        return 1
    if not sources:
        print("Aucune source à mapper dans article.json (figures sans champ source).")
        out = article_dir / "source_to_pdf.json"
        with out.open("w", encoding="utf-8") as f:
            json.dump({}, f, ensure_ascii=False, indent=2)
        print(f"Écrit {out} (vide)")
        return 0
    if HAS_EMBEDDINGS:
        mapping = _match_with_embeddings(sources, pdfs)
    else:
        mapping = _match_with_fallback(sources, pdfs)
    out = article_dir / "source_to_pdf.json"
    with out.open("w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
    print(f"Écrit {out} ({len(mapping)} entrées)")
    for src, pdf in sorted(mapping.items(), key=lambda x: x[0]):
        print(f"  {src!r} -> {pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
