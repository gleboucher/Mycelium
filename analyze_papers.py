"""
Pipeline: load PDFs from articles/<slug>/, then Claude → vulgarisation article (FR)
with metaphors, structured sections, and optional figures (JSON + PNG via matplotlib).
No embeddings / no RAG retrieval.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from datetime import date
from pathlib import Path
from typing import List, Literal, Optional

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

# -----------------------------------------------------------------------------
# Figure schemas (reused by the front-end SVG renderer)
# -----------------------------------------------------------------------------


class DataPoint(BaseModel):
    x: str = Field(..., description="Libellé catégorie ou année.")
    y: float = Field(..., description="Valeur numérique.")


class GraphSeries(BaseModel):
    name: str = Field(..., description="Légende courte (< 40 car.).")
    legend_description: Optional[str] = Field(
        default=None, description="Une ligne : ce que représente la série."
    )
    data_points: List[DataPoint] = Field(..., description="Points ordonnés.")
    source: Optional[str] = Field(default=None, description="Référence courte à l'étude.")


class GraphData(BaseModel):
    title: str
    graph_type: Literal["line", "bar"] = Field(
        ..., description="line = évolution ; bar = comparaison de catégories."
    )
    x_axis_label: Optional[str] = None
    y_axis_label: Optional[str] = None
    series: List[GraphSeries] = Field(..., description="Une ou plusieurs séries.")
    source: Optional[str] = None


class ChartCategory(BaseModel):
    label: str
    value: float


class ChartData(BaseModel):
    title: str
    chart_type: Literal["bar", "pie"] = Field(..., description="bar ou pie.")
    value_label: Optional[str] = None
    categories: List[ChartCategory] = Field(...)
    source: Optional[str] = None


class ArticleFigure(BaseModel):
    """Une figure illustrant des chiffres présents dans les extraits."""

    id: str = Field(..., description="Identifiant stable, ex. fig_prevalence.")
    title: str
    caption: str = Field(..., description="Phrase qui explique la figure au lecteur.")
    figure_type: Literal["graph", "chart"] = Field(
        ..., description="graph = ligne/barres (GraphData) ; chart = bar/pie (ChartData)."
    )
    graph: Optional[GraphData] = None
    chart: Optional[ChartData] = None


class ArticleSection(BaseModel):
    id: str = Field(default="", description="Identifiant court, ex. intro_mecanismes.")
    heading: str = Field(default="", description="Titre de section accessible.")
    paragraphs: List[str] = Field(
        default_factory=list,
        description="Paragraphes de vulgarisation (texte brut, pas de HTML).",
    )
    metaphor_box: Optional[str] = Field(
        default=None,
        description="Encadré 'En image' : une métaphore concrète (2–4 phrases).",
    )


class ArticleVulgarisation(BaseModel):
    """Article de vulgarisation scientifique, entièrement en français."""

    title: str = Field(default="", description="Titre principal, clair et engageant.")
    subtitle: str = Field(default="", description="Sous-titre : angle et promesse pour le lecteur.")
    deck: str = Field(
        default="",
        description="Une ou deux phrases : résumé accrocheur (chapô).",
    )
    reading_time_min: int = Field(default=5, ge=1, le=20, description="Temps de lecture estimé.")
    sections: List[ArticleSection] = Field(
        default_factory=list,
        description="4 à 8 sections structurant l'article.",
    )
    key_takeaways: List[str] = Field(
        default_factory=list,
        description="3 à 7 points clés en une ligne chacun.",
    )
    figures: List[ArticleFigure] = Field(
        default_factory=list,
        description="0 à 4 figures ; uniquement si les extraits contiennent des données exploitables.",
        max_length=4,
    )
    limitations: str = Field(
        default="",
        description="Paragraphe honnête : limites des études, généralisations à éviter.",
    )
    practical_intro: str = Field(
        default="",
        description="Une phrase courte (max ~25 mots) qui introduit les conseils pratiques.",
    )
    practical_tips: List[str] = Field(
        default_factory=list,
        description="4 à 6 actions concrètes (ados, parents, utilisateurs) pour limiter les risques ou mieux prendre conscience ; une ligne par conseil.",
    )
    glossary: List[str] = Field(
        default_factory=list,
        description="Optionnel : 3–8 définitions courtes 'terme : définition'.",
    )


def load_pdfs_from_dir(papers_dir: Path) -> List[Document]:
    docs: List[Document] = []
    for path in sorted(papers_dir.glob("*.pdf")):
        loader = PyPDFLoader(str(path))
        pages = loader.load()
        for i, page in enumerate(pages):
            page.metadata["source_file"] = path.name
            page.metadata["page"] = i + 1
        docs.extend(pages)
    return docs


def build_full_context(documents: List[Document], max_chars: int = 180_000) -> str:
    """Concatenate loaded pages as context (no retrieval, no embeddings)."""
    parts: List[str] = []
    total = 0
    for doc in documents:
        src = doc.metadata.get("source_file", "unknown")
        block = f"[Source fichier: {src}]\n{doc.page_content}"
        if total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block)
    return "\n\n---\n\n".join(parts)


# -----------------------------------------------------------------------------
# Prompts (vulgarisation + métaphores + rigueur)
# -----------------------------------------------------------------------------

ARTICLE_SYSTEM = """Tu es un·e journaliste scientifique et vulgarisateur·rice senior, spécialisé·e
en psychologie, sciences sociales et médias numériques. Tu écris en français clair,
chaleureux et précis, pour un public curieux sans formation scientifique.

Objectif : produire un ARTICLE de vulgarisation à partir UNIQUEMENT des extraits fournis.

PUBLIC CIBLE PRIORITAIRE :
- Adolescent·es (environ 13-18 ans), jeunes adultes et enseignants.
- Style relativement simple, concret, vulgarisé mais ne simplifiant pas les idées.
- Eviter le jargon ; si un terme technique est utile, l'expliquer en une phrase.
- Ton direct, clair, jamais infantilisant.

Principes éditoriaux :
- Ne jamais inventer de chiffres, pourcentages, tailles d'échantillon ou conclusions absentes des extraits.
- Si une information manque, l'indiquer avec prudence ou s'en abstenir.
- Expliquer les idées avec des analogies et métaphores concrètes (quotidien, nature, sport, cuisine, cartographie…)
  sans infantiliser : une métaphore par section au plus dans le champ metaphor_box.
- Varier le rythme : phrases courtes, exemples, reformulations.
- Citer implicitement les sources via les noms de fichiers [Source fichier: …] quand tu t'appuies sur un résultat précis.

Figures (graphiques) :
- Proposer au plus 4 figures, seulement si des données chiffrées comparables ou des séries temporelles
  apparaissent clairement dans les extraits.
- Chaque point ou barre doit correspondre à une donnée explicitement présente dans le texte.
- Si les données sont trop floues pour un graphique fiable, laisser la liste figures vide.
- Si une figure est utile pedagogiquement, preferer un schema tres simple (barres/categories courtes)
  avant une visualisation complexe.

LONGUEUR ET CLARTE (obligatoire) :
- Article court : 1000 a 1500 mots au total environ.
- Titre tres court (6 a 10 mots), clair et impactant. 
- Sous-titre court (max 14 mots).
- 3 a 5 sections maximum.
- 1 a 2 paragraphes par section, 2 a 4 phrases par paragraphe.
- "A retenir" : 3 a 5 points, une ligne chacun.

CONSEILS PRATIQUES (obligatoire) :
- Remplir practical_intro : une phrase tres courte qui annonce la suite (ton bienveillant, pas moralisateur).
- Remplir practical_tips : 4 a 6 conseils concrets, realisables au quotidien, formules en style direct (tu peux / evite de / parle a / regle ton / etc.).
- S'appuyer sur ce que les extraits suggerent (temps d'ecran, comparaison sociale, signalement, aide, hygiene numerique) ; ne pas inventer de solution miracle ni de promesse medicale.
- Si les etudes ne donnent pas de piste explicite, proposer des gestes prudents et generiques coherents avec le sujet.

Structure imposee par le schema JSON (sections, points cles, limites, practical_intro + practical_tips, glossaire optionnel)."""


ARTICLE_HUMAN = """Contexte dossier (slug) : {article_slug}
Thème / consigne optionnelle pour cibler l'angle : {topic_hint}

Extraits des PDF (ne pas dépasser ce qui est écrit) :

{retrieved_context}

Rédige l'article structuré selon le schéma. Langue : français. Ton : accessible, rigoureux, bienveillant."""


def build_article_chain():
    load_dotenv()
    api_key = os.getenv("API_KEY")
    if not api_key:
        raise RuntimeError(
            "API_KEY manquant. Définissez la clé API Anthropic (ou .env) avant d'exécuter le script."
        )
    llm = ChatAnthropic(
        model="claude-opus-4-5",
        temperature=0.2,
        api_key=api_key,
    )
    structured = llm.with_structured_output(ArticleVulgarisation)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", ARTICLE_SYSTEM),
            ("human", ARTICLE_HUMAN),
        ]
    )
    return prompt | structured


def _slug_to_title(slug: str) -> str:
    return slug.replace("_", " ").replace("-", " ").strip().title()


def sanitize_article(article: ArticleVulgarisation, slug: str) -> ArticleVulgarisation:
    """Fill missing fields defensively so JSON generation never fails downstream."""
    if not article.title.strip():
        article.title = _slug_to_title(slug)
    if not article.subtitle.strip():
        article.subtitle = "Comprendre les résultats scientifiques en langage clair"
    if not article.deck.strip():
        article.deck = "Synthese vulgarisee des publications du dossier."
    # Keep titles short and punchy for teen audience.
    article.title = " ".join(article.title.split()[:8]).strip()
    article.subtitle = " ".join(article.subtitle.split()[:14]).strip()
    if not article.title:
        article.title = _slug_to_title(slug)
    if not article.subtitle:
        article.subtitle = "Comprendre vite l'essentiel"
    article.reading_time_min = max(1, min(int(article.reading_time_min or 5), 12))
    if not article.sections:
        article.sections = [
            ArticleSection(
                id="intro",
                heading="Introduction",
                paragraphs=["Les documents ont ete analyses pour produire cette synthese."],
            )
        ]
    # Keep output concise: max 5 sections.
    article.sections = article.sections[:5]
    for i, sec in enumerate(article.sections):
        if not sec.id.strip():
            sec.id = f"section_{i+1}"
        if not sec.heading.strip():
            sec.heading = f"Section {i+1}"
        if not sec.paragraphs:
            sec.paragraphs = ["Contenu indisponible pour cette section."]
        # Keep section content compact and easy to scan.
        sec.paragraphs = sec.paragraphs[:2]
    if not article.key_takeaways:
        article.key_takeaways = ["Les resultats doivent etre interpretes avec prudence."]
    article.key_takeaways = article.key_takeaways[:5]
    if not article.limitations.strip():
        article.limitations = "Certaines informations sont partielles et dependent des etudes disponibles."
    if not article.practical_intro.strip():
        article.practical_intro = "Quelques pistes simples pour mieux te proteger au quotidien."
    article.practical_intro = " ".join(article.practical_intro.split()[:25]).strip()
    if not article.practical_tips:
        article.practical_tips = [
            "Faire des pauses ecran et noter comment tu te sens avant et apres les reseaux.",
            "Limiter les applis ou les comptes qui te poussent a te comparer en permanence.",
            "Parler a une personne de confiance ou a un professionnel si ca pese sur ton moral.",
        ]
    article.practical_tips = article.practical_tips[:6]
    return article


def slug_to_safe_filename(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", s).strip("_") or "figure"


def export_figures_matplotlib(
    article: ArticleVulgarisation,
    figures_dir: Path,
) -> dict[str, str]:
    """Écrit des PNG pour chaque figure ; retourne id -> chemin relatif depuis article.json."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib non installé : figures PNG ignorées (JSON inchangé).")
        return {}

    figures_dir.mkdir(parents=True, exist_ok=True)
    rel_paths: dict[str, str] = {}
    saved_index = 0

    for fig in article.figures:
        plt.figure(figsize=(7, 4.2), dpi=120)
        ax = plt.gca()

        if fig.figure_type == "graph" and not fig.graph:
            plt.close()
            continue
        if fig.figure_type == "chart" and not fig.chart:
            plt.close()
            continue

        saved_index += 1
        fname = f"{saved_index:02d}_{slug_to_safe_filename(fig.id)}.png"
        out_path = figures_dir / fname

        if fig.figure_type == "graph" and fig.graph:
            gd = fig.graph
            if gd.graph_type == "line" and gd.series:
                for s in gd.series:
                    xs = [p.x for p in s.data_points]
                    ys = [p.y for p in s.data_points]
                    ax.plot(xs, ys, marker="o", label=s.name)
                ax.legend(loc="best", fontsize=8)
            elif gd.graph_type == "bar" and gd.series:
                s0 = gd.series[0]
                labels = [p.x for p in s0.data_points]
                vals = [p.y for p in s0.data_points]
                ax.bar(labels, vals, color="#3d6b47")
                plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
                if len(gd.series) > 1:
                    # Plusieurs séries : barres groupées simplifiées
                    ax.clear()
                    x = range(len(labels))
                    width = 0.8 / len(gd.series)
                    for si, series in enumerate(gd.series):
                        vals_si = [p.y for p in series.data_points]
                        ax.bar(
                            [xi + si * width for xi in x],
                            vals_si,
                            width=width * 0.9,
                            label=series.name,
                        )
                    ax.set_xticks([xi + width * (len(gd.series) - 1) / 2 for xi in x])
                    ax.set_xticklabels(labels, rotation=25, ha="right")
                    ax.legend(loc="best", fontsize=8)
            if gd.x_axis_label:
                ax.set_xlabel(gd.x_axis_label)
            if gd.y_axis_label:
                ax.set_ylabel(gd.y_axis_label)
            ax.set_title(gd.title, fontsize=11)
        elif fig.figure_type == "chart" and fig.chart:
            cd = fig.chart
            labels = [c.label for c in cd.categories]
            vals = [c.value for c in cd.categories]
            if cd.chart_type == "pie":
                ax.pie(vals, labels=labels, autopct="%1.0f%%", textprops={"fontsize": 8})
                ax.set_title(cd.title, fontsize=11)
            else:
                ax.bar(labels, vals, color="#c17f3a")
                ax.set_title(cd.title, fontsize=11)
                if cd.value_label:
                    ax.set_ylabel(cd.value_label)
                plt.setp(ax.get_xticklabels(), rotation=25, ha="right")

        plt.tight_layout()
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
        rel_paths[fig.id] = f"figures/{fname}"

    return rel_paths


def _match_sources_to_pdfs(sources: set[str], pdf_filenames: List[str]) -> dict[str, str]:
    """Associe chaque libellé de source aux PDF du dossier (tokens communs)."""
    if not pdf_filenames:
        return {s: "" for s in sources}
    mapping: dict[str, str] = {}
    for src in sources:
        best = pdf_filenames[0]
        best_score = -1
        src_lower = src.lower()
        tokens = [
            t.strip("(),")
            for t in re.split(r"\s+", src_lower.replace("(", " ").replace(")", " "))
            if len(t.strip("(),")) > 2
        ]
        for fn in pdf_filenames:
            stem = Path(fn).stem.lower().replace("_", " ").replace("-", " ")
            score = sum(1 for t in tokens if t in stem or t in fn.lower())
            if score > best_score:
                best_score = score
                best = fn
        mapping[src] = best
    return mapping


def write_article_source_mapping(
    article_dir: Path,
    article: ArticleVulgarisation,
    pdf_filenames: List[str],
) -> None:
    sources: set[str] = set()
    for fig in article.figures:
        if fig.graph:
            if fig.graph.source:
                sources.add(fig.graph.source.strip())
            for s in fig.graph.series:
                if s.source:
                    sources.add(s.source.strip())
        if fig.chart and fig.chart.source:
            sources.add(fig.chart.source.strip())
    if not sources:
        return
    mapping = _match_sources_to_pdfs(sources, pdf_filenames)
    out = article_dir / "source_to_pdf.json"
    with out.open("w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)


def update_manifest(
    project_root: Path,
    slug: str,
    title: str,
    article_json_rel: str,
) -> None:
    manifest_path = project_root / "articles" / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    data = {"articles": []}
    if manifest_path.exists():
        with manifest_path.open(encoding="utf-8") as f:
            data = json.load(f)
    articles = data.get("articles", [])
    articles = [a for a in articles if a.get("slug") != slug]
    articles.append(
        {
            "slug": slug,
            "title": title,
            "path": article_json_rel,
            "updated": date.today().isoformat(),
        }
    )
    articles.sort(key=lambda x: x["slug"].lower())
    data["articles"] = articles
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Genere des articles de vulgarisation a partir des PDF dans articles/<slug>/."
    )
    p.add_argument(
        "--article",
        default="",
        help="Slug d'un dossier a traiter (ex. adore_detester). Si absent, traite tous les dossiers dans articles/.",
    )
    p.add_argument(
        "--topic",
        default="",
        help="Consigne courte pour orienter l'angle de redaction.",
    )
    p.add_argument(
        "--skip-png",
        action="store_true",
        help="Ne pas générer les PNG matplotlib (uniquement JSON).",
    )
    p.add_argument(
        "--max-context-chars",
        type=int,
        default=180000,
        help="Taille max du contexte envoye au modele.",
    )
    return p.parse_args()


def generate_article_for_slug(project_root: Path, slug: str, args: argparse.Namespace) -> None:
    project_root = Path(__file__).resolve().parent
    article_dir = project_root / "articles" / slug
    if not article_dir.is_dir():
        raise FileNotFoundError(f"Dossier article introuvable : {article_dir}")
    pdf_paths = list(article_dir.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"Aucun PDF dans {article_dir}")

    print(f"Chargement de {len(pdf_paths)} PDF(s) depuis {article_dir} …")
    documents = load_pdfs_from_dir(article_dir)
    print(f"  → {len(documents)} pages.")

    print("Construction du contexte complet (sans RAG)…")
    retrieved = build_full_context(documents, max_chars=args.max_context_chars)
    print(f"  → {len(retrieved)} caracteres d'extraits.")

    print("Appel modèle (article structuré)…")
    chain = build_article_chain()
    article: ArticleVulgarisation = chain.invoke(
        {
            "article_slug": slug,
            "topic_hint": args.topic or "(aucune — déduire du contenu des PDF)",
            "retrieved_context": retrieved,
        }
    )
    article = sanitize_article(article, slug)

    figure_png_paths: dict[str, str] = {}
    if not args.skip_png and article.figures:
        fig_dir = article_dir / "figures"
        figure_png_paths = export_figures_matplotlib(article, fig_dir)

    payload = {
        "slug": slug,
        "generated_at": date.today().isoformat(),
        "title": article.title,
        "subtitle": article.subtitle,
        "deck": article.deck,
        "reading_time_min": article.reading_time_min,
        "sections": [s.model_dump() for s in article.sections],
        "key_takeaways": article.key_takeaways,
        "figures": [],
        "limitations": article.limitations,
        "practical_intro": article.practical_intro,
        "practical_tips": article.practical_tips,
        "glossary": article.glossary,
        "sources_index": sorted({p.name for p in pdf_paths}),
    }

    for fig in article.figures:
        fd = fig.model_dump()
        png = figure_png_paths.get(fig.id)
        if png:
            fd["png_path"] = png
        payload["figures"].append(fd)

    write_article_source_mapping(article_dir, article, sorted(p.name for p in pdf_paths))

    out_json = article_dir / "article.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    update_manifest(project_root, slug, article.title, f"articles/{slug}/article.json")
    print(f"\nArticle enregistré : {out_json}")
    print(f"Manifest mis à jour : articles/manifest.json")
    if figure_png_paths:
        print(f"Figures PNG : {article_dir / 'figures'}")


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    articles_root = project_root / "articles"
    if not articles_root.exists():
        raise FileNotFoundError(f"Dossier introuvable: {articles_root}")

    if args.article.strip():
        generate_article_for_slug(project_root, args.article.strip().strip("/"), args)
        return

    slugs = sorted(
        p.name
        for p in articles_root.iterdir()
        if p.is_dir() and list(p.glob("*.pdf"))
    )
    if not slugs:
        raise FileNotFoundError("Aucun dossier d'article avec PDF dans articles/.")

    for slug in slugs:
        print(f"\n=== Traitement: {slug} ===")
        generate_article_for_slug(project_root, slug, args)


if __name__ == "__main__":
    main()
