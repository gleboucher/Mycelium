/**
 * Mycélium — Articles de vulgarisation scientifique
 * - Sans ?article= : affiche la liste (articles/manifest.json)
 * - Avec ?article=slug : charge articles/<slug>/article.json et affiche le contenu + figures
 */

const COLORS = ['amber', 'forest', 'terracotta'];

let sourceToPdfMap = {};
let currentArticleSlug = '';

function resolvePdfFilename(source) {
  if (!source || typeof source !== 'string') return null;
  const s = source.trim();
  if (s.endsWith('.pdf')) return s;
  return sourceToPdfMap[s] || null;
}

function buildSourceUrl(pdfFilename, title, sourceCitation) {
  const params = new URLSearchParams();
  if (currentArticleSlug) params.set('article', currentArticleSlug);
  if (pdfFilename) params.set('pdf', pdfFilename);
  if (title) params.set('title', title);
  if (sourceCitation) params.set('source', sourceCitation);
  return 'source.html' + (params.toString() ? '?' + params.toString() : '');
}

function escapeHtml(s) {
  if (s == null || s === '') return '';
  const div = document.createElement('div');
  div.textContent = s;
  return div.innerHTML;
}

function computeYTicks(yMin, yMax, maxTicks = 5) {
  const range = yMax - yMin || 1;
  const roughStep = range / (maxTicks - 1);
  const mag = Math.pow(10, Math.floor(Math.log10(roughStep)));
  const norm = roughStep / mag;
  const step = mag * (norm <= 1 ? 1 : norm <= 2 ? 2 : norm <= 5 ? 5 : 10);
  const start = Math.floor(yMin / step) * step;
  const ticks = [];
  for (let v = start; v <= yMax + step * 0.5; v += step) {
    if (v >= yMin - 1e-9) ticks.push(v);
  }
  if (ticks.length < 2) ticks.push(yMin, yMax);
  return ticks;
}

function renderBarChart(chart, subjectSource, opts = {}) {
  const showInnerTitle = !opts.suppressInnerTitle;
  if (!chart || !chart.categories || !chart.categories.length) return '';
  const categories = chart.categories;
  const values = categories.map((c) => c.value);
  const maxVal = Math.max(...values, 1);
  const yMin = 0;
  const yMax = Math.ceil(maxVal * 1.1 / 20) * 20 || 100;
  const yRange = yMax - yMin || 1;
  const yTicks = computeYTicks(yMin, yMax);

  const padding = { top: 24, right: 24, bottom: 52, left: 48 };
  const width = 420;
  const height = 260;
  const chartLeft = padding.left;
  const chartRight = width - padding.right;
  const chartTop = padding.top;
  const chartBottom = height - padding.bottom;
  const chartWidth = chartRight - chartLeft;
  const chartHeight = chartBottom - chartTop;

  const yScale = (v) => chartBottom - ((v - yMin) / yRange) * chartHeight;
  const n = categories.length;
  const slotWidth = chartWidth / Math.max(n, 1);
  const barWidth = slotWidth * 0.7;
  const xForBar = (i) => chartLeft + (i + 0.5) * slotWidth;

  const yAxisLine = `<line class="axis-line" x1="${chartLeft}" y1="${chartTop}" x2="${chartLeft}" y2="${chartBottom}" />`;
  const xAxisLine = `<line class="axis-line" x1="${chartLeft}" y1="${chartBottom}" x2="${chartRight}" y2="${chartBottom}" />`;

  const yTickEls = yTicks
    .map((v) => {
      const y = yScale(v);
      const label = Number.isInteger(v) ? v : v.toFixed(1);
      return `
        <line class="axis-tick" x1="${chartLeft}" y1="${y}" x2="${chartLeft - 6}" y2="${y}" />
        <text class="axis-tick-label" x="${chartLeft - 8}" y="${y}" text-anchor="end" dominant-baseline="middle">${escapeHtml(String(label))}</text>
      `;
    })
    .join('');

  const xTickEls = categories
    .map((cat, i) => {
      const x = xForBar(i);
      const short = String(cat.label).length > 14 ? String(cat.label).slice(0, 12) + '…' : String(cat.label);
      return `
        <line class="axis-tick" x1="${x}" y1="${chartBottom}" x2="${x}" y2="${chartBottom + 6}" />
        <text class="axis-tick-label axis-tick-label-x" x="${x}" y="${chartBottom + 22}" text-anchor="middle">${escapeHtml(short)}</text>
      `;
    })
    .join('');

  const yAxisTitle = chart.value_label
    ? `<text class="axis-title" x="${chartLeft - 26}" y="${(chartTop + chartBottom) / 2}" text-anchor="middle" transform="rotate(-90, ${chartLeft - 26}, ${(chartTop + chartBottom) / 2})">${escapeHtml(chart.value_label)}</text>`
    : '';
  const xAxisTitle = `<text class="axis-title axis-title-x" x="${(chartLeft + chartRight) / 2}" y="${height - 8}" text-anchor="middle">Catégorie</text>`;

  const bars = categories
    .map((cat, i) => {
      const cx = xForBar(i);
      const x = cx - barWidth / 2;
      const barH = ((cat.value - yMin) / yRange) * chartHeight;
      const y = chartBottom - barH;
      const color = COLORS[i % COLORS.length];
      return `<rect class="bar-fill-svg ${color}" x="${x}" y="${y}" width="${barWidth}" height="${Math.max(barH, 0)}" rx="4" />`;
    })
    .join('');

  const rawSource = subjectSource || chart.source || (chart.categories[0] && chart.categories[0].source);
  const pdfFile = resolvePdfFilename(rawSource);
  const sourceUrl = pdfFile ? buildSourceUrl(pdfFile || rawSource, chart.title, rawSource) : '';

  const linkOpen = sourceUrl ? `<a href="${sourceUrl}" class="viz-source-link">Voir la publication</a>` : '';

  const innerTitle = showInnerTitle ? `<h4 class="article-viz-title">${escapeHtml(chart.title)}</h4>` : '';

  return `
    <div class="article-viz-inner">
      ${innerTitle}
      <div class="line-graph-wrap">
        <div class="line-graph">
          <svg class="chart-svg" viewBox="0 0 ${width} ${height}" preserveAspectRatio="xMidYMid meet">
            ${yAxisLine}
            ${xAxisLine}
            ${yTickEls}
            ${xTickEls}
            ${yAxisTitle}
            ${xAxisTitle}
            ${bars}
          </svg>
        </div>
      </div>
      ${linkOpen}
    </div>
  `;
}

function renderLineGraph(graph, subjectSource, opts = {}) {
  const showInnerTitle = !opts.suppressInnerTitle;
  if (!graph || !graph.series || !graph.series.length) return '';
  const allPoints = graph.series.flatMap((s) => s.data_points);
  const xs = [...new Set(allPoints.map((p) => p.x))];
  const yValues = allPoints.map((p) => p.y);
  const yMin = Math.min(0, ...yValues);
  const yMax = Math.max(...yValues, 1);
  const yRange = yMax - yMin || 1;
  const padding = { top: 24, right: 24, bottom: 52, left: 52 };
  const width = 420;
  const height = 260;
  const chartLeft = padding.left;
  const chartRight = width - padding.right;
  const chartTop = padding.top;
  const chartBottom = height - padding.bottom;
  const chartWidth = chartRight - chartLeft;
  const chartHeight = chartBottom - chartTop;

  const xScale = (i) => chartLeft + (i / Math.max(xs.length - 1, 1)) * chartWidth;
  const yScale = (v) => chartBottom - ((v - yMin) / yRange) * chartHeight;
  const isBarGraph = graph.graph_type === 'bar';

  const yTicks = computeYTicks(yMin, yMax);

  const yAxisLine = `<line class="axis-line" x1="${chartLeft}" y1="${chartTop}" x2="${chartLeft}" y2="${chartBottom}" />`;
  const xAxisLine = `<line class="axis-line" x1="${chartLeft}" y1="${chartBottom}" x2="${chartRight}" y2="${chartBottom}" />`;

  const yTickEls = yTicks
    .map((v) => {
      const y = yScale(v);
      const label = Number.isInteger(v) ? v : v.toFixed(1);
      return `
        <line class="axis-tick" x1="${chartLeft}" y1="${y}" x2="${chartLeft - 6}" y2="${y}" />
        <text class="axis-tick-label" x="${chartLeft - 8}" y="${y}" text-anchor="end" dominant-baseline="middle">${escapeHtml(String(label))}</text>
      `;
    })
    .join('');

  const slotWidth = chartWidth / Math.max(xs.length, 1);
  const xTickPos = (i) => (isBarGraph ? chartLeft + (i + 0.5) * slotWidth : xScale(i));
  const xTickEls = xs
    .map((label, i) => {
      const x = xTickPos(i);
      const short = String(label).length > 12 ? String(label).slice(0, 10) + '…' : String(label);
      return `
        <line class="axis-tick" x1="${x}" y1="${chartBottom}" x2="${x}" y2="${chartBottom + 6}" />
        <text class="axis-tick-label axis-tick-label-x" x="${x}" y="${chartBottom + 22}" text-anchor="middle">${escapeHtml(short)}</text>
      `;
    })
    .join('');

  const yAxisTitle = graph.y_axis_label
    ? `<text class="axis-title" x="${chartLeft - 28}" y="${(chartTop + chartBottom) / 2}" text-anchor="middle" transform="rotate(-90, ${chartLeft - 28}, ${(chartTop + chartBottom) / 2})">${escapeHtml(graph.y_axis_label)}</text>`
    : '';
  const xAxisTitle = graph.x_axis_label
    ? `<text class="axis-title axis-title-x" x="${(chartLeft + chartRight) / 2}" y="${height - 8}" text-anchor="middle">${escapeHtml(graph.x_axis_label)}</text>`
    : '';

  let dataEls = '';

  if (isBarGraph && graph.series.length > 0) {
    const series0 = graph.series[0];
    const n = xs.length;
    const sw = chartWidth / Math.max(n, 1);
    const barWidth = sw * 0.7;
    const color = COLORS[0];
    series0.data_points.forEach((p, i) => {
      const xi = xs.indexOf(p.x);
      const idx = xi >= 0 ? xi : i;
      const cx = chartLeft + (idx + 0.5) * sw;
      const x = cx - barWidth / 2;
      const barH = ((p.y - yMin) / yRange) * chartHeight;
      const y = chartBottom - barH;
      dataEls += `<rect class="bar-fill-svg ${color}" x="${x}" y="${y}" width="${barWidth}" height="${Math.max(barH, 0)}" rx="4" />`;
    });
  } else {
    const paths = graph.series
      .map((series, seriesIndex) => {
        const color = COLORS[seriesIndex % COLORS.length];
        const d = series.data_points
          .map((p, i) => {
            const xi = xs.indexOf(p.x);
            const x = xScale(xi >= 0 ? xi : i);
            const y = yScale(p.y);
            return `${x},${y}`;
          })
          .join(' L ');
        return d ? `<path class="line ${color}" d="M ${d}" />` : '';
      })
      .join('');
    const dots = graph.series
      .flatMap((series, seriesIndex) => {
        const color = COLORS[seriesIndex % COLORS.length];
        return series.data_points.map((p, i) => {
          const xi = xs.indexOf(p.x);
          const x = xScale(xi >= 0 ? xi : i);
          const y = yScale(p.y);
          return `<circle class="dot ${color}" cx="${x}" cy="${y}" r="4" />`;
        });
      })
      .join('');
    dataEls = paths + dots;
  }

  const legend = graph.series
    .map((s, i) => {
      const desc = s.legend_description
        ? `<span class="legend-desc">${escapeHtml(s.legend_description)}</span>`
        : '';
      return `
        <div class="legend-item">
          <span class="legend-dot ${COLORS[i % COLORS.length]}"></span>
          <div class="legend-text">
            <span class="legend-name">${escapeHtml(s.name)}</span>
            ${desc}
          </div>
        </div>`;
    })
    .join('');

  const rawSource = typeof subjectSource === 'string' ? subjectSource : graph.source || (graph.series[0] && graph.series[0].source);
  const pdfFile = resolvePdfFilename(rawSource);
  const sourceUrl = pdfFile ? buildSourceUrl(pdfFile || rawSource, graph.title, rawSource) : '';

  const innerTitle = showInnerTitle ? `<h4 class="article-viz-title">${escapeHtml(graph.title)}</h4>` : '';

  return `
    <div class="article-viz-inner">
      ${innerTitle}
      <div class="line-graph-wrap">
        <div class="line-graph">
          <svg class="chart-svg" viewBox="0 0 ${width} ${height}" preserveAspectRatio="xMidYMid meet">
            ${yAxisLine}
            ${xAxisLine}
            ${yTickEls}
            ${xTickEls}
            ${yAxisTitle}
            ${xAxisTitle}
            ${dataEls}
          </svg>
        </div>
      </div>
      <div class="legend-box">
        <span class="legend-title">Légende</span>
        <div class="legend">${legend}</div>
      </div>
      ${sourceUrl ? `<a href="${sourceUrl}" class="viz-source-link">Voir la publication</a>` : ''}
    </div>
  `;
}

function renderFigureBlock(fig, slug) {
  const base = `articles/${encodeURIComponent(slug)}/`;
  const figHeading = fig.title ? `<h3 class="article-figure-heading">${escapeHtml(fig.title)}</h3>` : '';
  const caption = fig.caption ? `<figcaption>${escapeHtml(fig.caption)}</figcaption>` : '';
  if (fig.png_path) {
    return `
      <figure class="article-figure">
        ${figHeading}
        <img src="${base}${escapeHtml(fig.png_path)}" alt="${escapeHtml(fig.title || '')}" loading="lazy" width="800" />
        ${caption}
      </figure>
    `;
  }
  const suppressInner = !!fig.title;
  if (fig.figure_type === 'graph' && fig.graph) {
    const src = fig.graph.source || (fig.graph.series && fig.graph.series[0] && fig.graph.series[0].source);
    return `<figure class="article-figure article-figure-svg">${figHeading}${renderLineGraph(fig.graph, src, { suppressInnerTitle: suppressInner })}${caption}</figure>`;
  }
  if (fig.figure_type === 'chart' && fig.chart) {
    if (fig.chart.chart_type === 'bar') {
      const src = fig.chart.source;
      return `<figure class="article-figure article-figure-svg">${figHeading}${renderBarChart(fig.chart, src, { suppressInnerTitle: suppressInner })}${caption}</figure>`;
    }
    return `<figure class="article-figure">${figHeading}<p class="viz-fallback">Graphique non disponible en interactif ; régénérez l’article avec PNG ou utilisez un diagramme en barres.</p>${caption}</figure>`;
  }
  return '';
}

function renderArticle(data, slug) {
  const sectionsHtml = (data.sections || [])
    .map(
      (sec) => `
    <section class="article-section" id="${escapeHtml(sec.id)}">
      <h2 class="article-section-title">${escapeHtml(sec.heading)}</h2>
      ${(sec.paragraphs || []).map((p) => `<p class="article-p">${escapeHtml(p)}</p>`).join('')}
      ${
        sec.metaphor_box
          ? `<aside class="metaphor-box"><span class="metaphor-label">En image</span><p>${escapeHtml(sec.metaphor_box)}</p></aside>`
          : ''
      }
    </section>
  `
    )
    .join('');

  const takeaways = (data.key_takeaways || [])
    .map((t) => `<li>${escapeHtml(t)}</li>`)
    .join('');

  const figuresHtml = (data.figures || []).map((f) => renderFigureBlock(f, slug)).join('');

  const glossaryItems = (data.glossary || [])
    .map((line) => `<li>${escapeHtml(line)}</li>`)
    .join('');
  const glossaryBlock =
    glossaryItems ?
      `<section class="article-section article-glossary"><h2 class="article-section-title">Petit lexique</h2><ul class="takeaways-list">${glossaryItems}</ul></section>`
    : '';

  const practicalTipsFallback = [
    'Faire des pauses écran et noter comment tu te sens avant / après.',
    'Limiter le temps ou les contenus qui te poussent à te comparer aux autres.',
    'Parler à une personne de confiance ou à un professionnel si ça pèse sur ton moral.',
  ];
  const practicalTips =
    Array.isArray(data.practical_tips) && data.practical_tips.length
      ? data.practical_tips
      : practicalTipsFallback;
  const practicalItems = practicalTips.map((t) => `<li>${escapeHtml(t)}</li>`).join('');
  const practicalLead =
    data.practical_intro && String(data.practical_intro).trim()
      ? `<p class="article-practical-lead">${escapeHtml(String(data.practical_intro).trim())}</p>`
      : '<p class="article-practical-lead">Quelques pistes concrètes pour limiter les risques et rester plus vigilant·e.</p>';
  const practicalBlock = `<section class="article-section article-practical" aria-labelledby="practical-heading">
        <h2 id="practical-heading" class="article-section-title">En pratique</h2>
        ${practicalLead}
        <ul class="practical-list">${practicalItems}</ul>
      </section>`;

  return `
    <article class="article-body">
      <header class="article-header">
        <a href="index.html" class="back-link">← Tous les articles</a>
        <h1 class="article-title">${escapeHtml(data.title)}</h1>
        <p class="article-subtitle">${escapeHtml(data.subtitle)}</p>
        <p class="article-deck">${escapeHtml(data.deck)}</p>
        <p class="article-meta">Temps de lecture estimé : ${escapeHtml(String(data.reading_time_min))} min</p>
      </header>
      ${
        takeaways
          ? `<section class="takeaways"><h2 class="takeaways-title">À retenir</h2><ul class="takeaways-list">${takeaways}</ul></section>`
          : ''
      }
      ${sectionsHtml}
      ${figuresHtml ? `<section class="article-section"><h2 class="article-section-title">Figures</h2><div class="article-figures">${figuresHtml}</div></section>` : ''}
      ${
        data.limitations
          ? `<section class="article-section article-limitations"><h2 class="article-section-title">Limites et prudence</h2><p class="article-p">${escapeHtml(data.limitations)}</p></section>`
          : ''
      }
      ${glossaryBlock}
      ${practicalBlock}
    </article>
  `;
}

function renderArticleList(manifest) {
  const items = (manifest.articles || [])
    .map(
      (a) => `
    <a href="index.html?article=${encodeURIComponent(a.slug)}" class="article-card">
      <h2 class="article-card-title">${escapeHtml(a.title)}</h2>
      <p class="article-card-meta">${escapeHtml(a.updated || '')}</p>
    </a>
  `
    )
    .join('');
  return `
    <div class="article-list-wrap">
      <p class="article-list-intro">Choisissez un dossier de publications pour lire la vulgarisation générée.</p>
      <div class="article-list">
        ${items || '<p class="loading">Aucun article pour le moment. Lancez <code>python analyze_papers.py --article &lt;dossier&gt;</code> depuis la racine du projet.</p>'}
      </div>
    </div>
  `;
}

function getQueryArticleSlug() {
  const q = new URLSearchParams(window.location.search).get('article');
  return q ? q.trim() : '';
}

function loadApp() {
  const root = document.getElementById('app');
  if (!root) return;

  const slug = getQueryArticleSlug();
  currentArticleSlug = slug;

  if (!slug) {
    root.innerHTML = '<p class="loading">Chargement de la liste…</p>';
    fetch('articles/manifest.json')
      .then((r) => (r.ok ? r.json() : { articles: [] }))
      .catch(() => ({ articles: [] }))
      .then((manifest) => {
        root.innerHTML = renderArticleList(manifest);
      });
    return;
  }

  root.innerHTML = '<p class="loading">Chargement de l’article…</p>';
  const base = `articles/${encodeURIComponent(slug)}/`;
  Promise.all([
    fetch(base + 'source_to_pdf.json')
      .then((r) => (r.ok ? r.json() : {}))
      .catch(() => ({})),
    fetch(base + 'article.json').then((r) => {
      if (!r.ok) throw new Error('Article introuvable pour ce dossier.');
      return r.json();
    }),
  ])
    .then(([map, data]) => {
      sourceToPdfMap = map || {};
      if (data.slug && data.slug !== slug) {
        console.warn('Slug JSON différent de l’URL');
      }
      root.innerHTML = renderArticle(data, slug);
    })
    .catch((err) => {
      root.innerHTML = `<p class="error">${escapeHtml(err.message)}</p><p><a href="index.html" class="back-link">← Retour</a></p>`;
    });
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', loadApp);
} else {
  loadApp();
}
