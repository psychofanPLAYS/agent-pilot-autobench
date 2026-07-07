"""Chart rendering for the results pages.

A vendored Chart.js (MIT, ``assets/chart.umd.min.js``) is inlined into the HTML so
the generated pages are fully self-contained and work offline from ``file://``.

This module is split into two halves:

- **Runtime** (`chartjs_runtime`, `render_chart`): emit the inlined library, dark
  theme defaults, and per-chart ``<canvas>`` + init snippets.
- **Builders** (`*_config` / `*_chart` functions): pure functions that turn plain
  data (lists/dicts) into Chart.js config dicts. These are the testable seam — no
  HTML, no I/O.

Design: docs/superpowers/specs/2026-06-29-world-class-results-page-design.md §4-5.
"""

from __future__ import annotations

from functools import lru_cache
from html import escape
import json
from pathlib import Path

_ASSET_PATH = Path(__file__).parent / "assets" / "chart.umd.min.js"

# Palette mirrors the results page CSS variables (reports.py::_html_css).
PALETTE = {
    "green": "#3ddc84",
    "red": "#ff6b6b",
    "gold": "#f2c94c",
    "blue": "#65b7ff",
    "violet": "#9b8cff",
    "text": "#f5f7fb",
    "muted": "#a7b0c0",
    "line": "#303846",
}
# Stable per-series colors cycled for multi-model charts.
SERIES_COLORS = ["#65b7ff", "#3ddc84", "#f2c94c", "#9b8cff", "#ff6b6b", "#54d2bd", "#f4a261"]


# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def chartjs_source() -> str:
    return _ASSET_PATH.read_text(encoding="utf-8")


_THEME_JS = """
(function () {
  if (!window.Chart) return;
  var C = window.Chart;
  C.defaults.color = '%(muted)s';
  C.defaults.borderColor = 'rgba(255,255,255,0.06)';
  C.defaults.font.family = "Segoe UI, system-ui, sans-serif";
  C.defaults.font.size = 12;
  C.defaults.plugins = C.defaults.plugins || {};
  C.defaults.plugins.legend = {labels: {color: '%(text)s', boxWidth: 12, boxHeight: 12}};
  C.defaults.plugins.tooltip = {
    callbacks: {
      label: function (ctx) {
        var raw = ctx.raw;
        if (raw && typeof raw === 'object' && raw.label) {
          var x = raw.x, y = raw.y;
          return raw.label + (x !== undefined ? '  (' + x + ', ' + y + ')' : '');
        }
        var v = ctx.formattedValue;
        return (ctx.dataset && ctx.dataset.label ? ctx.dataset.label + ': ' : '') + v;
      }
    }
  };
  window.__apCharts = window.__apCharts || {};
  window.renderChart = function (id, cfg) {
    var el = document.getElementById(id);
    if (!el) return;
    if (window.__apCharts[id]) { window.__apCharts[id].destroy(); }
    window.__apCharts[id] = new C(el, cfg);
  };
})();
""" % {"muted": PALETTE["muted"], "text": PALETTE["text"]}


def chartjs_runtime() -> str:
    """Inlined Chart.js library + dark-theme defaults + `renderChart` helper."""
    return f"<script>{chartjs_source()}</script>\n<script>{_THEME_JS}</script>"


def render_chart(canvas_id: str, config: dict, *, height: int = 320) -> str:
    """A ``<canvas>`` plus the init snippet that draws *config* into it."""
    payload = json.dumps(config)
    cid = escape(canvas_id)
    return (
        f'<div class="chart-box" style="position:relative;height:{int(height)}px">'
        f'<canvas id="{cid}"></canvas></div>'
        f"<script>renderChart({json.dumps(canvas_id)}, {payload});</script>"
    )


def _color(index: int) -> str:
    return SERIES_COLORS[index % len(SERIES_COLORS)]


def _accuracy_color(value: float) -> str:
    """Red (0) -> amber (.5) -> green (1) for a 0..1 score."""
    value = max(0.0, min(1.0, value))
    if value < 0.5:
        r, g, b = 220, int(70 + (value / 0.5) * 100), 70
    else:
        ratio = (value - 0.5) / 0.5
        r, g, b = int(220 - ratio * 160), int(170 + ratio * 30), int(70 + ratio * 20)
    return f"rgb({r},{g},{b})"


# ---------------------------------------------------------------------------
# Global dashboard builders
# Each takes `models`: a list of dicts with keys:
#   name, agent_index (0..100|None), gen_tps, prompt_tps, serving_tps (|None),
#   cold_ttft (|None), vram_gb (|None), pack_scores ({pack_id: 0..1}), family
# ---------------------------------------------------------------------------


def quality_vs_speed_config(models: list[dict]) -> dict:
    """The hero frontier scatter: x = gen tok/s, y = Agent Index, bubble = VRAM."""
    points = []
    for i, m in enumerate(models):
        if m.get("agent_index") is None or not m.get("gen_tps"):
            continue
        vram = m.get("vram_gb")
        radius = 6.0 if not vram else max(5.0, min(22.0, float(vram) * 1.4))
        points.append(
            {
                "x": float(m["gen_tps"]),
                "y": float(m["agent_index"]),
                "r": radius,
                "label": m.get("name", ""),
                "_bg": _color(i),
            }
        )
    return {
        "type": "bubble",
        "data": {
            "datasets": [
                {
                    "label": "Models",
                    "data": points,
                    "backgroundColor": [p["_bg"] + "cc" for p in points],
                    "borderColor": [p["_bg"] for p in points],
                    "borderWidth": 1.5,
                }
            ]
        },
        "options": {
            "responsive": True,
            "maintainAspectRatio": False,
            "plugins": {"legend": {"display": False}},
            "scales": {
                "x": {"title": {"display": True, "text": "Generation tok/s"}},
                "y": {
                    "title": {"display": True, "text": "Agent Index"},
                    "min": 0,
                    "max": 100,
                },
            },
        },
    }


def agent_index_bar_config(models: list[dict]) -> dict:
    """Horizontal ranked bar of Agent Index, red->green colored."""
    scored = [m for m in models if m.get("agent_index") is not None]
    scored.sort(key=lambda m: m["agent_index"], reverse=True)
    labels = [m.get("name", "") for m in scored]
    values = [round(float(m["agent_index"]), 1) for m in scored]
    colors = [_accuracy_color(float(m["agent_index"]) / 100.0) for m in scored]
    return {
        "type": "bar",
        "data": {
            "labels": labels,
            "datasets": [{"label": "Agent Index", "data": values, "backgroundColor": colors}],
        },
        "options": {
            "indexAxis": "y",
            "responsive": True,
            "maintainAspectRatio": False,
            "plugins": {"legend": {"display": False}},
            "scales": {"x": {"min": 0, "max": 100}},
        },
    }


def pack_radar_config(models: list[dict], pack_ids: list[str]) -> dict:
    """Capability radar: one dataset per model across librarian packs."""
    labels = [pid[len("librarian-") :] if pid.startswith("librarian-") else pid for pid in pack_ids]
    datasets = []
    for i, m in enumerate(models):
        scores = m.get("pack_scores") or {}
        if not any(pid in scores for pid in pack_ids):
            continue
        color = _color(i)
        datasets.append(
            {
                "label": m.get("name", ""),
                "data": [
                    round(float(scores[pid]) * 100, 1) if pid in scores else None
                    for pid in pack_ids
                ],
                "backgroundColor": color + "33",
                "borderColor": color,
                "borderWidth": 2,
                "pointBackgroundColor": color,
            }
        )
    return {
        "type": "radar",
        "data": {"labels": labels, "datasets": datasets},
        "options": {
            "responsive": True,
            "maintainAspectRatio": False,
            "scales": {
                "r": {
                    "min": 0,
                    "max": 100,
                    "angleLines": {"color": PALETTE["line"]},
                    "grid": {"color": PALETTE["line"]},
                    "pointLabels": {"color": PALETTE["muted"]},
                    "ticks": {"display": False},
                }
            },
        },
    }


def speed_bars_config(models: list[dict]) -> dict:
    """Grouped bars: generation / prompt / serving tok/s per model."""
    labels = [m.get("name", "") for m in models]
    return {
        "type": "bar",
        "data": {
            "labels": labels,
            "datasets": [
                {
                    "label": "Generation tok/s",
                    "data": [round(float(m.get("gen_tps") or 0), 1) for m in models],
                    "backgroundColor": PALETTE["blue"],
                },
                {
                    "label": "Prompt tok/s",
                    "data": [round(float(m.get("prompt_tps") or 0), 1) for m in models],
                    "backgroundColor": PALETTE["violet"],
                },
                {
                    "label": "Serving tok/s",
                    "data": [
                        round(float(m["serving_tps"]), 1) if m.get("serving_tps") else None
                        for m in models
                    ],
                    "backgroundColor": PALETTE["green"],
                },
            ],
        },
        "options": {
            "responsive": True,
            "maintainAspectRatio": False,
            "scales": {"y": {"title": {"display": True, "text": "tok/s"}}},
        },
    }


def efficiency_bars_config(models: list[dict]) -> dict | None:
    """Quality-per-GB bars. Returns None when no model has VRAM data."""
    rows = [m for m in models if m.get("vram_gb") and m.get("agent_index") is not None]
    if not rows:
        return None
    rows.sort(key=lambda m: m["agent_index"] / m["vram_gb"], reverse=True)
    labels = [m.get("name", "") for m in rows]
    values = [round(float(m["agent_index"]) / float(m["vram_gb"]), 2) for m in rows]
    return {
        "type": "bar",
        "data": {
            "labels": labels,
            "datasets": [
                {
                    "label": "Agent Index per GB VRAM",
                    "data": values,
                    "backgroundColor": PALETTE["gold"],
                }
            ],
        },
        "options": {
            "indexAxis": "y",
            "responsive": True,
            "maintainAspectRatio": False,
            "plugins": {"legend": {"display": False}},
        },
    }


def index_trend_config(history: list[dict]) -> dict | None:
    """Trend line of Agent Index over time. `history`: [{label, index}] sorted.

    Returns None with fewer than two points.
    """
    points = [h for h in history if h.get("index") is not None]
    if len(points) < 2:
        return None
    return {
        "type": "line",
        "data": {
            "labels": [str(h.get("label", "")) for h in points],
            "datasets": [
                {
                    "label": "Agent Index over time",
                    "data": [round(float(h["index"]), 1) for h in points],
                    "borderColor": PALETTE["green"],
                    "backgroundColor": PALETTE["green"] + "22",
                    "fill": True,
                    "tension": 0.25,
                }
            ],
        },
        "options": {
            "responsive": True,
            "maintainAspectRatio": False,
            "scales": {"y": {"min": 0, "max": 100}},
        },
    }


# ---------------------------------------------------------------------------
# Per-run report builders
# ---------------------------------------------------------------------------


def context_scaling_config(rows: list[dict]) -> dict | None:
    """Context-size -> gen tok/s (left axis) + cold TTFT (right axis).

    `rows`: [{context_size, generation_tps, cold_ttft_ms}]. None if <2 rows.
    """
    ordered = sorted((r for r in rows if r.get("context_size")), key=lambda r: r["context_size"])
    if len(ordered) < 2:
        return None
    labels = [str(int(r["context_size"])) for r in ordered]
    datasets = [
        {
            "label": "Generation tok/s",
            "data": [round(float(r.get("generation_tps") or 0), 2) for r in ordered],
            "borderColor": PALETTE["blue"],
            "backgroundColor": PALETTE["blue"] + "22",
            "yAxisID": "y",
            "tension": 0.25,
        }
    ]
    if any(r.get("cold_ttft_ms") is not None for r in ordered):
        datasets.append(
            {
                "label": "Cold TTFT (ms)",
                "data": [
                    round(float(r["cold_ttft_ms"]), 0)
                    if r.get("cold_ttft_ms") is not None
                    else None
                    for r in ordered
                ],
                "borderColor": PALETTE["gold"],
                "backgroundColor": PALETTE["gold"] + "22",
                "yAxisID": "y1",
                "tension": 0.25,
            }
        )
    return {
        "type": "line",
        "data": {"labels": labels, "datasets": datasets},
        "options": {
            "responsive": True,
            "maintainAspectRatio": False,
            "scales": {
                "x": {"title": {"display": True, "text": "Context size"}},
                "y": {"position": "left", "title": {"display": True, "text": "tok/s"}},
                "y1": {
                    "position": "right",
                    "title": {"display": True, "text": "TTFT ms"},
                    "grid": {"drawOnChartArea": False},
                },
            },
        },
    }


def attempts_progression_config(attempts: list[dict]) -> dict | None:
    """Optimizer score per attempt. `attempts`: [{attempt, score}]. None if empty."""
    ordered = sorted(attempts, key=lambda a: a.get("attempt", 0))
    if not ordered:
        return None
    return {
        "type": "line",
        "data": {
            "labels": [f"#{int(a.get('attempt', i + 1))}" for i, a in enumerate(ordered)],
            "datasets": [
                {
                    "label": "Score",
                    "data": [round(float(a.get("score") or 0), 3) for a in ordered],
                    "borderColor": PALETTE["violet"],
                    "backgroundColor": PALETTE["violet"] + "22",
                    "fill": True,
                    "tension": 0.2,
                }
            ],
        },
        "options": {"responsive": True, "maintainAspectRatio": False},
    }


def outcome_doughnut_config(correct: int, wrong: int, incomplete: int) -> dict | None:
    """Question outcome doughnut. None when there are no questions."""
    if correct + wrong + incomplete <= 0:
        return None
    return {
        "type": "doughnut",
        "data": {
            "labels": ["Correct", "Wrong", "Incomplete"],
            "datasets": [
                {
                    "data": [int(correct), int(wrong), int(incomplete)],
                    "backgroundColor": [PALETTE["green"], PALETTE["red"], PALETTE["muted"]],
                }
            ],
        },
        "options": {"responsive": True, "maintainAspectRatio": False},
    }


def pack_accuracy_bars_config(packs: list[dict]) -> dict | None:
    """Per-pack accuracy bars for a single run. `packs`: [{pack_id, accuracy}]."""
    rows = [p for p in packs if p.get("accuracy") is not None]
    if not rows:
        return None
    labels = [
        p["pack_id"][len("librarian-") :]
        if str(p["pack_id"]).startswith("librarian-")
        else p["pack_id"]
        for p in rows
    ]
    values = [round(float(p["accuracy"]) * 100, 1) for p in rows]
    colors = [_accuracy_color(float(p["accuracy"])) for p in rows]
    return {
        "type": "bar",
        "data": {
            "labels": labels,
            "datasets": [{"label": "Accuracy %", "data": values, "backgroundColor": colors}],
        },
        "options": {
            "responsive": True,
            "maintainAspectRatio": False,
            "plugins": {"legend": {"display": False}},
            "scales": {"y": {"min": 0, "max": 100}},
        },
    }
