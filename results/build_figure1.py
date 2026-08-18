"""Render the paper's apples-to-apples Figure 1 as a deterministic, standalone SVG."""
import html
import json
import pathlib


ROOT = pathlib.Path(__file__).resolve().parents[1]
ROWS = ROOT / "results" / "rows"
OUTPUT = ROOT / "results" / "figure1-scale.svg"

# Okabe-Ito-derived palette used by the reviewed paper figure.
OURS = "#0072B2"
FRONTIER = "#009E73"
COMMUNITY = "#999999"
BASE_CHECKPOINT = "#CC79A7"

SCALE = [
    ("stock Socratic SegEnc\n(our port)", "segenc-port-full-test", "406M", BASE_CHECKPOINT),
    ("span-trained SegEnc", "segenc-c8cont-e4-test", "406M", OURS),
    ("ours, promoted", "loc-w375-l12-b2000-test", "1.2B + 33M", OURS),
    ("ours, protocol-exact", "m3-fusion-lfm2.5-1.2b-test", "1.2B + 22.7M", OURS),
    ("gpt-5.6-luna", "baseline-gpt-5.6-luna-test", "Size undisclosed", FRONTIER),
    ("gpt-5.6-sol", "baseline-gpt-5.6-sol-test", "Size undisclosed", FRONTIER),
    ("base LFM2.5, our spans", "zeroshot-lfm25-spans-test", "1.2B + 33M", BASE_CHECKPOINT),
    ("claude-opus-5", "baseline-claude-opus-5-test", "Size undisclosed", FRONTIER),
    ("claude-haiku-4-5", "baseline-claude-haiku-4-5-test", "Size undisclosed", FRONTIER),
    ("distilbart", "row-distilbart-test", "306M", COMMUNITY),
    ("base LFM2.5, truncated", "zeroshot-lfm25-trunc-test", "1.2B", BASE_CHECKPOINT),
    ("claude-sonnet-5", "baseline-claude-sonnet-5-test", "Size undisclosed", FRONTIER),
    ("bart-large-cnn", "row-bart-large-cnn-test", "406M", COMMUNITY),
    ("pegasus", "row-pegasus-test", "570M", COMMUNITY),
    ("led-base", "row-led-base-test", "162M", COMMUNITY),
]


def rouge1(run_id):
    row = json.loads((ROWS / f"{run_id}.json").read_text(encoding="utf-8"))
    return row["rouge1"] * 100


def _text(x, y, value, **attrs):
    rendered = " ".join(f'{key.replace("_", "-")}="{html.escape(str(val), quote=True)}"'
                        for key, val in attrs.items())
    return f'<text x="{x}" y="{y}" {rendered}>{html.escape(str(value))}</text>'


def build_svg():
    width, height = 1040, 790
    left, top, plot_width = 270, 42, 665
    row_gap, bar_height, maximum = 38, 26, 43
    axis_y = top + len(SCALE) * row_gap
    data = sorted(((label, run_id, rouge1(run_id), size, colour)
                   for label, run_id, size, colour in SCALE), key=lambda item: -item[2])

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">ROUGE-1 comparison under one frozen QMSum scorer</title>',
        '<desc id="desc">Sorted horizontal bars compare locally executable systems and '
        'zero-shot frontier models. Bars include parameter size labels.</desc>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<g font-family="Georgia, Times New Roman, serif" fill="#111111">',
    ]

    for tick in range(0, 41, 10):
        x = left + tick / maximum * plot_width
        lines.append(f'<line x1="{x:.2f}" y1="{top - 8}" x2="{x:.2f}" y2="{axis_y}" '
                     'stroke="#d9d9d9" stroke-width="1"/>')
        lines.append(_text(f"{x:.2f}", axis_y + 21, tick, text_anchor="middle", font_size="13"))

    for index, (label, run_id, score, size, colour) in enumerate(data):
        y = top + index * row_gap
        centre = y + bar_height / 2
        bar_width = score / maximum * plot_width
        label_lines = label.split("\n")
        if len(label_lines) == 1:
            lines.append(_text(left - 12, centre + 5, label, text_anchor="end", font_size="14"))
        else:
            lines.append(f'<text x="{left - 12}" y="{centre - 2}" text-anchor="end" '
                         'font-size="13">'
                         f'<tspan x="{left - 12}" dy="0">{html.escape(label_lines[0])}</tspan>'
                         f'<tspan x="{left - 12}" dy="14">{html.escape(label_lines[1])}</tspan>'
                         '</text>')
        lines.append(f'<rect data-run-id="{html.escape(run_id, quote=True)}" x="{left}" y="{y}" '
                     f'width="{bar_width:.2f}" height="{bar_height}" fill="{colour}"/>')
        lines.append(_text(left + 10, centre + 4.5, size, fill="#ffffff", font_size="12.5",
                           font_weight="600"))
        lines.append(_text(f"{left + bar_width + 8:.2f}", centre + 5, f"{score:.2f}",
                           font_size="13"))

    lines.append(f'<line x1="{left}" y1="{axis_y}" x2="{left + plot_width}" y2="{axis_y}" '
                 'stroke="#111111" stroke-width="1"/>')
    lines.append(_text(left + plot_width / 2, axis_y + 48,
                       "ROUGE-1, official test split (n=281), one frozen scorer",
                       text_anchor="middle", font_size="15"))

    legend = [
        (OURS, "our fine-tuned systems"),
        (FRONTIER, "2026 frontier, zero-shot"),
        (COMMUNITY, "community checkpoint"),
        (BASE_CHECKPOINT, "stock / base checkpoint"),
    ]
    legend_y = axis_y + 88
    for index, (colour, label) in enumerate(legend):
        column, row = index % 2, index // 2
        x, y = left + column * 330, legend_y + row * 28
        lines.append(f'<rect x="{x}" y="{y - 13}" width="20" height="13" fill="{colour}"/>')
        lines.append(_text(x + 29, y - 1, label, font_size="13"))

    lines.extend(['</g>', '</svg>', ''])
    return "\n".join(lines)


def main():
    OUTPUT.write_text(build_svg(), encoding="utf-8")
    print(f"figure 1: {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
