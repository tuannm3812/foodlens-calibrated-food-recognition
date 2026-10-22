"""Export detector crops for hand labelling, and score the labels afterwards.

The crop figures in ``docs/3_model_results.md`` (section 20) score every crop
against the photo's dish label. A crop of a side bowl or a garnish then counts
as a miss even when the classifier named it correctly. This tool gives each
crop its own label so the crop path can be judged on what the crop shows.

``export`` reads a saved ``scripts/measure_runtime.py`` run (its rows carry
every crop's box and classification), draws a fixed, seeded subset of images
where the detector proposed real regions, and writes:

- ``crops/<n>_<detection index>.jpg``: each crop, cut from the source image;
- ``labels.csv``: one row per crop, with the label columns left blank;
- ``review.html``: each image with its boxes drawn, its crops beside it, and
  whatever the label columns hold, to read in a browser.

Fill the label columns of ``labels.csv`` (by hand, or a proposed first pass
that a person then checks), re-run ``export --refresh-html`` to see them on
the page, then ``score`` the file.

Label columns:

- ``region_kind``: what the crop mostly shows, one of ``REGION_KINDS``.
- ``region_label``: the Food-101 class the crop shows, if it shows one;
  blank when it shows no Food-101 class (a plain bowl, a drink, a hand).
- ``labeller``: who set the row, e.g. ``claude-proposed`` or ``mike``.
- ``notes``: free text.

Nothing is written to ``app/artifacts/``. Output is untracked under
``results/`` by default.

Usage:
    python scripts/crop_review.py export \\
        --measurement results/runtime_measurement_val_20261015T060352Z/measurement.json \\
        --images-dir data/food41/images --count 50
    python scripts/crop_review.py score results/crop_review_<stamp>/labels.csv
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import random
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

REGION_KINDS = (
    "whole_dish",  # the labelled dish, all or nearly all of it
    "part_of_dish",  # a recognisable part of the labelled dish
    "side_item",  # another food beside the dish (rice, salad, sauce)
    "drink",
    "empty_container",  # a bowl, plate or cup with little or no food
    "non_food",  # hands, cutlery, table, text, people
    "unclear",  # too small or blurred to tell
)

LABEL_COLUMNS = ("region_kind", "region_label", "labeller", "notes")
COLUMNS = (
    "image",
    "dish_label",
    "crop_file",
    "detection_index",
    "detector_label",
    "proposal_role",
    "detector_confidence",
    "area_ratio",
    "box",
    "classifier_top1",
    "classifier_confidence",
    "app_pick",
    *LABEL_COLUMNS,
)


def select_images(
    rows: Sequence[Mapping[str, Any]], count: int, seed: int
) -> list[Mapping[str, Any]]:
    """A seeded sample of rows where the detector proposed real regions.

    Detector fallbacks are left out: their only crop is the whole image, which
    the dish label already describes. Rows are sorted by image first, so the
    same file and seed give the same sample anywhere.
    """
    eligible = sorted(
        (row for row in rows if row.get("crops") and not row.get("detector_fallback")),
        key=lambda row: row["image"],
    )
    picked = random.Random(seed).sample(eligible, min(count, len(eligible)))
    return sorted(picked, key=lambda row: row["image"])


def app_pick_index(crops: Sequence[Mapping[str, Any]]) -> int:
    """The position of the crop the app shows first (highest classifier score)."""
    return max(range(len(crops)), key=lambda i: crops[i]["top1_confidence"])


def label_rows(selected: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """One blank-labelled row per crop of each selected image."""
    out = []
    for number, row in enumerate(selected, start=1):
        pick = app_pick_index(row["crops"])
        for position, crop in enumerate(row["crops"]):
            out.append(
                {
                    "image": row["image"],
                    "dish_label": row["label"],
                    "crop_file": f"crops/{number:03d}_{crop['detection_index']}.jpg",
                    "detection_index": crop["detection_index"],
                    "detector_label": crop["detector_label"],
                    "proposal_role": crop["proposal_role"],
                    "detector_confidence": round(crop["detector_confidence"], 4),
                    "area_ratio": round(crop["area_ratio"], 4),
                    "box": " ".join(str(value) for value in crop["box"]),
                    "classifier_top1": crop["top1"],
                    "classifier_confidence": round(crop["top1_confidence"], 4),
                    "app_pick": "yes" if position == pick else "no",
                    **{column: "" for column in LABEL_COLUMNS},
                }
            )
    return out


def write_labels(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def read_labels(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def normalized(row: Mapping[str, str]) -> dict[str, str]:
    """The row with surrounding whitespace removed from every text cell.

    A hand-edited ``"pizza "`` must validate and score as ``"pizza"``; checking
    stripped values but scoring raw ones would pass a row and then miscount it.
    """
    return {key: value.strip() if isinstance(value, str) else value for key, value in row.items()}


def label_problems(rows: Sequence[Mapping[str, str]], class_names: set[str]) -> list[str]:
    """Rows whose labels are missing or not one of the allowed values."""
    problems = []
    for line, row in enumerate(map(normalized, rows), start=2):
        kind = row["region_kind"]
        region = row["region_label"]
        if kind not in REGION_KINDS:
            problems.append(f"line {line}: region_kind {kind!r} is not one of {REGION_KINDS}")
        if region and region not in class_names:
            problems.append(f"line {line}: region_label {region!r} is not a Food-101 class")
        if kind in ("whole_dish", "part_of_dish") and region != row["dish_label"]:
            problems.append(
                f"line {line}: a {kind} crop should carry the dish label {row['dish_label']!r}"
            )
    return problems


def rate(hits: int, total: int) -> float | None:
    return round(hits / total, 4) if total else None


def score(rows: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    """What the crops show, and how often the classifier names it.

    ``region_correct`` compares the classifier with the crop's own label, so a
    side item named correctly counts as right. ``dish_agreement`` is the old
    proxy (classifier against the photo's dish label), kept for comparison.
    Crops with no Food-101 class in them (a plain bowl, non-food) cannot be
    named correctly; they are counted separately.
    """
    rows = [normalized(row) for row in rows]

    def summary(subset: Sequence[Mapping[str, str]]) -> dict[str, Any]:
        nameable = [row for row in subset if row["region_label"]]
        return {
            "crops": len(subset),
            "kinds": dict(Counter(row["region_kind"] for row in subset).most_common()),
            "nameable": len(nameable),
            "region_correct": rate(
                sum(row["classifier_top1"] == row["region_label"] for row in nameable),
                len(nameable),
            ),
            "dish_agreement": rate(
                sum(row["classifier_top1"] == row["dish_label"] for row in subset), len(subset)
            ),
        }

    by_detector: dict[str, list[Mapping[str, str]]] = defaultdict(list)
    for row in rows:
        by_detector[row["detector_label"]].append(row)
    picks = [row for row in rows if row["app_pick"] == "yes"]
    return {
        "images": len({row["image"] for row in rows}),
        "all_crops": summary(rows),
        "app_picks": summary(picks),
        # The app's pick shows the dish (all or part) and names it.
        "app_pick_shows_and_names_dish": rate(
            sum(
                row["region_kind"] in ("whole_dish", "part_of_dish")
                and row["classifier_top1"] == row["dish_label"]
                for row in picks
            ),
            len(picks),
        ),
        "by_detector_label": {
            label: summary(subset) for label, subset in sorted(by_detector.items())
        },
        "labellers": dict(Counter(row["labeller"] for row in rows)),
    }


# --- Images and the review page (needs Pillow) ----------------------------------


def save_crops(
    selected: Sequence[Mapping[str, Any]], images_dir: Path, output: Path
) -> list[str]:
    """Cut every crop from its source image; returns images that were missing.

    Images are decoded exactly as ``measure_runtime.py`` decoded them, through
    the backend's ``open_rgb_image`` (no EXIF transpose), so each saved box
    cuts the same pixels the classifier saw.
    """
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from app.backend.imaging import open_rgb_image

    missing = []
    (output / "crops").mkdir(parents=True, exist_ok=True)
    (output / "images").mkdir(exist_ok=True)
    for number, row in enumerate(selected, start=1):
        source = images_dir / row["image"]
        if not source.is_file():
            missing.append(str(source))
            continue
        image = open_rgb_image(source.read_bytes())
        image.save(output / "images" / f"{number:03d}.jpg", quality=88)
        for crop in row["crops"]:
            x1, y1, x2, y2 = crop["box"]
            image.crop((x1, y1, x2, y2)).save(
                output / "crops" / f"{number:03d}_{crop['detection_index']}.jpg", quality=92
            )
    return missing


def review_html(rows: Sequence[Mapping[str, str]], title: str) -> str:
    """A static page: each image with its boxes drawn, then its crops and labels."""
    by_image: dict[str, list[Mapping[str, str]]] = defaultdict(list)
    for row in rows:
        by_image[row["image"]].append(row)
    sections = []
    for image, crops in by_image.items():
        number = crops[0]["crop_file"].split("/")[1].split("_")[0]
        boxes = []
        for crop in crops:
            x1, y1, x2, y2 = (int(value) for value in crop["box"].split())
            colour = "#d33" if crop["app_pick"] == "yes" else "#36c"
            boxes.append(
                f'<rect x="{x1}" y="{y1}" width="{x2 - x1}" height="{y2 - y1}" '
                f'fill="none" stroke="{colour}" stroke-width="4"/>'
                f'<text x="{x1 + 6}" y="{y1 + 26}" fill="{colour}" font-size="24">'
                f"{html.escape(str(crop['detection_index']))}</text>"
            )
        tiles = []
        for crop in crops:
            pick = " (app shows this)" if crop["app_pick"] == "yes" else ""
            label = html.escape(crop["region_kind"] or "unlabelled")
            region = html.escape(crop["region_label"] or "no Food-101 class")
            tiles.append(
                f'<figure><img src="{html.escape(crop["crop_file"])}" loading="lazy">'
                f"<figcaption><b>#{html.escape(str(crop['detection_index']))}{pick}</b><br>"
                f"detector: {html.escape(crop['detector_label'])} "
                f"({float(crop['area_ratio']):.0%} of image)<br>"
                f"classifier: {html.escape(crop['classifier_top1'])} "
                f"{float(crop['classifier_confidence']):.0%}<br>"
                f"label: <b>{label}</b>, {region}<br>"
                f"<small>{html.escape(crop['labeller'])} "
                f"{html.escape(crop['notes'])}</small></figcaption></figure>"
            )
        sections.append(
            f"<section><h2>{number}. {html.escape(image)}</h2>"
            f'<div class="row"><svg viewBox="0 0 1 1" class="photo" data-src="images/{number}.jpg">'
            f'<image href="images/{number}.jpg"/>{"".join(boxes)}</svg>'
            f'<div class="crops">{"".join(tiles)}</div></div></section>'
        )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
body {{ font: 14px/1.4 system-ui, sans-serif; margin: 16px; background: #fff; color: #111; }}
section {{ border-top: 1px solid #ccc; padding: 12px 0; }}
h2 {{ font-size: 16px; margin: 0 0 8px; }}
.row {{ display: flex; flex-wrap: wrap; gap: 16px; align-items: flex-start; }}
.photo {{ width: 360px; max-width: 100%; height: auto; }}
.crops {{ display: flex; flex-wrap: wrap; gap: 12px; }}
figure {{ margin: 0; width: 180px; }}
figure img {{ width: 180px; height: 180px; object-fit: contain; background: #eee; }}
</style></head><body>
<h1>{html.escape(title)}</h1>
<p>Red box: the crop the app shows first. Blue: other crops. Labels come from
<code>labels.csv</code>; edit it and rerun <code>export --refresh-html</code>.</p>
{"".join(sections)}
<script>
// Size each SVG to its photo so the boxes, in source pixels, line up.
for (const svg of document.querySelectorAll("svg.photo")) {{
  const img = new Image();
  img.onload = () => {{
    svg.setAttribute("viewBox", `0 0 ${{img.naturalWidth}} ${{img.naturalHeight}}`);
    const inner = svg.querySelector("image");
    inner.setAttribute("width", img.naturalWidth);
    inner.setAttribute("height", img.naturalHeight);
  }};
  img.src = svg.dataset.src;
}}
</script>
</body></html>
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export", help="write crops, labels.csv and review.html")
    export.add_argument("--measurement", type=Path, help="a saved measurement.json")
    export.add_argument("--images-dir", type=Path, help="directory of class folders")
    export.add_argument("--count", type=int, default=50)
    export.add_argument("--seed", type=int, default=20261017)
    export.add_argument("--output-dir", type=Path)
    export.add_argument(
        "--refresh-html",
        type=Path,
        metavar="DIR",
        help="only rebuild DIR/review.html from DIR/labels.csv",
    )
    scorer = commands.add_parser("score", help="summarise a filled labels.csv")
    scorer.add_argument("labels", type=Path)
    scorer.add_argument(
        "--class-names",
        type=Path,
        default=REPO_ROOT / "app" / "artifacts" / "class_names.json",
        help="the served class list, to check region_label values (read only)",
    )
    args = parser.parse_args(argv)

    if args.command == "score":
        rows = read_labels(args.labels)
        class_names = set(json.loads(args.class_names.read_text()))
        problems = label_problems(rows, class_names)
        if problems:
            print("\n".join(problems))
            return 1
        print(json.dumps(score(rows), indent=2))
        return 0

    if args.refresh_html:
        rows = read_labels(args.refresh_html / "labels.csv")
        (args.refresh_html / "review.html").write_text(
            review_html(rows, f"Crop review: {args.refresh_html.name}"), encoding="utf-8"
        )
        print(f"wrote {args.refresh_html / 'review.html'}")
        return 0
    if args.measurement is None or args.images_dir is None:
        parser.error("export needs --measurement and --images-dir (or --refresh-html)")
    if args.count < 1:
        parser.error("--count must be at least 1")

    measurement = json.loads(args.measurement.read_text())
    selected = select_images(measurement["rows"], args.count, args.seed)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output = args.output_dir or REPO_ROOT / "results" / f"crop_review_{stamp}"
    if output.exists() and any(output.iterdir()):
        # Re-exporting would replace a labels.csv that may hold finished labels.
        parser.error(
            f"{output} is not empty; choose a new --output-dir, "
            "or use --refresh-html to rebuild only the page"
        )
    output.mkdir(parents=True, exist_ok=True)
    missing = save_crops(selected, args.images_dir, output)
    if missing:
        raise SystemExit(f"{len(missing)} source images are missing, e.g. {missing[0]}")
    rows = label_rows(selected)
    write_labels(output / "labels.csv", rows)
    (output / "review.html").write_text(
        review_html(rows, f"Crop review: {output.name}"), encoding="utf-8"
    )
    (output / "export.json").write_text(
        json.dumps(
            {
                "measurement": str(args.measurement),
                "measured_at": measurement.get("measured_at"),
                "model": measurement.get("model"),
                "sample": measurement.get("sample"),
                "count": args.count,
                "seed": args.seed,
                "images": len(selected),
                "crops": len(rows),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"wrote {len(selected)} images and {len(rows)} crops to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
