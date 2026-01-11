import argparse
import json
import os
from collections import Counter, defaultdict

import pandas as pd
from PIL import Image


ALLOWED_COLORS_12 = {
    "black",
    "white",
    "gray",
    "red",
    "orange",
    "yellow",
    "green",
    "blue",
    "purple",
    "pink",
    "brown",
    "beige",
    "unknown",
}


def _safe_str(x):
    if x is None:
        return None
    if isinstance(x, float) and pd.isna(x):
        return None
    s = str(x).strip()
    return s if s != "" else None


def _norm_lower(s: str) -> str:
    return s.strip().lower()


def _map_hm_color_to_12(color_name: str) -> str:
    if not color_name:
        return "unknown"

    s = _norm_lower(color_name)

    # H&M has many fine-grained names; we do a conservative mapping.
    if any(k in s for k in ["black", "dark"]):
        return "black"
    if "white" in s or "off white" in s:
        return "white"
    if "grey" in s or "gray" in s:
        return "gray"
    if "red" in s:
        return "red"
    if any(k in s for k in ["orange", "coral"]):
        return "orange"
    if any(k in s for k in ["yellow", "gold"]):
        return "yellow"
    if any(k in s for k in ["green", "khaki", "olive"]):
        return "green"
    if any(k in s for k in ["blue", "navy", "turquoise", "teal"]):
        return "blue"
    if any(k in s for k in ["purple", "lavender", "violet"]):
        return "purple"
    if any(k in s for k in ["pink", "rose", "fuchsia"]):
        return "pink"
    if any(k in s for k in ["brown", "chocolate"]):
        return "brown"
    if any(k in s for k in ["beige", "cream", "sand"]):
        return "beige"

    return "unknown"


def _map_category(product_type_name: str) -> str:
    # For Day1 we keep category close to the original value (lowercased).
    # Day2+ can introduce a stricter category taxonomy.
    if not product_type_name:
        return "other"
    s = _norm_lower(product_type_name)
    return s


def _map_style(index_name: str, section_name: str) -> str:
    # very coarse style proxy
    val = index_name or section_name
    if not val:
        return "unknown"
    return _norm_lower(val)


def _map_gender(index_group_name: str) -> str:
    if not index_group_name:
        return "unknown"
    s = _norm_lower(index_group_name)
    # common H&M values: ladies, men, baby/children, sport, etc.
    if "ladies" in s or "women" in s or "female" in s:
        return "female"
    if "men" in s or "male" in s:
        return "male"
    if "baby" in s or "children" in s or "kids" in s or "child" in s:
        return "kids"
    return "unknown"


def _image_readable(path: str) -> bool:
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--items_file",
        default=os.path.join("vlm_rec_project", "data", "processed", "items.jsonl"),
    )
    parser.add_argument(
        "--raw_dir",
        default=os.path.join("vlm_rec_project", "data", "raw", "hm"),
    )
    parser.add_argument(
        "--out_items_file",
        default=os.path.join("vlm_rec_project", "data", "processed", "items.normalized.jsonl"),
        help="Write normalized items (optional, useful for debugging)",
    )
    parser.add_argument(
        "--report_file",
        default=os.path.join("vlm_rec_project", "reports", "data_profile.md"),
    )
    parser.add_argument(
        "--max_items",
        type=int,
        default=0,
        help="Limit number of items for quick profiling (0 means no limit)",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.items_file):
        raise FileNotFoundError(f"items file not found: {args.items_file}")

    os.makedirs(os.path.dirname(args.report_file), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_items_file), exist_ok=True)

    counters = defaultdict(Counter)

    total = 0
    readable = 0

    unknown_counts = Counter()
    field_counts = Counter()

    n_limit = args.max_items if args.max_items and args.max_items > 0 else None

    with open(args.items_file, "r", encoding="utf-8") as fin, open(
        args.out_items_file, "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            if not line.strip():
                continue

            item = json.loads(line)
            total += 1
            if n_limit is not None and total > n_limit:
                break

            raw = item.get("raw") or {}

            category = _map_category(_safe_str(raw.get("product_type_name")))
            color = _map_hm_color_to_12(
                _safe_str(raw.get("perceived_colour_master_name"))
                or _safe_str(raw.get("colour_group_name"))
            )
            style = _map_style(
                _safe_str(raw.get("index_name")),
                _safe_str(raw.get("section_name")),
            )
            pattern = (
                _norm_lower(_safe_str(raw.get("graphical_appearance_name")))
                if _safe_str(raw.get("graphical_appearance_name"))
                else "unknown"
            )
            gender = _map_gender(_safe_str(raw.get("index_group_name")))

            attrs = item.get("attrs") or {}
            attrs.update(
                {
                    "category": category,
                    "color": color,
                    "style": style,
                    "season": "unknown",
                    "material": "unknown",
                    "pattern": pattern,
                    "gender": gender,
                    "fit": "unknown",
                    "sleeve_length": "unknown",
                    "neckline": "unknown",
                }
            )
            item["attrs"] = attrs

            # image readability
            image_rel = item.get("image_path")
            image_abs = (
                os.path.join(args.raw_dir, image_rel)
                if image_rel
                else None
            )
            if image_abs and os.path.isfile(image_abs) and _image_readable(image_abs):
                readable += 1

            # stats
            counters["category"][category] += 1
            counters["color"][color] += 1
            counters["style"][style] += 1
            counters["pattern"][pattern] += 1
            counters["gender"][gender] += 1

            for k, v in attrs.items():
                field_counts[k] += 1
                if v == "unknown" or v is None:
                    unknown_counts[k] += 1

            fout.write(json.dumps(item, ensure_ascii=False) + "\n")

    img_read_rate = readable / total if total > 0 else 0.0

    def topk(counter: Counter, k: int = 20):
        return counter.most_common(k)

    lines = []
    lines.append("# Data Profile (H&M)\n")
    lines.append(f"Generated from: `{args.items_file}`\n")
    lines.append(f"Total items profiled: **{total}**\n")
    lines.append(f"Image readable rate: **{img_read_rate:.4f}** ({readable}/{total})\n")

    lines.append("## Field distributions (top 20)\n")
    for field in ["category", "color", "style", "pattern", "gender"]:
        lines.append(f"### {field}\n")
        for val, cnt in topk(counters[field], 20):
            lines.append(f"- {val}: {cnt}\n")
        lines.append("\n")

    lines.append("## Unknown ratio\n")
    for k in [
        "category",
        "color",
        "style",
        "season",
        "material",
        "pattern",
        "gender",
        "fit",
        "sleeve_length",
        "neckline",
    ]:
        denom = field_counts.get(k, total)
        unk = unknown_counts.get(k, 0)
        ratio = (unk / denom) if denom else 0.0
        lines.append(f"- {k}: {ratio:.4f} ({unk}/{denom})\n")

    with open(args.report_file, "w", encoding="utf-8") as f:
        f.writelines(lines)

    print(
        json.dumps(
            {
                "items_file": args.items_file,
                "out_items_file": args.out_items_file,
                "report_file": args.report_file,
                "total": total,
                "image_readable": readable,
                "image_readable_rate": img_read_rate,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

