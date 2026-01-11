import argparse
import json
import os
from datetime import datetime

import pandas as pd


def _safe_str(x):
    if x is None:
        return None
    if isinstance(x, float) and pd.isna(x):
        return None
    s = str(x).strip()
    return s if s != "" else None


def _article_id_to_image_relpath(article_id: str) -> str:
    # H&M images are stored as images/0xx/0xxxxxxxxx.jpg
    # where 0xx is the first three digits of the zero-padded 10-digit article_id
    s = str(article_id)
    s = s.zfill(10)
    prefix = s[:3]
    return os.path.join("images", prefix, f"{s}.jpg")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw_dir",
        default=os.path.join("vlm_rec_project", "data", "raw", "hm"),
        help="Path to H&M raw data directory containing articles.csv and images/",
    )
    parser.add_argument(
        "--out_file",
        default=os.path.join("vlm_rec_project", "data", "processed", "items.jsonl"),
        help="Output jsonl path",
    )
    parser.add_argument(
        "--max_items",
        type=int,
        default=0,
        help="Limit number of items for quick runs (0 means no limit)",
    )
    args = parser.parse_args()

    raw_dir = args.raw_dir
    articles_csv = os.path.join(raw_dir, "articles.csv")
    images_dir = os.path.join(raw_dir, "images")

    if not os.path.isfile(articles_csv):
        raise FileNotFoundError(f"articles.csv not found: {articles_csv}")
    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"images dir not found: {images_dir}")

    os.makedirs(os.path.dirname(args.out_file), exist_ok=True)

    df = pd.read_csv(articles_csv, dtype={"article_id": str})

    # keep only a minimal, stable subset of columns + raw dump
    required_cols = [
        "article_id",
        "product_type_name",
        "product_group_name",
        "index_group_name",
        "index_name",
        "section_name",
        "colour_group_name",
        "perceived_colour_master_name",
        "graphical_appearance_name",
        "detail_desc",
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            "articles.csv missing required columns: " + ", ".join(missing)
        )

    total = len(df)
    n = total if args.max_items <= 0 else min(total, args.max_items)

    out_count = 0
    missing_img = 0

    with open(args.out_file, "w", encoding="utf-8") as f:
        for i in range(n):
            row = df.iloc[i]
            article_id = _safe_str(row["article_id"])
            if not article_id:
                continue

            image_rel = _article_id_to_image_relpath(article_id)
            image_abs = os.path.join(raw_dir, image_rel)
            if not os.path.isfile(image_abs):
                missing_img += 1
                continue

            # build a pseudo-title from taxonomy fields
            title_parts = [
                _safe_str(row.get("product_type_name")),
                _safe_str(row.get("product_group_name")),
                _safe_str(row.get("index_group_name")),
                _safe_str(row.get("section_name")),
            ]
            title = " | ".join([p for p in title_parts if p]) if any(title_parts) else None

            desc = _safe_str(row.get("detail_desc"))

            raw = {c: _safe_str(row.get(c)) for c in required_cols if c != "article_id"}

            item = {
                "item_id": article_id,
                "image_path": image_rel.replace("\\", "/"),
                "title": title,
                "desc": desc,
                "raw": raw,
                "attrs": {
                    "category": None,
                    "color": None,
                    "style": None,
                    "season": None,
                    "material": None,
                    "pattern": None,
                    "gender": None,
                    "fit": None,
                    "sleeve_length": None,
                    "neckline": None,
                },
            }

            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            out_count += 1

    summary = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "raw_dir": raw_dir,
        "total_articles": total,
        "processed_rows": n,
        "written_items": out_count,
        "missing_images_skipped": missing_img,
        "out_file": args.out_file,
    }

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

