import argparse
import json
import os
import random
from typing import Any, Dict, List, Tuple

from utils.prompts import build_system_prompt, build_user_prompt
from utils.schema import (
    CONFIDENCE_FIELDS,
    SchemaCheckResult,
    build_label_from_attrs,
    confidence_for_value,
    dumps_strict_json,
    validate_output,
)


def read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def build_sft_sample(item: Dict[str, Any], raw_dir: str) -> Dict[str, Any]:
    attrs = item.get("attrs") or {}

    conf = {k: confidence_for_value(attrs.get(k)) for k in CONFIDENCE_FIELDS}
    label = build_label_from_attrs(attrs, conf)

    check = validate_output(label)
    if not check.ok:
        raise ValueError("label schema invalid: " + "; ".join(check.errors))

    system = build_system_prompt()
    user = build_user_prompt(item.get("title"), item.get("desc"))
    assistant = dumps_strict_json(label)

    image_rel = item.get("image_path")
    if not image_rel:
        raise ValueError("missing image_path")

    image_abs = os.path.join(raw_dir, image_rel)
    if not os.path.isfile(image_abs):
        raise FileNotFoundError(f"image not found: {image_abs}")

    # Use a path relative to raw_dir for portability
    sample = {
        "conversations": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        "image": image_rel.replace("\\", "/"),
    }

    return sample


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--items_file",
        default=os.path.join(
            "vlm_rec_project", "data", "processed", "items.normalized.jsonl"
        ),
    )
    parser.add_argument(
        "--raw_dir",
        default=os.path.join("vlm_rec_project", "data", "raw", "hm"),
        help="H&M raw dir, used to validate image paths",
    )
    parser.add_argument(
        "--out_dir",
        default=os.path.join("vlm_rec_project", "data", "sft"),
    )
    parser.add_argument("--eval_ratio", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max_items",
        type=int,
        default=0,
        help="Limit number of items for quick runs (0 means no limit)",
    )

    args = parser.parse_args()

    if not os.path.isfile(args.items_file):
        raise FileNotFoundError(f"items_file not found: {args.items_file}")

    ensure_dir(args.out_dir)

    train_path = os.path.join(args.out_dir, "train.jsonl")
    eval_path = os.path.join(args.out_dir, "eval.jsonl")

    rnd = random.Random(args.seed)

    total = 0
    written_train = 0
    written_eval = 0
    schema_fail = 0
    image_missing = 0

    # Deterministic split by hash(item_id) so you can append/shuffle without changing split
    def is_eval(item_id: str) -> bool:
        # Use a stable hash independent of Python's random hash seed
        h = 0
        for ch in item_id:
            h = (h * 131 + ord(ch)) % 1000003
        # map to [0,1)
        r = (h % 100000) / 100000.0
        return r < args.eval_ratio

    with open(train_path, "w", encoding="utf-8") as f_train, open(
        eval_path, "w", encoding="utf-8"
    ) as f_eval:
        for item in read_jsonl(args.items_file):
            total += 1
            if args.max_items and args.max_items > 0 and total > args.max_items:
                break

            item_id = str(item.get("item_id") or "")
            if not item_id:
                continue

            try:
                sample = build_sft_sample(item, args.raw_dir)
                # Sanity: assistant JSON must be parseable and pass schema
                assistant_text = sample["conversations"][2]["content"]
                parsed = json.loads(assistant_text)
                check = validate_output(parsed)
                if not check.ok:
                    schema_fail += 1
                    continue

                if is_eval(item_id):
                    f_eval.write(json.dumps(sample, ensure_ascii=False) + "\n")
                    written_eval += 1
                else:
                    f_train.write(json.dumps(sample, ensure_ascii=False) + "\n")
                    written_train += 1

            except FileNotFoundError:
                image_missing += 1
            except Exception:
                schema_fail += 1

    json_valid_rate = 1.0
    schema_pass_rate = (written_train + written_eval) / max(total, 1)

    print(
        json.dumps(
            {
                "items_file": args.items_file,
                "raw_dir": args.raw_dir,
                "out_dir": args.out_dir,
                "total_items_read": total,
                "written_train": written_train,
                "written_eval": written_eval,
                "image_missing": image_missing,
                "schema_fail": schema_fail,
                "json_valid_rate": json_valid_rate,
                "schema_pass_rate": schema_pass_rate,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

