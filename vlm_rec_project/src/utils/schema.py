import json
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


COLOR_12 = [
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
]


SCHEMA_FIELDS = [
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
    "confidence",
]


CONFIDENCE_FIELDS = [
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
]


@dataclass
class SchemaCheckResult:
    ok: bool
    errors: List[str]


def make_empty_output() -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "category": "unknown",
        "color": "unknown",
        "style": "unknown",
        "season": "unknown",
        "material": "unknown",
        "pattern": "unknown",
        "gender": "unknown",
        "fit": "unknown",
        "sleeve_length": "unknown",
        "neckline": "unknown",
        "confidence": {k: 0.0 for k in CONFIDENCE_FIELDS},
    }
    return out


def dumps_strict_json(obj: Any) -> str:
    # ensure_ascii=False is important for Chinese prompts / logs
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def validate_output(obj: Any) -> SchemaCheckResult:
    errors: List[str] = []

    if not isinstance(obj, dict):
        return SchemaCheckResult(False, ["output is not a dict"])

    # field presence
    for k in SCHEMA_FIELDS:
        if k not in obj:
            errors.append(f"missing field: {k}")

    # extra fields not allowed
    for k in obj.keys():
        if k not in SCHEMA_FIELDS:
            errors.append(f"extra field: {k}")

    if "confidence" in obj:
        conf = obj.get("confidence")
        if not isinstance(conf, dict):
            errors.append("confidence is not a dict")
        else:
            for k in CONFIDENCE_FIELDS:
                if k not in conf:
                    errors.append(f"confidence missing: {k}")
                else:
                    v = conf.get(k)
                    if not isinstance(v, (int, float)):
                        errors.append(f"confidence[{k}] is not number")
                    else:
                        if v < 0 or v > 1:
                            errors.append(f"confidence[{k}] out of range")
            for k in conf.keys():
                if k not in CONFIDENCE_FIELDS:
                    errors.append(f"confidence extra: {k}")

    # enum checks (strict only for color in Day2)
    color = obj.get("color")
    if isinstance(color, str):
        if color not in COLOR_12:
            errors.append(f"invalid color enum: {color}")
    else:
        errors.append("color is not a string")

    # basic type checks for other string fields
    for k in CONFIDENCE_FIELDS:
        if k == "color":
            continue
        v = obj.get(k)
        if not isinstance(v, str):
            errors.append(f"{k} is not a string")

    return SchemaCheckResult(len(errors) == 0, errors)


def build_label_from_attrs(attrs: Dict[str, Any], conf_values: Dict[str, float]) -> Dict[str, Any]:
    out = make_empty_output()
    for k in CONFIDENCE_FIELDS:
        v = attrs.get(k)
        if isinstance(v, str) and v.strip() != "":
            out[k] = v
    out["confidence"] = {k: float(conf_values.get(k, 0.1)) for k in CONFIDENCE_FIELDS}
    return out


def confidence_for_value(value: str) -> float:
    if not value or value == "unknown":
        return 0.1
    return 0.9

