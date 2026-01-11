from .schema import dumps_strict_json, make_empty_output


def build_system_prompt() -> str:
    return (
        "你是服饰电商属性抽取助手。\n"
        "你必须严格只输出合法 JSON，对象字段必须与给定 schema 完全一致，禁止输出任何额外文本。\n"
        "如果不确定，请用 unknown 填充。confidence 范围必须在 0~1。"
    )


def build_user_prompt(title: str | None, desc: str | None) -> str:
    schema_json = dumps_strict_json(make_empty_output())
    title = title or ""
    desc = desc or ""
    return (
        "任务：根据输入商品图片与标题/描述，提取服饰属性，按给定 JSON schema 输出。\n"
        f"Schema: {schema_json}\n"
        f"标题: {title}\n"
        f"描述: {desc}\n"
        "<image>"
    )

