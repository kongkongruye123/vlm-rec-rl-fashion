# VLM + 推荐系统（服饰电商）算法岗作品集：绝对详细执行方案

> 目标：在本仓库（minimind-v）基础上，构建一个面向“多模态大模型 + 推荐系统”的**算法岗**项目闭环：
> - 训练/微调一个“图片 → 结构化属性 JSON”的 VLM（SFT + LoRA）
> - 用结构化属性做可解释重排（B）
> - 用强化学习做列表级重排/多样性优化（C）
> - 输出可量化对比：Zero-shot vs SFT/LoRA，MMR vs RL
> - 给出可演示 Demo：对话式约束输入 + 可解释推荐理由

---

## 0. 项目定义（冻结，不在执行中反复改）

### 0.1 目标（必须达成）
- 以 **Kaggle 公开服饰电商数据**（带图片与属性字段）为监督信号，构建数据管线。
- 在同一基座 VLM 上完成：
  - Base（不微调，zero-shot/提示词抽取）
  - SFT（领域 SFT，强制 JSON 输出）
  - SFT+LoRA（推荐主版本）
- 输出可量化对比：
  - 结构化属性抽取：JSON_valid_rate、field_F1、schema_pass_rate
  - 可解释重排：constraint_satisfaction@K、explanation_faithfulness
  - 列表级优化：ILD@K、coverage@K、NDCG@K（或 proxy relevance）
- 输出可演示 Demo（本地即可，不要求上线）：
  - 输入：文字约束（如“蓝色/度假风/长袖”）+ 可选参考图
  - 输出：Top-K 商品图 + 每条结果的“基于属性证据”的解释 + 可选多样性开关（Base/MMR/RL）
- 代码可复现：从下载/导入 Kaggle 数据 → 生成训练集 → 微调 → 评测 → Demo 的命令链条可跑通。

### 0.2 交付物清单（检查表）
- [ ] data/raw/kaggle_manifest.json（数据集来源、版本、字段说明；不提交大文件）
- [ ] data/raw/（本地：图片与元数据；gitignore）
- [ ] data/processed/items.jsonl（统一 item 表：item_id + image_path + 原始字段 + 规范化字段）
- [ ] data/sft/train.jsonl + data/sft/eval.jsonl（VLM SFT 数据）
- [ ] outputs/adapters/sft_lora/（LoRA 权重）
- [ ] reports/extraction_base.json + extraction_sft.json + extraction_lora.json（逐样本抽取评测）
- [ ] reports/rerank_report.md（约束满足率/解释忠实度/样例）
- [ ] reports/rl_report.md（RL vs MMR vs Base 的指标表）
- [ ] apps/demo_gradio.py（本地 Demo）
- [ ] docs/vlm_rec_plan.md（本文件，作为执行准绳）

### 0.3 固定技术选择（禁止扩散）
- 数据来源：Kaggle 服饰电商数据（带图片与属性字段）。
- VLM：使用本仓库 `MiniMindVLM`，视觉编码器为 CLIP ViT-B/16（仓库默认）。
- 微调方式：优先 LoRA（或 QLoRA 若你额外接入 bitsandbytes；第一版不强制）。
- 训练任务：**结构化属性抽取**（image + title/desc → JSON schema）。
- RL 训练：离线仿真（不依赖线上日志）。
- 评测：离线指标为主，强调可复现与对比实验。

---

## 1. 目录结构（在仓库内落地）

在仓库根目录创建如下路径（不污染原训练脚本，独立成项目目录）：
```text
vlm_rec_project/
  data/
    raw/                       # Kaggle 原始数据(本地，gitignore)
    processed/                 # 统一格式 items.jsonl
    sft/                       # train.jsonl / eval.jsonl
    eval/                      # extraction_eval.jsonl / rerank_eval.jsonl
    cache/                     # 中间缓存（embedding、属性预测等）
  src/
    00_prepare_kaggle_data.py   # 读取 Kaggle CSV/图片，生成 items.jsonl
    01_normalize_schema.py      # 字段映射与枚举规范化
    02_make_sft_jsonl.py        # 自动生成 SFT 数据（对话格式+JSON）
    03_train_sft_lora.py        # LoRA 微调训练（复用/封装仓库训练能力）
    04_eval_extraction.py       # 抽取评测：Base vs SFT vs LoRA
    05_build_index.py           # 向量索引/候选池构建（可选）
    06_rerank_explain.py        # 约束重排 + 可解释性评测
    07_train_rl_rerank.py       # RL 列表级重排（离线仿真）
    08_eval_rl.py               # RL vs MMR vs Base 指标对比
    09_demo_gradio.py           # Demo
    utils/
      schema.py                 # schema 定义与枚举
      prompts.py                # 统一 prompt 模板
      io.py                     # jsonl 读写
      metrics.py                # F1/valid_rate/ILD/coverage 等
      rerank.py                 # Base/MMR/RL 重排逻辑
  outputs/
    adapters/
      sft_lora/
    logs/
  reports/
    extraction_*.json
    rerank_report.md
    rl_report.md
```

---

## 2. 数据集选择与导入（严格要求字段，避免后续返工）

### 2.1 Kaggle 数据集硬性要求
必须满足：
- 图片可用（本地图片路径或可下载 URL）
- 至少一个类别字段（`category`/`product_type`/`masterCategory` 等）
- 至少 2 个可用属性字段（例如 `color`, `season`, `usage`, `gender`, `pattern`, `material` 等）

### 2.1.1 推荐可用公开数据集（优先级从高到低，建议先选 1 个跑通闭环）

> 说明：我无法在当前环境直接联网检索 Kaggle 的最新页面，因此这里给出在 Kaggle/公开社区中长期稳定、使用频率高、且满足“图片+属性字段”的常见数据集方向与对应链接入口。你实际执行时只需打开链接确认字段名并下载到 `vlm_rec_project/data/raw/`。

#### 方案 A（最推荐，字段最接近“电商服饰属性”）：H&M Personalized Fashion Recommendations
- 链接：`https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations`
- 你能用到的字段：
  - `articles.csv`：
    - `article_id`（item_id）
    - `product_type_name` / `product_group_name` / `graphical_appearance_name`
    - `colour_group_name` / `perceived_colour_value_name` / `perceived_colour_master_name`
    - `department_name` / `index_name` / `section_name`
    - `detail_desc`（desc）
  - 图片：`images/0xx/0xxxxxxxxx.jpg`（按 article_id 对应）
- 优点：
  - 真实电商属性字段非常全（颜色、外观、品类、部门等），天然适合做 schema 监督。
  - 还带 `transactions_train.csv`（可选）用于构造用户偏好画像与 RL 环境更贴真实。
- 注意：
  - 图片数量较大，第一版可只抽 20k–100k article。

#### 方案 B（偏“商品图+类目/属性”，容易跑通）：Fashion Product Images Dataset (Myntra)
- 链接（常见入口之一）：`https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset`
- 常见字段（不同镜像版本可能略有差异）：
  - `id`（item_id）
  - `masterCategory` / `subCategory` / `articleType`
  - `baseColour` / `season` / `usage` / `gender`
  - 图片：通常为 `images/{id}.jpg`
- 优点：
  - “颜色/季节/usage/gender”非常适合做结构化抽取与约束重排 demo。
  - 数据规模适中（个人 GPU 训练比较友好）。

#### 方案 C（备选，偏分类/少量属性）：DeepFashion 相关 Kaggle 镜像
- 链接（搜索入口）：`https://www.kaggle.com/search?q=deepfashion+images+attributes`
- 说明：
  - DeepFashion 的 Kaggle 镜像通常存在多个版本，字段与划分不一。
  - 若你更想做“学术属性多标签”任务，可选此方向；但第一版建议用 A/B。

#### 方案 D（兜底）：Kaggle 上任意带 `color/category/gender` 的服饰商品数据
- 搜索入口：`https://www.kaggle.com/search?q=fashion+images+attributes+color+category`
- 选择标准：
  - 至少包含 `image` + `category` + `color`（或等价字段）
  - 最好包含 `season/usage/gender/material/pattern` 中的 2 个以上

### 2.1.2 本项目默认推荐选择
- 默认选择：**方案 B（Myntra Fashion Product Images）**用于最快跑通。
- 若你希望把 RL/推荐做得更“业务真实”：选择 **方案 A（H&M）**，因为它附带 transactions。


### 2.2 数据导入产物：items.jsonl（统一格式）
每行一个商品：
```json
{
  "item_id": "string",
  "image_path": "relative/or/absolute/path",
  "title": "string|null",
  "desc": "string|null",
  "raw": {"...": "original fields"},
  "attrs": {
    "category": "dress",
    "color": "blue",
    "style": "vacation",
    "season": "summer",
    "material": "cotton",
    "pattern": "solid",
    "gender": "female",
    "fit": "unknown",
    "sleeve_length": "unknown",
    "neckline": "unknown"
  }
}
```

### 2.3 字段映射与规范化（写死，避免“看感觉”）
- 颜色：归一到 12 色（black/white/gray/red/orange/yellow/green/blue/purple/pink/brown/beige）+ `unknown`
- 类别：归一到 20–50 个常见服饰类（dress/top/pants/shoes/bag/...）+ `other`
- 风格/usage：若数据集中有 `usage` 直接映射到 style；没有则从 title 规则抽取，抽不到 `unknown`
- season/material/pattern/gender/fit：优先使用原字段；否则 `unknown`

验收标准：
- 图片可读率 >= 99%（损坏图片剔除）
- 规范化字段缺失率（unknown 比例）统计输出到 reports/data_profile.md

---

## 3. 统一输出 JSON Schema（写死）

所有 VLM 输出必须严格 JSON，且字段固定：
```json
{
  "category": "string",
  "color": "string",
  "style": "string",
  "season": "string",
  "material": "string",
  "pattern": "string",
  "gender": "string",
  "fit": "string",
  "sleeve_length": "string",
  "neckline": "string",
  "confidence": {
    "category": 0.0,
    "color": 0.0,
    "style": 0.0,
    "season": 0.0,
    "material": 0.0,
    "pattern": 0.0,
    "gender": 0.0,
    "fit": 0.0,
    "sleeve_length": 0.0,
    "neckline": 0.0
  }
}
```
规则：
- 只能输出上述字段，禁止额外文本。
- 枚举不在允许集合内 → 视为不合法（schema_fail）。
- confidence 缺失/不可解析 → schema_fail。

---

## 4. Prompt 模板（写死）

### 4.1 System Prompt（固定）
- 角色：服饰电商属性抽取助手
- 约束：只输出合法 JSON；字段必须齐全；未知填 unknown；confidence 0~1。

### 4.2 User 输入格式（固定）
```text
任务：根据输入商品图片与标题/描述，提取服饰属性，按给定 JSON schema 输出。
Schema: {schema_json}
标题: {title}
描述: {desc}
<image>
```

Assistant 输出：严格 JSON（不包裹 ```）。

---

## 5. Step 1：数据准备（Day1）

### 5.1 产出
- data/processed/items.jsonl
- reports/data_profile.md（字段分布/unknown比例/图片损坏率）

### 5.2 具体操作
1. 下载 Kaggle 数据到 `vlm_rec_project/data/raw/`（本地，不入库）。
2. 实现 `00_prepare_kaggle_data.py`：读取 CSV/图片，生成 `items.jsonl`。
3. 实现 `01_normalize_schema.py`：字段映射、枚举规范化、输出统计。

### 5.3 验收标准
- items.jsonl 行数 >= 10k（建议 20k–100k，按硬盘与训练时间）
- 图片可读率 >= 99%
- profile 报告生成成功

---

## 6. Step 2：SFT 数据构建（Day2）

### 6.1 产出
- data/sft/train.jsonl（建议 20k–100k）
- data/sft/eval.jsonl（建议 1k–5k）

### 6.2 样本生成规则（必须可验证）
- 对每个 item 生成 1 条样本（必要时可做数据增强生成 2–3 条不同 prompt 版本）。
- assistant 输出由 items.jsonl 的 `attrs` 直接生成。
- confidence 初始可统一设置：
  - 原字段有值：0.9
  - 规则抽取：0.7
  - unknown：0.1

### 6.3 数据质量门槛（写死）
- JSON 可解析率 = 100%（生成时校验，不通过直接丢弃）
- schema_pass_rate >= 99%（枚举合法性）

---

## 7. Step 3：VLM 微调训练（SFT + LoRA）（Day3–Day4）

### 7.1 训练目标
- 让 VLM 学会稳定输出结构化 JSON（减少 hallucination，提升可控性）。

### 7.2 训练设置（推荐）
- 冻结 vision encoder（CLIP）
- 训练 vision projection + LoRA(LLM)
- max_length：1024 或 2048（按显存）
- 优先 bf16；不支持则 fp16

### 7.3 消融实验（必须跑）
- E0（Base）：不训练，仅 prompt
- E1（SFT）：不使用 LoRA（或最小可训层）
- E2（SFT+LoRA）：主版本

### 7.4 产出
- outputs/adapters/sft_lora/
- outputs/logs/train_*.log

### 7.5 验收标准
- 在 eval 集上：
  - json_valid_rate >= 98%
  - schema_pass_rate >= 95%
  - field_micro_F1（对可监督字段）相对 Base 提升明显（建议 +20pp 作为目标）

---

## 8. Step 4：属性抽取评测与报告（Day4）

### 8.1 评测集定义（不需人工标注）
- 直接使用 Kaggle 的字段作为“银标”标签。
- 评测字段：只评 Kaggle 真实提供的字段集合（例如 color/category/gender/...）。

### 8.2 指标定义（完全确定）
- json_valid_rate
- schema_pass_rate
- per_field_acc / micro-F1 / macro-F1
- confusion（例如颜色混淆矩阵）

### 8.3 产出
- reports/extraction_base.json / extraction_sft.json / extraction_lora.json
- reports/extraction_compare.md（指标表 + 失败案例）

---

## 9. Step 5：可解释重排（B）（Day5）

### 9.1 任务定义
- 输入：用户约束（文本）+ 候选商品集合（从同类目随机采样或 embedding 召回）
- 输出：Top-K 结果 + 每条解释（引用其 JSON 字段作为证据）

### 9.2 约束解析（可规则化）
- 将用户文本映射到 schema 约束：
  - 颜色词典（blue/黑色/…）
  - 类别词典（连衣裙/dress/…）
  - 风格词典（通勤/度假/运动/…）

### 9.3 重排打分（写死）
- `score = base_score + α*match - β*conflict`
- match：字段相等记 1；unknown 记 0；支持多值

### 9.4 解释忠实度（必须自动校验）
- 解释中每个声称的字段必须能在商品 JSON 中找到对应值，否则计为 hallucination。

### 9.5 指标
- constraint_satisfaction@K
- explanation_faithfulness@K

---

## 10. Step 6：RL 列表级重排（C）（Day6–Day7）

### 10.1 环境（离线仿真，写死）
- 用户画像：从历史喜欢的属性分布采样（或从数据集中聚类得到风格原型）。
- 状态：`user_profile + candidate_pool + selected_list_summary`
- 动作：从候选池选择下一 item（构造 TopK 列表）

### 10.2 Reward 设计（写死）
- relevance：属性匹配分（与约束/偏好一致）
- redundancy penalty：与已选列表在 embedding 或属性上过于相似
- novelty bonus：选到长尾属性/新风格

### 10.3 基线对比（必须）
- Base Rank（只 relevance）
- MMR（多样性经典基线）
- RL Policy

### 10.4 指标
- NDCG@K（以 relevance 作为 proxy label）
- ILD@K（intra-list diversity）
- coverage@K（属性覆盖）

---

## 11. Step 7：Demo（Day8）

### 11.1 功能（必须实现）
- 模型选择：Base / LoRA
- 重排策略选择：Base / MMR / RL
- 输入：文本约束 + 可选参考图
- 输出：TopK 商品图 + 属性 JSON + 可解释理由（基于字段）

### 11.2 禁止项
- 不做线上部署
- 不引入外部搜索服务

---

## 12. 一键命令清单（保持不变）
```bash
python vlm_rec_project/src/00_prepare_kaggle_data.py
python vlm_rec_project/src/01_normalize_schema.py
python vlm_rec_project/src/02_make_sft_jsonl.py
python vlm_rec_project/src/03_train_sft_lora.py
python vlm_rec_project/src/04_eval_extraction.py
python vlm_rec_project/src/06_rerank_explain.py
python vlm_rec_project/src/07_train_rl_rerank.py
python vlm_rec_project/src/08_eval_rl.py
python vlm_rec_project/src/09_demo_gradio.py
```

---

## 13. 日计划（逐日必须完成的具体事项，按最短 8 日闭环）

> 说明：
> - 每天必须产出“可运行命令 + 产物路径”，写入 `vlm_rec_project/outputs/logs/dayX_commands.txt` 与 `dayX_outputs.txt`。
> - 每天结束必须用 20–30 条样本做 sanity check（不达标就原地修，不进入下一天）。

### Day1：Kaggle 数据落地 + items.jsonl 统一格式
- 必须完成：
  - 确定 Kaggle 数据集版本与字段列表，写入 `data/raw/kaggle_manifest.json`（包含 dataset slug、下载方式、字段释义、license 简述）。
  - 实现并跑通：`00_prepare_kaggle_data.py` → 产出 `data/processed/items.jsonl`。
  - 实现并跑通：`01_normalize_schema.py` → 产出 `reports/data_profile.md`。
- 必须验收：
  - items 数量 >= 10k（建议 20k+）。
  - 图片可读率 >= 99%（坏图剔除并统计）。
  - `attrs` 字段齐全（缺失用 `unknown`）。

### Day2：SFT 数据生成（0 人工标注）+ JSON 质量门槛
- 必须完成：
  - 实现并跑通：`02_make_sft_jsonl.py` 生成 `data/sft/train.jsonl` 与 `data/sft/eval.jsonl`。
  - 固化 `schema.py` 与 `prompts.py`（这两者从今天起尽量少改）。
- 必须验收：
  - train >= 20k、eval >= 1k（按算力可放大）。
  - JSON 可解析率 = 100%（生成时就校验）。
  - schema_pass_rate >= 99%（枚举合法性）。

### Day3：Base（zero-shot）评测基线 + 训练配置定稿
- 必须完成：
  - 实现 `04_eval_extraction.py` 的 Base 分支：用 Base 模型 + 固定 prompt 做抽取。
  - 固化评测字段集合（只评 Kaggle 有监督的字段）。
  - 写出训练配置文件/参数（max_length、batch、accum、lr、lora_r 等）并记录到 `outputs/logs/train_config.md`。
- 必须验收：
  - 输出 `reports/extraction_base.json`（逐样本）与 Base 指标汇总。
  - 至少导出 30 条失败 case（非法 JSON/枚举越界/胡编字段），写入 `reports/extraction_cases_base.md`。

### Day4：SFT（不加 LoRA 或最小可训）跑通一轮训练 + 评测
- 必须完成：
  - 实现并跑通：`03_train_sft_lora.py` 的 E1 配置（无 LoRA 或极小可训层）。
  - 训练至少跑到：1 epoch 或固定 steps（例如 2k steps）。
  - 跑评测：输出 `reports/extraction_sft.json`。
- 必须验收：
  - json_valid_rate、schema_pass_rate 相对 Base 提升明显（至少 +10pp 作为最低门槛）。
  - 保存 adapter/权重到 `outputs/adapters/`。

### Day5：SFT+LoRA 主版本训练（E2）+ 消融对比表
- 必须完成：
  - 跑 E2：projection + LoRA(LLM)（主版本）。
  - 训练完成后评测：`reports/extraction_lora.json`。
  - 生成对比报告：`reports/extraction_compare.md`（Base vs E1 vs E2）。
- 必须验收：
  - field_micro_F1 相对 Base 提升目标：+20pp（不足则调：prompt/枚举映射/训练步数/学习率）。
  - 失败案例分类统计（JSON 失败、枚举越界、字段错位、属性混淆）。

### Day6：B 模块落地：约束解析 + 可解释重排 + 忠实度校验
- 必须完成：
  - 实现 `06_rerank_explain.py`：
    - 约束解析（颜色/类别/风格词典）
    - 计算 match/conflict
    - 输出解释（必须引用 JSON 字段）
    - 自动校验 explanation_faithfulness
  - 生成 `reports/rerank_report.md`：指标 + 20 条样例。
- 必须验收：
  - constraint_satisfaction@10、faithfulness@10 两个指标都要有（即使数值一般，也要能解释原因）。

### Day7：C 模块落地：MMR 基线 + RL 训练 + 指标对比
- 必须完成：
  - 实现 MMR 重排（作为强基线）。
  - 实现 RL 环境与训练脚本：`07_train_rl_rerank.py`。
  - 评测脚本：`08_eval_rl.py` 输出 `reports/rl_report.md`。
- 必须验收：
  - RL vs Base/MMR：至少在 ILD@K、coverage@K 有提升，同时 NDCG@K 不显著崩（给出 trade-off 曲线或 λ/μ 扫描）。

### Day8：Demo 收口 + 可复现命令链
- 必须完成：
  - `09_demo_gradio.py` 跑通：模型选择（Base/LoRA），策略选择（Base/MMR/RL）。
  - 输出 TopK 图 + JSON + 解释。
  - 整理“一键命令清单”与复现步骤（写入本 MD 末尾或 `reports/README_run.md`）。
- 必须验收：
  - 录制 2–3 段演示 case（可选，但强烈建议）：
    - 纯文本约束
    - 约束 + 参考图
    - 开启多样性（MMR/RL）前后对比

---

## 14. 变更控制（避免偏差）
允许改动范围（只允许这三类）：
1) schema/枚举与映射规则（必须同步更新评测）
2) prompt 模板与 JSON 校验细节
3) 训练超参（batch/lr/epoch/lora_r）

禁止改动范围：
- 不把任务变成泛聊天
- 不把评测改为“主观感觉”
- 不引入额外不可复现的数据源

