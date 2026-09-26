# CogExtract — Slim Reproduction Package

本仓库用于复现 **"CogExtract: Credit Assignment for Visual-Language Web
Information Extraction"**（ICLR 2027 投稿）的**主实验（Cog）**与**消融实验**。
包含自包含的代码 + 数据 + 页面缓存快照。

## 裁剪说明

本仓库是完整补充材料包的精简版，已移除以下内容：

- **`analyses/`** — 零 API 消耗的离线机制复现分析（论文章节 §5 / 附录）未打包。
- **`expected/`** — 归档的基线 / 期望数值对照表未打包。
- **`cxs`（Contrastive XPath Synthesis）方法及其文件**已删除：
  `vgs/cxs_pipeline.py`、`vgs/contrastive_synthesis.py`，以及 `vgs/prompts.py` 中的 CXS 提示词块。
- **`cache/pages`** 精简为离线 `quick` 运行所需的 **8 个归档页面**（每个页面保留 `page.html` + 截图）。

其余目录布局与原仓库一致，默认路径均在包根目录内解析。

## 目录结构

```
CogExtract/
├── run_all.sh            # 统一入口：quick | full | smoke
├── main.py               # 单条运行 CLI
├── configs/              # VGSConfig；cache_dir 默认指向 <root>/cache/pages
├── utils/                # DataLoader、CachedBrowserManager、Evaluator、LLM 客户端
├── cog/                  # Cog 流水线（多候选生成 + 跨页验证 + 失败驱动重生成）
├── vgs/                  # VGS 基线（视觉定位抽取器）
├── baselines/            # CoT / Reflexion 提示词基线
├── ablations/            # 5 个消融变体 + 批量驱动
├── scripts/              # 多模型批量运行、严格评测、预缓存、SWDE 等
├── data/LiveWeb_IE/      # 完整 LiveWeb-IE 数据集（约 5.6 MB）
└── cache/pages/          # 8 个归档页面（page.html + 截图），供离线 quick 使用
```

## 环境准备

```bash
pip install -r requirements.txt
playwright install chromium        # 离线图片重生成 / 在线抓取时需要
export DASHSCOPE_API_KEY=sk-...    # 填入你自己的 Key，仓库不含任何凭证
```

Python 3.10+。`smoke` 模式无需网络。`configs/config.py` 从环境变量读取
API Key（`DASHSCOPE_API_KEY`，可选 `_2/_3` 用于限流轮转池）。

## 运行模式

### 0. 离线自检（无需 API Key）— 约 5 秒

```bash
./run_all.sh smoke
```

校验数据集加载、8 页缓存覆盖、每个缓存页是否都带 Set-of-Mark 截图。

### 1. 小规模复现（含 8 页缓存，需 API Key）— 约 10-15 分钟

```bash
./run_all.sh quick                        # 默认模型：qwen3-vl-8b-instruct
MODEL=qwen3.7-plus ./run_all.sh quick     # 论文主用开源权重模型
```

在 4 种任务类型上各跑 3 组 × 2 个 URL（每方法 24 个样本），运行 **Cog** 与
**VGS** 基线。所有页面从 `cache/pages/` 读取，不发起在线浏览。分类型指标存于
`quick_out/<model>/`。

### 2. 完整复现（需 API Key，每模型约数小时）

```bash
./run_all.sh full
```

跨 `scripts/run_multi_model.py` 中的模型列表，运行 Cog（主）、VGS 基线、CoT /
Reflexion 基线及全部消融。归档缓存仅覆盖 8 个 quick 页面，完整运行会在线抓取页面
（需 playwright + chromium）；站点自归档以来可能已漂移——这本身就是论文研究的现象。

分家族命令：

```bash
python3 scripts/run_multi_model.py --models qwen3.7-plus --concurrency 3   # 主实验 (Cog)
python3 scripts/run_vgs_multi_model.py --models qwen3.7-plus               # VGS
python3 scripts/run_baselines.py --models qwen3.7-plus --methods cot reflexion
python3 ablations/run_ablation.py multi_only --model qwen3.7-plus          # 消融
```

## 注意事项

- **Checkpoint 警告**：`main.py eval` 始终写入同一目录（`experiments/cog_v8`），
  且 checkpoint **未按模型 / 方法 / 任务类型隔离**——`results_checkpoint.json`
  （Cog，按 `(group,url)` 键）与 `phase1_checkpoint.json`（VGS，按组索引
  0/1/2 键，会跨任务类型冲突）。残留 checkpoint 会被**静默复用**并重评分，导致
  F1 失真。因此 `run_all.sh quick` 每次运行前会清空整个输出目录。若手动调用
  `main.py eval`，务必在不同模型 / 方法 / 类型之间执行 `rm -rf experiments/cog_v8`。
- **离线图片重生成**：8 个 quick 页面已带截图；若需为其他（在线抓取的）页面
  离线重生成图片，运行 `python3 scripts/precache_stage3b.py`。
- **指标约定**：所有 P/R/F1 均为各样本得分的宏平均（%），由 `utils/evaluator.py` 计算。
