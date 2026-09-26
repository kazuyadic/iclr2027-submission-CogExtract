# A1: w/o Multi-Hypothesis Generation

## 目的

验证 **多假设生成 (MHG)** 的贡献：将 K=3 候选缩减为 K=1（仅保留首个候选）。

## 预期效果

- CPSV 退化为二值判断（pass/fail），无法从多个候选中择优
- UIR 仍然可用，但每轮反思只能生成 1 个候选（或仍生成 3 个但只保留第 1 个）
- 预测 F1 下降 3-5 点，尤其在 Type 2/3（列表页）上明显

## 代码改动

### 方案 A：Pipeline 层截断（推荐，改动最小）

文件：`cog/pipeline_cog.py`

在 `_run_inner` 方法中，`generate_candidates` 返回后立即截断：

```python
# 原代码（约 line 117-120）
candidates, gen_metadata = await self.generator.generate_candidates(
    seed_page, attribute, screenshot_dir,
)

# ↓ 新增一行
candidates = candidates[:1]   # A1: 只保留第 1 个候选
```

### 方案 B：Generator 层修改（备选）

文件：`cog/generator_cog.py`

在 `generate_candidates` 中修改 prompt，只请求 1 个 XPath，并将 K=1 硬编码。
但方案 A 更简单，且 UIR 反思时仍会生成 K=3（用于探索更多解），截断只在初始生成时。

## 注意事项

- **UIR 反思不改**：UIR 的 `_unified_reflection` 方法仍会生成 K=3 新候选，这是 UIR 的能力，不是 MHG 的。
  - 如果也想去掉 UIR 中的多候选：将 `_unified_reflection` 返回的 candidates 也截断为 1
  - 但更合理的做法是：MHG 只影响初始生成，UIR 保留多候选（因为 UIR 本身就是"反思后重新发散"）
- **Stage 1-3 完全复用缓存**，无需重新跑

## 评测命令

```bash
python ablation/run_ablation.py --ablation A1
```

## 预期输出

`output/ablation_A1/metrics.json` 包含按 type 和 overall 的 P/R/F1。
