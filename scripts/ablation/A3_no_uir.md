# A3: w/o Unified Iterative Reflection

## 目的

验证 **统一迭代反思 (UIR)** 的贡献：保留 MHG + CPSV，但去掉反思循环。
当所有候选都未通过 CPSV 时，直接使用得分最高的失败候选（无修复）。

## 预期效果

- 失去自修复能力，所有初始失败的候选无法被补救
- Type 3（结构化列表页）影响最大（初始候选频繁失败，最需要反思）
- 预测 F1 下降 8-12 点（Type 3 上可能下降 20+ 点）

## 代码改动

文件：`cog/pipeline_cog.py`

### 改动：设置 MAX_REFLECTION_ROUNDS = 0

```python
class CogPipeline(CogBasePipeline):
    # 原值：MAX_REFLECTION_ROUNDS = 3
    MAX_REFLECTION_ROUNDS = 0   # A3: 禁用反思
```

### 逻辑分析

`_verify_and_refine` 中的循环：
```python
for round_num in range(self.MAX_REFLECTION_ROUNDS + 1):
    # round_num = 0 only (range(1))
    
    scores = self.verifier.score_candidates(...)
    
    # 如果有候选通过 → 正常返回（与完整系统相同）
    for s in scores:
        if s.passes_basic:
            return s.xpath, trace
    
    # 如果没有候选通过 → 进入反思判断
    all_failed.extend(scores)
    
    if round_num >= self.MAX_REFLECTION_ROUNDS:
        break  # ← 立即退出，不反思

# 回退到得分最高的失败候选
best = max(all_failed, key=lambda s: s.total_score)
trace.final_xpath = best.xpath
return trace.final_xpath, trace
```

**行为**：
1. 生成 K=3 候选
2. CPSV 评分
3. 有候选通过 → 使用（与完整系统一致）
4. 无候选通过 → **直接使用得分最高者**（完整系统会进入 UIR 反思）

## 注意事项

- 这是最干净的消融：只改一个参数，不改任何其他逻辑
- MHG 和 CPSV 完全保留，能验证"生成多候选 + 择优"的价值
- 如果 A3 与 A0 差距大，说明 UIR 的自修复能力是关键
- 如果 A3 与 A0 差距小，说明 MHG + CPSV 已经足够好，UIR 贡献有限

## 评测命令

```bash
python ablation/run_ablation.py --ablation A3
```

## 预期输出

`output/ablation_A3/metrics.json` 包含按 type 和 overall 的 P/R/F1。

## 预期结果分析

| Type | A0 F1 | A3 F1 (预测) | Δ | 分析 |
|------|-------|-------------|---|------|
| T1 | 74.07 | ~72 | −2 | MHG+CPSV 足够，UIR 贡献小 |
| T2 | 77.21 | ~73 | −4 | 列表页需要反思修复部分 case |
| T3 | 69.44 | ~50 | **−19** | 结构化列表最依赖 UIR |
| T4 | 57.58 | ~53 | −4 | 图像抽取 UIR 效果有限 |
| Overall | 75.04 | ~70 | −5 | UIR 总体贡献约 5 F1 |
