# Ablation Study Plan — CogExtract

## 实验目标

验证三个认知模块各自的贡献：
- **MHG** (Multi-Hypothesis Generation): K=3 多候选生成
- **CPSV** (Cross-Page Structural Verification): 6 维跨页验证
- **UIR** (Unified Iterative Reflection): 统一迭代反思

## 实验矩阵

| 编号 | 名称 | MHG | CPSV | UIR | 等价 | 代码改动 |
|------|------|-----|------|-----|------|----------|
| A0 | CogExtract (完整) | ✓ K=3 | ✓ | ✓ R=3 | v8 | 无 |
| A1 | w/o MHG | ✗ K=1 | ✓ | ✓ R=3 | — | `pipeline_cog.py`: 截断 candidates 为 1 |
| A2 | w/o CPSV | ✓ K=3 | ✗ | ✓ R=3 | — | `pipeline_cog.py`: 跳过验证，直接用首个候选 |
| A3 | w/o UIR | ✓ K=3 | ✓ | ✗ R=0 | — | `pipeline_cog.py`: `MAX_REFLECTION_ROUNDS = 0` |
| A7 | VGS Baseline | ✗ K=1 | ✗ | ✗ | vgs | 已有 `vgs/pipeline.py` |

## 评测数据集

- LiveWeb-IE 数据集（VGS 论文数据）
- 35,758 个抽取实例，4 种页面类型（Type 1–4）
- 与主实验完全相同的数据划分

## 运行流程

```bash
# 1. 各消融实验独立运行
python ablation/run_ablation.py --ablation A0   # 完整系统（已有 v8 结果）
python ablation/run_ablation.py --ablation A1   # 去掉 MHG
python ablation/run_ablation.py --ablation A2   # 去掉 CPSV
python ablation/run_ablation.py --ablation A3   # 去掉 UIR

# 2. 汇总结果
python ablation/aggregate_results.py
```

## 输出目录

每个消融实验输出到独立的子目录：
```
output/
├── ablation_A0/   (即 cog_v8/)
├── ablation_A1/
├── ablation_A2/
├── ablation_A3/
└── vgs_baseline/  (已有)
```

每个目录包含：
- `results.json`: 逐样本抽取结果
- `metrics.json`: 按 type 和 overall 汇总的 P/R/F1

## 结果呈现（论文表格）

最终在论文中呈现一张消融实验表：

| Method | P (%) | R (%) | F1 (%) | ΔF1 |
|--------|-------|-------|--------|-----|
| Full CogExtract (A0) | xx.xx | xx.xx | xx.xx | — |
| w/o MHG (A1) | xx.xx | xx.xx | xx.xx | −x.xx |
| w/o CPSV (A2) | xx.xx | xx.xx | xx.xx | −x.xx |
| w/o UIR (A3) | xx.xx | xx.xx | xx.xx | −x.xx |
| VGS Baseline (A7) | 60.09 | 76.29 | 63.44 | −11.60 |

## 关键注意事项

1. **缓存复用**：Stage 1-3（属性识别、视觉锚定、元素精确定位）的结果在所有消融中相同，可直接复用缓存，只需重跑 Stage 4 及后续。
2. **公平对比**：所有消融使用相同的 seed/validation 页面划分、相同的 ground truth、相同的评测代码。
3. **运行成本**：A1 只需改 K 值，A3 只需改 rounds 数，A2 改动稍大（跳过验证逻辑）。预计 A1/A3 可在 2-3 小时内跑完，A2 约 1-2 小时。
