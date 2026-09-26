# A2: w/o Cross-Page Structural Verification

## 目的

验证 **跨页结构验证 (CPSV)** 的贡献：跳过 6 维评分，直接使用 MHG 生成的第一个候选。

## 预期效果

- 过拟合 XPath（含页面特定文本谓词）无法被检测，直到应用于下游页面才失败
- UIR 只在抽取结果为空时触发（信号更弱）
- 预测 F1 下降 5-8 点，Type 2（列表页）影响最大

## 代码改动

文件：`cog/pipeline_cog.py`

### 改动 1：`_verify_and_refine` 方法

替换为直接取首个候选 + 简单空值触发 UIR：

```python
def _verify_and_refine(
    self, attribute, query, candidates, seed_html, val_html,
) -> tuple[str, ReflectionTrace]:
    """A2 ablation: skip CPSV, use first candidate."""
    trace = ReflectionTrace(
        attribute=attribute,
        initial_xpath=candidates[0] if candidates else "",
        final_xpath="",
    )

    if not candidates:
        trace.final_xpath = ""
        trace.success = False
        return "", trace

    # 直接用第一个候选，不做跨页验证
    xpath = candidates[0]
    trace.final_xpath = xpath
    trace.success = True
    trace.total_rounds = 0
    return xpath, trace
```

### 改动 2（可选）：保留 UIR 但触发条件更弱

如果希望 UIR 仍能工作（但触发条件变成"抽取结果为空"而非"CPSV 不通过"），可以在外层加一个简单检查：

```python
# 在 _run_inner 中，抽取后检查结果是否为空
final_result = await self.executor.execute_on_page(
    seed_page, xpaths[attribute], seed_url,
)
values[attribute] = final_result.values if final_result.success else []

# A2 的弱触发：如果结果为空，且还有 UIR，则触发反思
if not values[attribute] and self.MAX_REFLECTION_ROUNDS > 0:
    # 调用 UIR（但 UIR 内部仍需要 CPSV 来评分新候选...）
    # 这里存在设计冲突：UIR 依赖 CPSV 来验证新候选
    pass
```

**设计冲突说明**：UIR 内部依赖 CPSV 来验证反思产生的新候选。如果完全去掉 CPSV，UIR 也失去验证信号。因此 A2 有两种子变体：

| 子变体 | 说明 | 推荐 |
|--------|------|------|
| A2a | 去掉 CPSV，UIR 也失效（无验证信号）→ 等价于 K=3 取首 + 无修复 | **推荐** |
| A2b | 去掉 CPSV 的初始选择，但 UIR 内部仍用 CPSV 验证反思候选 | 不推荐（语义不清晰） |

**结论：A2 = 直接取 candidates[0]，UIR 不触发（因为 CPSV 不存在，无法判断"失败"）。**

## 注意事项

- 这等价于：MHG 生成 3 个候选 → 直接取第 1 个 → 应用
- 与 A1 的区别：A1 仍有 CPSV 择优，A2 没有
- 与 VGS 的区别：VGS 只生成 1 个候选且无 CPSV；A2 生成 3 个但只用第 1 个

## 评测命令

```bash
python ablation/run_ablation.py --ablation A2
```

## 预期输出

`output/ablation_A2/metrics.json` 包含按 type 和 overall 的 P/R/F1。
