---
name: extend-and-report
description: 添加扩展测试用例，导出 CSV，生成最终报告
---

# 扩展测试与报告

**目的**：根据已确认的测试计划，添加扩展用例，运行全部测试，生成 CSV 和报告。

---

## 前置条件

- 已完成 `/write-and-verify`（baseline 已通过）
- `logs/test_plan.md` 中有确认的测试计划

---

## 执行步骤

### 步骤 1：更新 TEST_CONFIGS

根据 `logs/test_plan.md`，将扩展测试添加到 `TEST_CONFIGS`：

```python
TEST_CONFIGS = [
    # Baseline（已验证通过）
    {"name": "baseline", "batch_size": 4, "seq_len": 128, "hidden_size": 256},
    
    # 形状覆盖
    {"name": "small", "batch_size": 1, "seq_len": 16, "hidden_size": 64},
    {"name": "large", "batch_size": 8, "seq_len": 512, "hidden_size": 512},
    
    # 数据类型
    {"name": "fp16", "batch_size": 4, "seq_len": 128, "hidden_size": 256, "dtype": torch.float16},
    {"name": "bf16", "batch_size": 4, "seq_len": 128, "hidden_size": 256, "dtype": torch.bfloat16},
    
    # 边界条件
    {"name": "batch_1", "batch_size": 1, "seq_len": 1, "hidden_size": 64},
]
```

### 步骤 2：运行全部测试

```bash
python test_<opname>.py
```

### 步骤 3：导出 CSV

文件：`results_<opname>_<timestamp>.csv`

```csv
case_name,batch_size,seq_len,hidden_size,dtype,scenario,cpu_status,npu_status,max_diff,pass
baseline,4,128,256,fp32,baseline,OK,OK,1.2e-7,PASS
small,1,16,64,fp32,shape,OK,OK,8.5e-8,PASS
large,8,512,512,fp32,shape,OK,OK,2.1e-6,PASS
fp16,4,128,256,fp16,dtype,OK,OK,3.2e-3,PASS
bf16,4,128,256,bf16,dtype,OK,OK,1.5e-2,PASS
batch_1,1,1,64,fp32,boundary,OK,OK,4.7e-8,PASS
```

### 步骤 4：生成报告

文件：`logs/test_report.md`

---

## 报告格式

```markdown
# 测试报告：<算子名>

## 概览

| 项目 | 值 |
|------|-----|
| 算子 | <opname> |
| 时间 | 2026-02-05 |
| 总用例 | N |
| 通过 | X (xx%) |
| 失败 | Y |

## 覆盖情况

### 形状覆盖
| 场景 | 配置 | 结果 |
|------|------|------|
| small | B=1,S=16,H=64 | ✅ |
| large | B=8,S=512,H=512 | ✅ |

### 数据类型覆盖
| dtype | 结果 | max_diff |
|-------|------|----------|
| fp32 | ✅ | 1.2e-7 |
| fp16 | ✅ | 3.2e-3 |
| bf16 | ✅ | 1.5e-2 |

## 失败用例分析

（若有失败用例，详细列出）

| 用例 | 问题 | max_diff | 建议 |
|------|------|----------|------|
| xxx | numeric_mismatch | 0.05 | 放宽容差或检查算法 |

## 结论

**PASS** / **FAIL**（需修复后复测）

## 附件

- CSV 结果：`results_<opname>_xxx.csv`
- 测试文件：`test_<opname>.py`
```

---

## CSV 导出代码

在 `test_<opname>.py` 中添加：

```python
import csv
from datetime import datetime

def export_csv(results, opname):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"results_{opname}_{timestamp}.csv"
    
    fieldnames = ["case_name", "batch_size", "seq_len", "hidden_size", 
                  "dtype", "scenario", "cpu_status", "npu_status", "max_diff", "pass"]
    
    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "case_name": r["name"],
                "batch_size": r.get("batch_size", ""),
                "seq_len": r.get("seq_len", ""),
                "hidden_size": r.get("hidden_size", ""),
                "dtype": str(r.get("dtype", "fp32")),
                "scenario": r.get("scenario", ""),
                "cpu_status": "OK",
                "npu_status": "OK" if r["status"] != "SKIP" else "SKIP",
                "max_diff": f"{r['diff']:.2e}" if r['diff'] else "",
                "pass": r["status"],
            })
    
    print(f"CSV exported: {filename}")
    return filename
```

---

## 用户交互

### 测试完成后

```
✅ 扩展测试完成

📊 结果摘要：
- 总用例：6
- 通过：6 (100%)
- 失败：0

📁 产物：
- CSV：results_layernorm_20260205_091800.csv
- 报告：logs/test_report.md
- 测试文件：test_layernorm.py

请回复：
- "完成" → 结束测试流程
- "查看失败详情" → 展示失败用例
- "重跑 xxx" → 重新测试特定用例
```

### 若有失败

```
⚠️ 部分测试失败

📊 结果摘要：
- 总用例：6
- 通过：4 (67%)
- 失败：2

失败用例：
1. fp16 - max_diff=0.08，超出容差 atol=0.01
   建议：放宽 fp16 容差到 0.1

2. large - shape_mismatch
   期望：(8,512,512)，实际：(8,512,256)
   建议：检查 hidden_size 参数传递

请回复：
- "放宽 fp16 容差到 0.1" → 调整后重跑
- "查看 large 详情" → 展示详细对比
- "忽略失败，完成" → 生成报告（标记失败）
```
