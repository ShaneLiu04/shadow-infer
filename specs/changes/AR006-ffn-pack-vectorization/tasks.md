# [AR006] 任务跟踪

| 字段 | 内容 |
|------|------|
| AR 编号 | AR006 |
| 关联 srs.md | ./srs.md |
| 关联 design.md | ./design.md |
| 创建日期 | 2026-10-19 |

## 任务列表

| ID | 任务描述 | 依赖 | 状态 | 备注 |
|----|---------|------|------|------|
| T001 | 基线取证：FFN pack 逐通道耗时（1024 通道 116.6 ms） | - | passing（已核验） | 2026-10-19（ar006_bench.py） |
| T002 | FR6.1 分组向量化 `_do_pack` + `_do_pack_reference` + 字节 helper | T001 | passing | 2026-10-19：0.072x；golden 逐位一致 |
| T003 | FR6.2 测试（UT-6.1~6.6）+ 全量回归 + 五件套 | T002 | passing | 2026-10-19：UT-6.1~6.5；FFN 回归 65 passed |
| T004 | FR6.3 README v3.8.0 + 版本同步 + ROADMAP | T003 | pending | |
| T005 | Review + ST + 归档 + Gitee 推送 | T004 | pending | 规则二 |

## 进度记录

### 2026-10-19 T001 基线取证

- FFN pack 微基准（CPU，3 次中位）：256×256 27.96 ms；1024×256 116.62 ms。
- 4 步小模型管线 cProfile：`PackedFFNWeight._do_pack` 自耗时 0.071 s（8192 次 `select`），为 AR005 后最大剩余项。
- 复现脚本：`specs/changes/AR006-ffn-pack-vectorization/ar006_bench.py`（seed=42）。
