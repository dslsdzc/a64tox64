# a64tox64 — ARM64 → x86-64 JIT 动态二进制翻译器

awine 核心引擎。将 ARM64 .so 指令动态翻译为 x86-64 机器码并执行。

## 论文

### ARM64 → x86-64 方向（你要写的）

### ARM64 → x86-64 方向

| 论文 | 年份 | 核心贡献 | 源文件 |
|------|------|----------|--------|
| **Arancini** "Hybrid DBT for Weak Memory Model Architectures" | **ASPLOS 2026** | AranciniIR 统一 IR + 混合静/动态翻译 + 形式化验证内存模型，up to 5× vs QEMU | `papers/arancini_asplos2026.pdf` |
| **Liu et al.** "Exploiting SIMD Asymmetry in ARM-to-x86 DBT" | 2019 | saSLP 寄存器映射，1.6× 提速，97% spill 减少 | `papers/liu2019_simd_asymmetry.pdf` |
| **You, Lin, Yang** "Translating AArch64 FP Instruction Set to x86-64" | 2019 | mc2llvm 扩展支持 AArch64 浮点，47% 原生性能，2.92× vs QEMU | `papers/you2019_aarch64_fp.pdf` |
| **Liu, Chen, Yang, Hsu** "mc2llvm: DBT for Multi-Threaded Programs" | 2014 | 共享代码缓存 + 两级地址映射表，8.8× vs QEMU | — |
| **Xie et al.** "Peephole Optimization in DBT" | 2024 | 死变量分析 + 指令融合，1.52× 提速 | `papers/xie2024_peephole.pdf` |

### 通用 DBT 架构（方向无关，可借鉴）

| 论文 | 年份 | 核心贡献 | 源文件 |
|------|------|----------|--------|
| **Spink & Franke** "Accelerating Shared Library Execution in a DBT" | **LCTES 2024** | mixed-mode 共享库执行 + IDL 自动生成 stub，2.7-6.3× 平均加速，up to 28× | `papers/spink2024_shared_lib_dbt.pdf` |
| **Spink et al.** "Efficient Code Generation in Region-Based DBT" | 2014 | 区域划分 + 链式翻译块，264% vs QEMU | — |
| **Mavrogeorgis** "Simplifying Heterogeneous Migration" (PhD) | 2021 | 统一栈布局 + LLVM 后端修改，≤6% 开销 | `papers/mavrogeorgis2021_thesis.html` |
| **CrossDBT** "LLVM-Based User-Level DBT" | 2022 | LLVM IR 中介，3.3× vs QEMU | — |

### 反方向（x86 → ARM64，设计可镜像）

| 论文 | 年份 | 核心贡献 |
|------|------|----------|
| **Risotto** "DBT for Weak Memory Model" | 2022 | Weak memory model 一致性 |
| **Hong et al.** "Processor-Tracing Guided Region Formation" | 2019 | Intel PT 硬件 trace 指导 DBT 区域划分 |

## 开源实现参考

- **Box64** (`refs/box64/`) — x86_64 → ARM64 JIT。DynaRec 模块：解码、寄存器映射、代码缓存
- **FEX** (`refs/fex/`) — x86/x86-64 → ARM64。IR 中间表示、thunklibs 系统
- **mc2llvm** — LLVM IR 作为中介的 ARM→x86 翻译器（源码需单独获取）

## 实现状态

a64tox64 核心引擎（Zig 实现，~2200 行，37 个测试全部通过）：

### IR 层
- 16 字节定长 IROp（packed struct），覆盖 ALU/内存/控制流/SIMD 类别
- IRBuffer：可增长的 IR 操作缓冲区

### 已实现的 ARM64 指令

| 类别 | 指令 |
|------|------|
| ALU 立即数 | ADD, SUB (含 LSL #12 移位) |
| 移位寄存器 | LSL, LSR, ASR (寄存器移位量) |
| 立即数移位 | LSL, LSR, ASR (立即数移位量) |
| 整数 ALU | ADD, SUB, AND, ORR, EOR, BIC, ORN, EON |
| 乘除 | MUL, MNEG, SDIV, UDIV |
| 移动 | MOVZ, MOVN, MOVK, NEG |
| 比较 | CMP, CMN, CCMP |
| 条件选择 | CSEL, CSINC, CSINV, CSNEG |
| 位域 | UBFM (UXTB/UXTH/LSR), SBFM (SXTB/SXTH/ASR) |
| 加载/存储 | LDR/STR (立即数/寄存器偏移), LDRB/LDRH, LDUR/STUR, LDP/STP (含 writeback), LDR literal |
| PC 相对寻址 | ADR, ADRP |
| 分支 | B, BL, BR, BLR, RET, B.cond |
| 系统 | SVC, NOP |

### 实现方式

IR-first 架构：ARM64 指令 → 解码 → IR (中间表示) → x86-64 机器码。复杂指令在 IR 层分解为基本操作（如 BIC → NOT + AND）。



## 方向对比

```
a64tox64:  ARM64 → x86-64  ✓
Box64:     x86-64 → ARM64  ✗ 但架构镜像可用
FEX:       x86 → ARM64     ✗ 同上
mc2llvm:   ARM → x86       ✓ 最近似的学术实现

共享的设计问题：
  - 基本块划分 / 区域形成
  - 寄存器映射（x0-x30 → 宿主 regs + spill）
  - 翻译块缓存管理 + 链式跳转
  - 间接跳转处理（查找表）
  - 自修改代码检测（SMC）
  - 系统调用转换
```
