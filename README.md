# a64tox64 — ARM64 → x86-64 JIT 动态二进制翻译器

awine 核心引擎。将 ARM64 .so 指令动态翻译为 x86-64 机器码并执行。

## 论文

### ARM64 → x86-64 方向（你要写的）

| 论文 | 年份 | 核心贡献 | 源文件 |
|------|------|----------|--------|
| **Liu et al.** "Exploiting SIMD Asymmetry in ARM-to-x86 DBT" | 2019 | saSLP 寄存器映射，1.6× 提速，97% spill 减少 | `papers/liu2019_simd_asymmetry.pdf` |
| **You, Lin, Yang** "Translating AArch64 FP Instruction Set to x86-64" | 2019 | mc2llvm 扩展支持 AArch64 浮点，47% 原生性能，2.92× vs QEMU | `papers/you2019_aarch64_fp.pdf` |
| **Liu, Chen, Yang, Hsu** "mc2llvm: DBT for Multi-Threaded Programs" | 2014 | 共享代码缓存 + 两级地址映射表，8.8× vs QEMU | — |
| **Xie et al.** "Peephole Optimization in DBT" | 2024 | 死变量分析 + 指令融合，1.52× 提速 | `papers/xie2024_peephole.pdf` |

### 通用 DBT 架构（方向无关，可借鉴）

| 论文 | 年份 | 核心贡献 |
|------|------|----------|
| **Spink et al.** "Efficient Code Generation in Region-Based DBT" | 2014 | 区域划分 + 链式翻译块，264% vs QEMU |
| **Mavrogeorgis** "Simplifying Heterogeneous Migration" (PhD) | 2021 | 统一栈布局 + LLVM 后端修改，≤6% 开销 | 
| **CrossDBT** "LLVM-Based User-Level DBT" | 2022 | LLVM IR 中介，3.3× vs QEMU |

### 反方向（x86 → ARM64，设计可镜像）

| 论文 | 年份 | 核心贡献 |
|------|------|----------|
| **Risotto** "DBT for Weak Memory Model" | 2022 | Weak memory model 一致性 |
| **Hong et al.** "Processor-Tracing Guided Region Formation" | 2019 | Intel PT 硬件 trace 指导 DBT 区域划分 |

## 开源实现参考

- **Box64** (`refs/box64/`) — x86_64 → ARM64 JIT。DynaRec 模块：解码、寄存器映射、代码缓存
- **FEX** (`refs/fex/`) — x86/x86-64 → ARM64。IR 中间表示、thunklibs 系统
- **mc2llvm** — LLVM IR 作为中介的 ARM→x86 翻译器（源码需单独获取）

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
