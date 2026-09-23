//! x86-64 machine code emitter.
//!
//! Translates IR ops into x86-64 machine code. Handles the 3-operand
//! to 2-operand mapping by emitting MOV+ALU when dst ≠ src0.

const std = @import("std");
const Ir = @import("ir.zig");
const IROp = Ir.IROp;
const Tag = Ir.Tag;
const Peephole = @import("peephole.zig");

pub const X86Reg = enum(u4) {
    rax = 0, rcx = 1, rdx = 2, rbx = 3,
    rsp = 4, rbp = 5, rsi = 6, rdi = 7,
    r8 = 8, r9 = 9, r10 = 10, r11 = 11,
    r12 = 12, r13 = 13, r14 = 14, r15 = 15,
};

/// Extended register mapping: ARM64 x0-x13 → x86-64 host registers.
/// Uses all 14 available GP registers (8 call-clobbered + 6 callee-saved).
/// x14-x30 → spill to RAX (temporary).
pub const DefaultMapping: [31]?X86Reg = .{
    .rdi, .rsi, .rdx, .rcx, .r8, .r9, .r10, .r11, // x0-x7: call-clobbered
    .rax, // x8 → RAX (syscall number)
    .rbx, .rbp, .r12, .r13, null, null,            // x9-x14: callee-saved
    null, null, null, null, null, null, null, null, // x15-x22: spill
    null, null, null, null, null, null, null, null, // x23-x30: spill
};

pub const RegisterMap = [31]?X86Reg;

pub const EmitContext = struct {
    buf: []u8,
    offset: usize,
    regmap: *const RegisterMap,

    pub fn init(buf: []u8, regmap: *const RegisterMap) EmitContext {
        return .{ .buf = buf, .offset = 0, .regmap = regmap };
    }

    pub fn byte(ctx: *EmitContext, b: u8) void {
        ctx.buf[ctx.offset] = b;
        ctx.offset += 1;
    }

    pub fn bytes(ctx: *EmitContext, data: []const u8) void {
        @memcpy(ctx.buf[ctx.offset..][0..data.len], data);
        ctx.offset += data.len;
    }

    pub fn rex(ctx: *EmitContext, w: bool, r: u4, x_: u4, b: u4) void {
        var val: u8 = 0x40;
        if (w) val |= 0x08;
        if (r & 0x08 != 0) val |= 0x04;
        if (x_ & 0x08 != 0) val |= 0x02;
        if (b & 0x08 != 0) val |= 0x01;
        if (val != 0x40) ctx.byte(val);
    }

    pub fn modrm(ctx: *EmitContext, mod_: u2, reg: u4, rm: u4) void {
        const m: u8 = @intCast(mod_);
        const r: u8 = reg;
        const r2: u8 = rm;
        ctx.byte(m << 6 | r << 3 | r2);
    }

    pub fn disp32(ctx: *EmitContext, disp: i32) void {
        ctx.bytes(std.mem.asBytes(&disp));
    }
};

fn mapReg(regmap: *const RegisterMap, arm_reg: u16) X86Reg {
    if (arm_reg >= 31) return .rax; // XZR → RAX as sentinel
    return regmap[arm_reg] orelse .r14;
}

/// Returns true if the ARM64 register is XZR (the zero register).
/// XZR reads as zero; writes to it are discarded.
fn isXzr(arm_reg: u16) bool {
    return arm_reg >= 31;
}

// ── Register copy ──────────────────────────────────────────────────

fn emitMovReg(ctx: *EmitContext, dst: X86Reg, src: X86Reg) void {
    if (dst == src) return;
    ctx.rex(true, @intFromEnum(src), 0, @intFromEnum(dst));
    ctx.byte(0x89);
    ctx.modrm(0b11, @intFromEnum(src), @intFromEnum(dst));
}

// ── 3-operand ALU: emit "dst = src0 op src1" ──────────────────────

fn threeOp(
    ctx: *EmitContext,
    dst: X86Reg,
    src0: X86Reg,
) void {
    if (dst == src0) return;
    emitMovReg(ctx, dst, src0);
}

// ── ALU emission ───────────────────────────────────────────────────

fn emitAdd(ctx: *EmitContext, op: IROp) void {
    // XZR dest = CMN: the result is discarded but the flags must be set.
    // Compute into R14 (scratch; only clobbered by BR/BLR L1 checks, which
    // come at region end, and by the .cond tail, which runs after this).
    const dst = if (isXzr(op.dest)) X86Reg.r14 else mapReg(ctx.regmap, op.dest);
    const src0_is_xzr = isXzr(op.src0);
    const cond = op.flags;

    // If flags contains a condition code (CSEL), emit CMOV instead
    if (cond != 0 and op.src1 != 0x1F and op.imm == 0) {
        emitCSel(ctx, dst, mapReg(ctx.regmap, op.src0), mapReg(ctx.regmap, op.src1), cond);
        return;
    }

    if (op.imm != 0) {
        if (src0_is_xzr) {
            emitMovCst(ctx, dst, op.imm);
            if (isXzr(op.dest)) {
                // CMN XZR, #imm: MOV sets no flags — force flag-setting ADD
                ctx.rex(true, 0, 0, @intFromEnum(dst));
                ctx.byte(0x83);
                ctx.modrm(0b11, 0, @intFromEnum(dst));
                ctx.byte(0); // ADD r/m64, imm8 0
            }
        } else {
            const src0 = mapReg(ctx.regmap, op.src0);
            threeOp(ctx, dst, src0);
            if (op.imm <= 127) {
                ctx.rex(true, 0, 0, @intFromEnum(dst));
                ctx.byte(0x83);
                ctx.modrm(0b11, 0, @intFromEnum(dst));
                ctx.byte(@truncate(op.imm));
            } else {
                ctx.rex(true, 0, 0, @intFromEnum(dst));
                ctx.byte(0x81);
                ctx.modrm(0b11, 0, @intFromEnum(dst));
                ctx.bytes(std.mem.asBytes(&@as(i32, @bitCast(op.imm))));
            }
        }
    } else if (op.src1 != 0x1F) {
        const src0 = mapReg(ctx.regmap, op.src0);
        threeOp(ctx, dst, src0);
        const src1 = mapReg(ctx.regmap, op.src1);
        ctx.rex(true, @intFromEnum(src1), 0, @intFromEnum(dst));
        ctx.byte(0x01);
        ctx.modrm(0b11, @intFromEnum(src1), @intFromEnum(dst));
    } else if (isXzr(op.dest)) {
    } else {
        ctx.rex(true, @intFromEnum(dst), 0, @intFromEnum(dst));
        ctx.byte(0x31);
        ctx.modrm(0b11, @intFromEnum(dst), @intFromEnum(dst));
    }
}

fn emitCSel(ctx: *EmitContext, dst: X86Reg, rn: X86Reg, rm: X86Reg, arm_cond: u16) void {
    // ARM64 CSEL Xd, Xn, Xm, cond → Xd = cond ? Xn : Xm
    // x86: MOV dst, rm; CMOVcc dst, rn (move false value first, overwrite if cond true)
    // Map ARM64 condition → x86 CMOV opcode suffix
    const cmov_suffix: u8 = switch (arm_cond & 0xF) {
        0b0000 => 0x44, // EQ  → CMOVE
        0b0001 => 0x45, // NE  → CMOVNE
        0b0010 => 0x43, // CS/HS → CMOVAE (CF=0)
        0b0011 => 0x42, // CC/LO → CMOVB  (CF=1)
        0b0100 => 0x48, // MI  → CMOVS
        0b0101 => 0x49, // PL  → CMOVNS
        0b0110 => 0x40, // VS  → CMOVO
        0b0111 => 0x41, // VC  → CMOVNO
        0b1000 => 0x47, // HI  → CMOVA
        0b1001 => 0x46, // LS  → CMOVBE
        0b1010 => 0x4D, // GE  → CMOVGE
        0b1011 => 0x4C, // LT  → CMOVL
        0b1100 => 0x4F, // GT  → CMOVG
        0b1101 => 0x4E, // LE  → CMOVLE
        else  => 0x44,  // fallback to CMOVE
    };

    // MOV dst, rm (move false-case value)
    emitMovReg(ctx, dst, rm);
    // CMOVcc dst, rn (overwrite if condition true)
    ctx.rex(true, @intFromEnum(rn), 0, @intFromEnum(dst));
    ctx.byte(0x0F);
    ctx.byte(cmov_suffix);
    ctx.modrm(0b11, @intFromEnum(rn), @intFromEnum(dst));
}

fn emitSub(ctx: *EmitContext, op: IROp) void {
    // XZR dest = CMP: the result is discarded but the flags must be set.
    // Compute into R14 (scratch; only clobbered by BR/BLR L1 checks, which
    // come at region end, and by the .cond tail, which runs after this).
    const dst = if (isXzr(op.dest)) X86Reg.r14 else mapReg(ctx.regmap, op.dest);
    const src0_is_xzr = isXzr(op.src0);

    if (op.imm != 0) {
        if (src0_is_xzr) {
            // 0 - imm → NEG: MOV dst, imm; NEG dst → actually emit MOV and NEG
            emitMovCst(ctx, dst, op.imm);
            ctx.rex(true, 0, 0, @intFromEnum(dst));
            ctx.byte(0xF7);
            ctx.modrm(0b11, 3, @intFromEnum(dst));
        } else {
            const src0 = mapReg(ctx.regmap, op.src0);
            threeOp(ctx, dst, src0);
            if (op.imm <= 127) {
                ctx.rex(true, 0, 0, @intFromEnum(dst));
                ctx.byte(0x83);
                ctx.modrm(0b11, 5, @intFromEnum(dst));
                ctx.byte(@truncate(op.imm));
            } else {
                ctx.rex(true, 0, 0, @intFromEnum(dst));
                ctx.byte(0x81);
                ctx.modrm(0b11, 5, @intFromEnum(dst));
                ctx.bytes(std.mem.asBytes(&@as(i32, @bitCast(op.imm))));
            }
        }
    } else if (op.src1 != 0x1F) {
        const src0 = mapReg(ctx.regmap, op.src0);
        threeOp(ctx, dst, src0);
        const src1 = mapReg(ctx.regmap, op.src1);
        ctx.rex(true, @intFromEnum(src1), 0, @intFromEnum(dst));
        ctx.byte(0x29);
        ctx.modrm(0b11, @intFromEnum(src1), @intFromEnum(dst));
    } else {
        // src0 only (no immediate, no src1)
        if (src0_is_xzr) {
            // 0 - 0 = 0: only flags matter when dest is XZR (CMP XZR, XZR)
            if (isXzr(op.dest)) {
                ctx.rex(true, 0, 0, @intFromEnum(X86Reg.r14));
                ctx.byte(0x31);
                ctx.modrm(0b11, @intFromEnum(X86Reg.r14), @intFromEnum(X86Reg.r14)); // XOR r14, r14 → Z=1
                ctx.rex(true, 0, 0, @intFromEnum(X86Reg.r14));
                ctx.byte(0x83);
                ctx.modrm(0b11, 5, @intFromEnum(X86Reg.r14));
                ctx.byte(0); // SUB r14, 0 → CF=0 (no borrow)
            }
            return;
        }
        const src0 = mapReg(ctx.regmap, op.src0);
        threeOp(ctx, dst, src0);
        if (isXzr(op.dest)) {
            // CMP Xn, XZR: MOV alone sets no flags — force a flag-setting SUB
            ctx.rex(true, 0, 0, @intFromEnum(dst));
            ctx.byte(0x83);
            ctx.modrm(0b11, 5, @intFromEnum(dst));
            ctx.byte(0); // SUB r/m64, imm8 0 → CF=0 (no borrow), Z/SF from src0
        }
    }
}

fn emitAddCarry(ctx: *EmitContext, op: IROp) void {
    if (isXzr(op.dest)) return;
    // ADC: dst = src0 + src1 + CF (same as ADD but with carry-in)
    // x86 opcode: 11 /r (instead of ADD's 01 /r)
    const dst = mapReg(ctx.regmap, op.dest);
    const src0 = mapReg(ctx.regmap, op.src0);
    if (op.src1 != 0x1F) {
        const src1 = mapReg(ctx.regmap, op.src1);
        threeOp(ctx, dst, src0);
        ctx.rex(true, @intFromEnum(src1), 0, @intFromEnum(dst));
        ctx.byte(0x11);
        ctx.modrm(0b11, @intFromEnum(src1), @intFromEnum(dst));
    }
}

fn emitSubBorrow(ctx: *EmitContext, op: IROp) void {
    if (isXzr(op.dest)) return;
    // ARM64 SBC: Xd = Xn - Xm - !C
    // x86 SBB:   dst = dst - src - CF
    //
    // After CMC in nzcv_update, x86 CF = ARM64 C (both = "no borrow").
    // ARM64 SBC needs borrow = !C, x86 SBB uses CF (= C after CMC).
    // Invert CF before SBB: CMC → CF = !C → SBB subtracts !C = ✓
    const dst = mapReg(ctx.regmap, op.dest);
    const src0 = mapReg(ctx.regmap, op.src0);
    if (op.src1 != 0x1F) {
        const src1 = mapReg(ctx.regmap, op.src1);
        ctx.byte(0xF5); // CMC: CF = !C (invert back for borrow semantics)
        threeOp(ctx, dst, src0);
        ctx.rex(true, @intFromEnum(src1), 0, @intFromEnum(dst));
        ctx.byte(0x19);
        ctx.modrm(0b11, @intFromEnum(src1), @intFromEnum(dst));
    }
}

fn emitMul(ctx: *EmitContext, op: IROp) void {
    if (isXzr(op.dest)) return;
    const dst = mapReg(ctx.regmap, op.dest);
    const src0 = mapReg(ctx.regmap, op.src0);

    threeOp(ctx, dst, src0);
    const src1 = mapReg(ctx.regmap, op.src1);

    ctx.rex(true, @intFromEnum(src1), 0, @intFromEnum(dst));
    ctx.byte(0x0F);
    ctx.byte(0xAF);
    ctx.modrm(0b11, @intFromEnum(src1), @intFromEnum(dst));
}

fn emitMulHiS(ctx: *EmitContext, op: IROp) void {
    if (isXzr(op.dest)) return;
    // SMULH: signed multiply high → RDX
    // mov rax, Rn; imul Rm; mov Rd, rdx
    const dst = mapReg(ctx.regmap, op.dest);
    const rn = mapReg(ctx.regmap, op.src0);
    const rm = mapReg(ctx.regmap, op.src1);
    if (rn != .rax) emitMovReg(ctx, .rax, rn);
    ctx.rex(true, 0, 0, @intFromEnum(rm));
    ctx.byte(0xF7);
    ctx.modrm(0b11, 5, @intFromEnum(rm)); // IMUL r/m64
    emitMovReg(ctx, dst, .rdx);
}

fn emitMulHiU(ctx: *EmitContext, op: IROp) void {
    if (isXzr(op.dest)) return;
    // UMULH: unsigned multiply high → RDX
    // mov rax, Rn; mul Rm; mov Rd, rdx
    const dst = mapReg(ctx.regmap, op.dest);
    const rn = mapReg(ctx.regmap, op.src0);
    const rm = mapReg(ctx.regmap, op.src1);
    if (rn != .rax) emitMovReg(ctx, .rax, rn);
    ctx.rex(true, 0, 0, @intFromEnum(rm));
    ctx.byte(0xF7);
    ctx.modrm(0b11, 4, @intFromEnum(rm)); // MUL r/m64
    emitMovReg(ctx, dst, .rdx);
}

fn emitLogical(ctx: *EmitContext, op: IROp, opcode_byte: u8) void {
    // XZR dest = TST: result discarded, flags must be set — use R14 scratch.
    const dst = if (isXzr(op.dest)) X86Reg.r14 else mapReg(ctx.regmap, op.dest);
    const src0 = mapReg(ctx.regmap, op.src0);

    threeOp(ctx, dst, src0);
    const src1 = mapReg(ctx.regmap, op.src1);

    ctx.rex(true, @intFromEnum(src1), 0, @intFromEnum(dst));
    ctx.byte(opcode_byte);
    ctx.modrm(0b11, @intFromEnum(src1), @intFromEnum(dst));
}

fn emitNeg(ctx: *EmitContext, op: IROp) void {
    if (isXzr(op.dest)) return;
    const dst = mapReg(ctx.regmap, op.dest);
    ctx.rex(true, 0, 0, @intFromEnum(dst));
    ctx.byte(0xF7);
    ctx.modrm(0b11, 3, @intFromEnum(dst)); // NEG r/m64
}

fn emitNot(ctx: *EmitContext, op: IROp) void {
    if (isXzr(op.dest)) return;
    const dst = mapReg(ctx.regmap, op.dest);
    ctx.rex(true, 0, 0, @intFromEnum(dst));
    ctx.byte(0xF7);
    ctx.modrm(0b11, 2, @intFromEnum(dst)); // NOT r/m64
}

fn emitClz(ctx: *EmitContext, op: IROp) void {
    if (isXzr(op.dest)) return;
    const dst = mapReg(ctx.regmap, op.dest);
    const is64 = op.flags == 3;
    // LZCNT r64/r32, r/m: F3 (REX.W) 0F BD /r.
    // Zero input returns the operand width (64 or 32), matching ARM64 CLZ.
    if (isXzr(op.src0)) {
        // XZR reads as zero: XOR dst,dst then LZCNT dst,dst → result = width.
        ctx.rex(is64, 0, 0, @intFromEnum(dst));
        ctx.byte(0x31);
        ctx.modrm(0b11, @intFromEnum(dst), @intFromEnum(dst));
        ctx.byte(0xF3);
        ctx.rex(is64, @intFromEnum(dst), 0, @intFromEnum(dst));
        ctx.byte(0x0F);
        ctx.byte(0xBD);
        ctx.modrm(0b11, @intFromEnum(dst), @intFromEnum(dst));
        return;
    }
    const src0 = mapReg(ctx.regmap, op.src0);
    ctx.byte(0xF3);
    ctx.rex(is64, @intFromEnum(dst), 0, @intFromEnum(src0));
    ctx.byte(0x0F);
    ctx.byte(0xBD);
    ctx.modrm(0b11, @intFromEnum(dst), @intFromEnum(src0));
}

fn emitCrc32(ctx: *EmitContext, op: IROp) void {
    if (isXzr(op.dest)) return;
    const dst = mapReg(ctx.regmap, op.dest);
    // x86 CRC32 accumulates into its destination, so seed it with src0 first:
    //   MOV dst, src0 ; CRC32 dst, src1   (dst = CRC32(src0, src1))
    if (isXzr(op.src0)) {
        // XZR seed = 0: XOR dst,dst
        ctx.rex(true, 0, 0, @intFromEnum(dst));
        ctx.byte(0x31);
        ctx.modrm(0b11, @intFromEnum(dst), @intFromEnum(dst));
    } else {
        threeOp(ctx, dst, mapReg(ctx.regmap, op.src0));
    }
    // XZR as data source maps to RAX (garbage) — accepted limitation, matches
    // how other ops treat XZR sources; CRC32 with XZR data is pathological.
    const src1 = mapReg(ctx.regmap, op.src1);
    const size: u2 = @truncate(op.flags);
    switch (size) {
        0 => { // CRC32 r32, r/m8: F2 (REX) 0F 38 F0 /r
            ctx.byte(0xF2);
            // Always emit a REX prefix (min 0x40) so r/m8 regs in the
            // rsp/rbp/rsi/rdi range decode as SPL/BPL/SIL/DIL, not AH/BH/CH/DH.
            var rex_val: u8 = 0x40;
            if (@intFromEnum(dst) & 0x08 != 0) rex_val |= 0x04;
            if (@intFromEnum(src1) & 0x08 != 0) rex_val |= 0x01;
            ctx.byte(rex_val);
            ctx.byte(0x0F);
            ctx.byte(0x38);
            ctx.byte(0xF0);
            ctx.modrm(0b11, @intFromEnum(dst), @intFromEnum(src1));
        },
        1 => { // CRC32 r32, r/m16: 66 F2 0F 38 F1 /r
            ctx.byte(0x66);
            ctx.byte(0xF2);
            ctx.rex(false, @intFromEnum(dst), 0, @intFromEnum(src1));
            ctx.byte(0x0F);
            ctx.byte(0x38);
            ctx.byte(0xF1);
            ctx.modrm(0b11, @intFromEnum(dst), @intFromEnum(src1));
        },
        2 => { // CRC32 r32, r/m32: F2 0F 38 F1 /r
            ctx.byte(0xF2);
            ctx.rex(false, @intFromEnum(dst), 0, @intFromEnum(src1));
            ctx.byte(0x0F);
            ctx.byte(0x38);
            ctx.byte(0xF1);
            ctx.modrm(0b11, @intFromEnum(dst), @intFromEnum(src1));
        },
        3 => { // CRC32 r64, r/m64: F2 48 0F 38 F1 /r
            ctx.byte(0xF2);
            ctx.rex(true, @intFromEnum(dst), 0, @intFromEnum(src1));
            ctx.byte(0x0F);
            ctx.byte(0x38);
            ctx.byte(0xF1);
            ctx.modrm(0b11, @intFromEnum(dst), @intFromEnum(src1));
        },
    }
}

fn emitDiv(ctx: *EmitContext, op: IROp, signed: bool) void {
    if (isXzr(op.dest)) return;
    const dst = mapReg(ctx.regmap, op.dest);
    const src0 = mapReg(ctx.regmap, op.src0);
    const src1 = mapReg(ctx.regmap, op.src1);

    // x86 DIV/IDIV: RDX:RAX / r/m64 → RAX=quotient, RDX=remainder
    // ARM64 SDIV/UDIV Rd, Rn, Rm: Rd = Rn / Rm
    // We need: mov rax, src0; cqo (sign-extend to RDX); div src1; mov dst, rax
    if (dst != .rax) emitMovReg(ctx, .rax, dst);
    emitMovReg(ctx, .rax, src0);
    if (signed) {
        ctx.byte(0x48); // REX.W
        ctx.byte(0x99); // CQO: sign-extend RAX→RDX:RAX
    } else {
        const rdx_reg: X86Reg = .rdx;
        ctx.rex(true, 0, 0, @intFromEnum(rdx_reg));
        ctx.byte(0x31); // XOR RDX, RDX (zero extend for unsigned)
        ctx.modrm(0b11, @intFromEnum(rdx_reg), @intFromEnum(rdx_reg));
    }
    ctx.rex(true, 0, 0, @intFromEnum(src1));
    ctx.byte(0xF7);
    ctx.modrm(0b11, if (signed) @as(u4, 7) else @as(u4, 6), @intFromEnum(src1));
    if (dst != .rax) emitMovReg(ctx, dst, .rax);
}

fn emitMovCst(ctx: *EmitContext, dst: X86Reg, imm: u32) void {
    // MOV reg32, imm32 (without REX.W, zero-extends to 64-bit)
    // This is 5 bytes vs 10 bytes for the 64-bit version
    if (@intFromEnum(dst) >= 8) {
        // Register needs REX.B for encoding
        ctx.rex(false, 0, 0, @intFromEnum(dst));
    }
    ctx.byte(0xB8 | (@as(u8, @intFromEnum(dst)) & 0x07));
    ctx.bytes(std.mem.asBytes(&imm));
}

fn emitShiftVar(ctx: *EmitContext, op: IROp, shift_type: u4) void {
    if (isXzr(op.dest)) return;
    const dst = mapReg(ctx.regmap, op.dest);
    // Variable shift: count in CL
    ctx.rex(true, 0, 0, @intFromEnum(dst));
    ctx.byte(0xD3);
    ctx.modrm(0b11, shift_type, @intFromEnum(dst));
}

fn emitShiftImm(ctx: *EmitContext, op: IROp, shift_type: u4) void {
    if (isXzr(op.dest)) return;
    const dst = mapReg(ctx.regmap, op.dest);
    const amount = op.imm;
    if (amount == 1) {
        ctx.rex(true, 0, 0, @intFromEnum(dst));
        ctx.byte(0xD1);
        ctx.modrm(0b11, shift_type, @intFromEnum(dst));
    } else {
        ctx.rex(true, 0, 0, @intFromEnum(dst));
        ctx.byte(0xC1);
        ctx.modrm(0b11, shift_type, @intFromEnum(dst));
        ctx.byte(@truncate(amount));
    }
}

// ── Memory ─────────────────────────────────────────────────────────

fn emitLoad(ctx: *EmitContext, op: IROp, opcode: u8, rex_w: bool) void {
    if (isXzr(op.dest)) return;
    const dst = mapReg(ctx.regmap, op.dest);
    const base = mapReg(ctx.regmap, op.src0);
    const offset = op.imm;

    const is_byte = opcode == 0x8A;
    if (is_byte) {
        // For 8-bit loads, always emit REX prefix (at minimum 0x40)
        // to ensure low-byte registers (DIL, SIL, BPL, SPL) are used
        // instead of legacy high-byte registers (AH, CH, DH, BH).
        var rex_val: u8 = 0x40;
        if (@intFromEnum(dst) & 0x08 != 0) rex_val |= 0x04;
        if (@intFromEnum(base) & 0x08 != 0) rex_val |= 0x01;
        ctx.byte(rex_val);
    } else {
        ctx.rex(rex_w, @intFromEnum(dst), 0, @intFromEnum(base));
    }

    if (offset == 0) {
        ctx.byte(opcode);
        // RBP/R13 with mod=00 encodes as RIP-relative on x86-64. Use mod=01 + 0-displacement.
        if (@intFromEnum(base) & 0x07 == 0b101) {
            ctx.modrm(1, @intFromEnum(dst), @intFromEnum(base));
            ctx.byte(0);
        } else {
            ctx.modrm(0, @intFromEnum(dst), @intFromEnum(base));
        }
    } else if (offset <= 0x7F) {
        ctx.byte(opcode);
        ctx.modrm(1, @intFromEnum(dst), @intFromEnum(base));
        ctx.byte(@truncate(offset));
    } else {
        ctx.byte(opcode);
        // Manual ModRM: mod=10 (disp32), reg=dst, r/m=base
        // (Using explicit arithmetic avoids R9 self-hosted backend bug
        //  where `modrm(2, ...)` produces mod=11 instead of mod=10)
        const d2: u8 = @intFromEnum(dst) & 7;
        const b2: u8 = @intFromEnum(base) & 7;
        ctx.byte(0x80 | (d2 << 3) | b2);
        ctx.bytes(std.mem.asBytes(&@as(i32, @bitCast(offset))));
    }
}

fn emitStore(ctx: *EmitContext, op: IROp, opcode: u8, rex_w: bool) void {
    const base_ = mapReg(ctx.regmap, op.src0);
    // x8 maps to RAX via DefaultMapping, but stores use RAX as base for
    // guest address computation. Redirect to R15 (reserved for SP, but used
    // here as temporary). R15 is callee-saved and not otherwise used during
    // emitStore.
    const base: X86Reg = if (base_ == .rax) .r15 else base_;
    const src = mapReg(ctx.regmap, op.src1);
    const offset = op.imm;

    const is_byte = opcode == 0x88;
    if (is_byte) {
        // For 8-bit stores, always emit REX prefix (at minimum 0x40)
        // to ensure low-byte registers (DIL, SIL, BPL, SPL) are used
        // instead of legacy high-byte registers (AH, CH, DH, BH).
        var rex_val: u8 = 0x40;
        if (@intFromEnum(src) & 0x08 != 0) rex_val |= 0x04;
        if (@intFromEnum(base) & 0x08 != 0) rex_val |= 0x01;
        ctx.byte(rex_val);
    } else {
        ctx.rex(rex_w, @intFromEnum(src), 0, @intFromEnum(base));
    }

    if (offset == 0) {
        ctx.byte(opcode);
        if (@intFromEnum(base) & 0x07 == 0b101) {
            ctx.modrm(1, @intFromEnum(src), @intFromEnum(base));
            ctx.byte(0);
        } else {
            ctx.modrm(0, @intFromEnum(src), @intFromEnum(base));
        }
    } else if (offset <= 0x7F) {
        ctx.byte(opcode);
        ctx.modrm(1, @intFromEnum(src), @intFromEnum(base));
        ctx.byte(@truncate(offset));
    } else {
        ctx.byte(opcode);
        const s2: u8 = @intFromEnum(src) & 7;
        const b2: u8 = @intFromEnum(base) & 7;
        ctx.byte(0x80 | (s2 << 3) | b2);
        ctx.bytes(std.mem.asBytes(&@as(i32, @bitCast(offset))));
    }
}

// ── Control flow ───────────────────────────────────────────────────

fn emitBranch(ctx: *EmitContext, op: IROp) void {
    if (op.flags == 0) {
        // Direct branch: JMP rel32 placeholder (patched by chaining)
        ctx.byte(0xE9);
        ctx.bytes(&[4]u8{ 0x00, 0x00, 0x00, 0x00 });
    } else {
        // Indirect branch (BR Xn): emit L1 inline cache check + fallback.
        // R14 = &runtime.l1_cache[0]; [R14-8] = indirect_target.
        const t = mapReg(ctx.regmap, op.src0);
        const t_n = @intFromEnum(t);
        const r11_n = @intFromEnum(X86Reg.r11);
        const r14_n = @intFromEnum(X86Reg.r14);

        // 1. MOV R11, t — copy target to R11 for hash+address
        emitMovReg(ctx, .r11, t);

        // 2. SHR R11, 2 — hash = target >> 2
        ctx.rex(true, 0, 0, r11_n);
        ctx.byte(0xC1);
        ctx.modrm(0b11, 5, r11_n); // 5 = SHR opcode extension
        ctx.byte(2);  // shift amount

        // 3. AND R11, 63*16 = AND R11, 1008 (mask hash * entry_size 16)
        // REX.W + 81 /4 id
        ctx.rex(true, 0, 0, r11_n);
        ctx.byte(0x81);
        ctx.modrm(0b11, 4, r11_n);
        ctx.bytes(std.mem.asBytes(&@as(i32, @intCast(@as(u32, 63 * 16)))));

        // 4. CMP [R14 + R11], t — check if l1_cache[hash].guest_pc == target
        // REX.W + REX.X(R11) + REX.B(R14): 0x48 | 0x02(R11>7?1:0) | 0x01(R14>7?1:0)
        // For R11=11(0b1011) and R14=14(0b1110): X=1, B=1
        const rex_cmp: u8 = 0x48 | @as(u8, if (r11_n > 7) 0x02 else 0) | @as(u8, if (r14_n > 7) 0x01 else 0);
        ctx.byte(rex_cmp);
        ctx.byte(0x39); // CMP r/m64, r64
        // ModRM: mod=00, reg=t&7, rm=100(SIB)
        ctx.modrm(0b00, t_n & 7, 0b100);
        // SIB: scale=00(1), index=R11&7, base=R14&7
        ctx.byte(@as(u8, (0 << 6) | ((r11_n & 7) << 3) | (r14_n & 7)));

        // 5. JNE skip (rel8 = +5 to skip over JMP instruction)
        ctx.byte(0x75);
        ctx.byte(5); // jump forward 5 bytes

        // 6. JMP [R14 + R11 + 8] — hit: jump to cached host address
        const rex_jmp: u8 = 0x40 | @as(u8, if (r11_n > 7) 0x02 else 0) | @as(u8, if (r14_n > 7) 0x01 else 0);
        if (rex_jmp != 0x40) ctx.byte(rex_jmp);
        ctx.byte(0xFF);
        ctx.modrm(0b01, 4, 0b100); // mod=01(disp8), reg=4(JMP), rm=100(SIB)
        ctx.byte(@as(u8, (0 << 6) | ((r11_n & 7) << 3) | (r14_n & 7)));
        ctx.byte(8); // disp8 = +8

        // 7. .miss: MOV [R14-8], t — store to indirect_target for runtime
        // REX.W + REX.B(R14): 0x48 | 0x01
        const rex_mov: u8 = 0x48 | @as(u8, if (r14_n > 7) 0x01 else 0);
        ctx.byte(rex_mov);
        ctx.byte(0x89);
        ctx.modrm(0b01, t_n & 7, r14_n & 7);
        ctx.byte(@as(u8, @bitCast(@as(i8, -8)))); // disp8 = -8

        // 8. RET
        ctx.byte(0xC3);
    }
}

fn emitCall(ctx: *EmitContext, op: IROp) void {
    _ = op;
    // CALL rel32=0 (placeholder — patched by chaining code if target cached)
    ctx.byte(0xE8);
    ctx.bytes(&[4]u8{ 0x00, 0x00, 0x00, 0x00 });
    // RET after CALL — when target returns, control goes back to execute()
    // Also serves as safe fallthrough if CALL never gets patched (rel32=0 → next insn)
    ctx.byte(0xC3);
}

fn emitCallReg(ctx: *EmitContext, op: IROp) void {
    // BLR Xn: indirect call via register with same L1 cache as BR.
    // The return address is handled by the block's store to state.x[30].
    const t = mapReg(ctx.regmap, op.src0);
    const t_n = @intFromEnum(t);
    const r11_n = @intFromEnum(X86Reg.r11);
    const r14_n = @intFromEnum(X86Reg.r14);

    // L1 inline cache: same as emitBranch indirect path
    emitMovReg(ctx, .r11, t);
    ctx.rex(true, 0, 0, r11_n);
    ctx.byte(0xC1);
    ctx.modrm(0b11, 5, r11_n);
    ctx.byte(2);
    ctx.rex(true, 0, 0, r11_n);
    ctx.byte(0x81);
    ctx.modrm(0b11, 4, r11_n);
    ctx.bytes(std.mem.asBytes(&@as(i32, @intCast(@as(u32, 63 * 16)))));

    const rex_cmp: u8 = 0x48 | @as(u8, if (r11_n > 7) 0x02 else 0) | @as(u8, if (r14_n > 7) 0x01 else 0);
    ctx.byte(rex_cmp);
    ctx.byte(0x39);
    ctx.modrm(0b00, t_n & 7, 0b100);
    ctx.byte(@as(u8, (0 << 6) | ((r11_n & 7) << 3) | (r14_n & 7)));

    ctx.byte(0x75);
    ctx.byte(5);

    const rex_jmp: u8 = 0x40 | @as(u8, if (r11_n > 7) 0x02 else 0) | @as(u8, if (r14_n > 7) 0x01 else 0);
    if (rex_jmp != 0x40) ctx.byte(rex_jmp);
    ctx.byte(0xFF);
    ctx.modrm(0b01, 4, 0b100);
    ctx.byte(@as(u8, (0 << 6) | ((r11_n & 7) << 3) | (r14_n & 7)));
    ctx.byte(8);

    const rex_mov: u8 = 0x48 | @as(u8, if (r14_n > 7) 0x01 else 0);
    ctx.byte(rex_mov);
    ctx.byte(0x89);
    ctx.modrm(0b01, t_n & 7, r14_n & 7);
    ctx.byte(@as(u8, @bitCast(@as(i8, -8))));
    ctx.byte(0xC3);
}

fn emitRet(ctx: *EmitContext) void {
    ctx.byte(0xC3);
}

fn emitBrCond(ctx: *EmitContext, op: IROp) void {
    // ARM64 condition code in op.flags → x86-64 JCC opcode
    const jcc_opcode: u8 = switch (op.flags & 0xF) {
        0b0000 => 0x84, // EQ  → JE  (ZF=1)
        0b0001 => 0x85, // NE  → JNE (ZF=0)
        0b0010 => 0x83, // CS/HS → JAE (CF=0) — C=1 in ARM64 = CF=0 in x86
        0b0011 => 0x82, // CC/LO → JB  (CF=1) — C=0 in ARM64 = CF=1 in x86
        0b0100 => 0x88, // MI  → JS  (SF=1)
        0b0101 => 0x89, // PL  → JNS (SF=0)
        0b0110 => 0x80, // VS  → JO  (OF=1)
        0b0111 => 0x81, // VC  → JNO (OF=0)
        0b1000 => 0x87, // HI  → JA  (CF=0 & ZF=0)
        0b1001 => 0x86, // LS  → JBE (CF=1 | ZF=1)
        0b1010 => 0x8D, // GE  → JGE (SF=OF)
        0b1011 => 0x8C, // LT  → JL  (SF≠OF)
        0b1100 => 0x8F, // GT  → JG  (ZF=0 & SF=OF)
        0b1101 => 0x8E, // LE  → JLE (ZF=1 | SF≠OF)
        0b1110 => 0x00, // AL  → JMP (unconditional, handled separately)
        else => 0x84,   // default to JE
    };

    if ((op.flags & 0xF) == 0b1110) {
        // AL = unconditional: JMP rel32
        ctx.byte(0xE9);
    } else {
        ctx.byte(0x0F);
        ctx.byte(jcc_opcode);
    }
    ctx.bytes(&[4]u8{ 0x00, 0x00, 0x00, 0x00 }); // placeholder offset
}

fn emitNZCVUpdate(ctx: *EmitContext, op: IROp) void {
    if (op.imm != 0) {
        ctx.byte(0xF5); // CMC: complement carry flag
    }
}

fn emitNZCVRead(ctx: *EmitContext, op: IROp) void {
    _ = op;
    // Materialize x86-64 RFLAGS into RAX so the next operation can read NZCV.
    // This is needed when the IR emits nzcv_read without a preceding
    // flag-setting instruction.
    ctx.byte(0x9C); // PUSHFQ — push RFLAGS onto stack
    ctx.byte(0x58); // POP RAX  — pop into RAX
}

// ── CCMP (conditional compare) ──────────────────────────────────────

fn emitPushf(ctx: *EmitContext) void {
    ctx.byte(0x9C); // PUSHFQ
}

fn emitPopf(ctx: *EmitContext) void {
    ctx.byte(0x9D); // POPFQ
}

fn emitAndRaxImm(ctx: *EmitContext, imm: u32) void {
    ctx.rex(true, 0, 0, 0);
    ctx.byte(0x25);
    ctx.bytes(std.mem.asBytes(&imm));
}

fn emitOrRaxImm(ctx: *EmitContext, imm: u32) void {
    ctx.rex(true, 0, 0, 0);
    ctx.byte(0x0D);
    ctx.bytes(std.mem.asBytes(&imm));
}

/// Map 4-bit NZCV to x86-64 RFLAGS bits.
/// N→SF(7), Z→ZF(6), C→CF(0), V→OF(11)
fn nzcvToRflags(nzcv: u32) u32 {
    var flags: u32 = 0;
    // C flag → CF bit 0
    if (nzcv & 1 != 0) flags |= 1;
    // V flag → OF bit 11
    if (nzcv & 2 != 0) flags |= 1 << 11;
    // Z flag → ZF bit 6
    if (nzcv & 4 != 0) flags |= 1 << 6;
    // N flag → SF bit 7
    if (nzcv & 8 != 0) flags |= 1 << 7;
    return flags;
}

/// CCMP: compare if condition true, else set NZCV from immediate.
/// Emits: Jcc_else → CMP+CMC+JMP | PUSHFQ+AND+OR+POPFQ
fn emitCCmp(ctx: *EmitContext, op: IROp) void {
    // op.flags = ARM64 condition to check
    // op.src0 = rn, op.src1 = rm
    // op.imm = 4-bit NZCV value
    const cond = op.flags & 0xF;
    const nzcv_val = op.imm & 0xF;

    // x86 condition inverse of ARM64 cond
    const inv_cc: u8 = switch (cond) {
        0b0000 => 0x85, // EQ → JNE (0F 85)
        0b0001 => 0x84, // NE → JE (0F 84)
        0b0010 => 0x82, // CS → JB (0F 82) — C=1 → fall through if C=0
        0b0011 => 0x83, // CC → JAE (0F 83)
        0b0100 => 0x89, // MI → JNS (0F 89)
        0b0101 => 0x88, // PL → JS (0F 88)
        0b0110 => 0x81, // VS → JNO (0F 81)
        0b0111 => 0x80, // VC → JO (0F 80)
        0b1000 => 0x86, // HI → JBE (0F 86)
        0b1001 => 0x87, // LS → JA (0F 87)
        0b1010 => 0x8C, // GE → JL (0F 8C)
        0b1011 => 0x8D, // LT → JGE (0F 8D)
        0b1100 => 0x8E, // GT → JLE (0F 8E)
        0b1101 => 0x8F, // LE → JG (0F 8F)
        0b1110 => return, // AL → always, just CMP (skip CCMP pattern)
        else => 0x85, // default to JNE
    };

    // We'll emit Jcc_else with placeholder, then CMP+CMC, JMP+placeholder,
    // then PUSHFQ...POPFQ.
    // Patch the jump offsets after we know how long each path is.

    ctx.byte(0x0F);
    ctx.byte(inv_cc);
    const else_rel32_off = ctx.offset; // placeholder
    ctx.bytes(&[4]u8{ 0x00, 0x00, 0x00, 0x00 });
    const else_branch_end = ctx.offset;

    // ── Condition met path: CMP + CMC ────────────────────────────
    const rn = mapReg(ctx.regmap, op.src0);
    const rm = mapReg(ctx.regmap, op.src1);
    // CMP rn, rm (SUB r/m64, r64)
    ctx.rex(true, @intFromEnum(rm), 0, @intFromEnum(rn));
    ctx.byte(0x39);
    ctx.modrm(0b11, @intFromEnum(rm), @intFromEnum(rn));
    ctx.byte(0xF5); // CMC
    ctx.byte(0xE9); // JMP rel32
    const end_rel32_off = ctx.offset;
    ctx.bytes(&[4]u8{ 0x00, 0x00, 0x00, 0x00 });
    const end_of_jmp = ctx.offset;

    // ── Condition NOT met path: set NZCV from immediate ──────────
    const else_path_start = ctx.offset;
    emitPushf(ctx);
    ctx.byte(0x58); // POP RAX
    const mask = ~(@as(u32, 1) | (1 << 6) | (1 << 7) | (1 << 11)); // clear CF,ZF,SF,OF
    emitAndRaxImm(ctx, mask);
    const flags_val = nzcvToRflags(nzcv_val);
    if (flags_val != 0) {
        emitOrRaxImm(ctx, flags_val);
    }
    emitPushf(ctx);
    emitPopf(ctx);
    const else_path_end = ctx.offset;

    // ── Patch branch offsets ─────────────────────────────────────
    // Jcc_else → else_path_start
    const else_rel32: i32 = @intCast(else_path_start - else_branch_end);
    std.mem.writeInt(i32, ctx.buf[else_rel32_off..][0..4], else_rel32, .little);
    // JMP .end → else_path_end
    const end_rel32: i32 = @intCast(else_path_end - end_of_jmp);
    std.mem.writeInt(i32, ctx.buf[end_rel32_off..][0..4], end_rel32, .little);
}

// ── Atomic / exclusive ────────────────────────────────────────────

fn emitLoadExcl(ctx: *EmitContext, op: IROp) void {
    // On x86-64, exclusive load is the same as a regular load.
    emitLoad(ctx, op, 0x8B, true);
}

fn emitStoreExcl(ctx: *EmitContext, op: IROp) void {
    // On x86-64, exclusive store is a regular store + status=0 (always succeed).
    // First, emit the store (same as store_u64 but with store_excl semantics).
    emitStore(ctx, op, 0x89, true);

    // Set status register (op.dest = Rs) to 0 (success).
    if (isXzr(op.dest)) return;
    const status = mapReg(ctx.regmap, op.dest);
    ctx.rex(true, @intFromEnum(status), 0, @intFromEnum(status));
    ctx.byte(0x31); // XOR r/m64, r64
    ctx.modrm(0b11, @intFromEnum(status) & 7, @intFromEnum(status) & 7);
}

fn emitAtomicAdd(ctx: *EmitContext, op: IROp) void {
    if (isXzr(op.dest)) return;
    const addr_reg = mapReg(ctx.regmap, op.src0);
    const val_reg = mapReg(ctx.regmap, op.src1);
    const dst_reg = mapReg(ctx.regmap, op.dest);

    ctx.byte(0xF0); // LOCK prefix
    ctx.rex(true, @intFromEnum(val_reg), 0, @intFromEnum(addr_reg));
    ctx.byte(0x0F);
    ctx.byte(0xC1); // XADD r/m64, r64
    ctx.modrm(0b00, @intFromEnum(val_reg) & 7, @intFromEnum(addr_reg) & 7);

    // After XADD, val_reg holds the OLD value from memory.
    if (dst_reg != val_reg) {
        emitMovReg(ctx, dst_reg, val_reg);
    }
}

fn emitAtomicCas(ctx: *EmitContext, op: IROp) void {
    // CAS Xs, Xt, [Xn]: if ([Rn] == Rs) then [Rn] = Rt; Rt = old [Rn]
    // x86 CMPXCHG: compares [addr] with RAX; if equal, [addr] = reg else RAX = [addr]
    if (isXzr(op.dest)) return;

    const addr_reg = mapReg(ctx.regmap, op.src0); // Rn
    const expected_reg = mapReg(ctx.regmap, op.src1); // Rs
    const new_val_reg = mapReg(ctx.regmap, op.dest); // Rt

    // Expected value must be in RAX for CMPXCHG
    if (expected_reg != .rax) {
        emitMovReg(ctx, .rax, expected_reg);
    }

    ctx.byte(0xF0); // LOCK prefix
    ctx.rex(true, @intFromEnum(new_val_reg), 0, @intFromEnum(addr_reg));
    ctx.byte(0x0F);
    ctx.byte(0xB1); // CMPXCHG r/m64, r64
    ctx.modrm(0b00, @intFromEnum(new_val_reg) & 7, @intFromEnum(addr_reg) & 7);

    // After CMPXCHG, RAX holds the old value from memory.
    if (new_val_reg != .rax) {
        emitMovReg(ctx, new_val_reg, .rax);
    }
}

// ── FPCR/FPSR state access ────────────────────────────────────────
// Arm64State is at offset 0 of JitRuntime. R14 points to JitRuntime.
// Offsets computed from state.zig layout:
//   x[31]:   0..247  (31 * 8)
//   sp:      248
//   pc:      256
//   nzcv:    264 (u32)
//   fpcr:    268 (u32)
//   fpsr:    272 (u32)
const STATE_FPCR_OFFSET: u16 = 268;
const STATE_FPSR_OFFSET: u16 = 272;

fn emitLoadFromState(ctx: *EmitContext, dst: X86Reg, offset: u16) void {
    // MOV dst (32-bit, zero-extending), [R14 + offset]
    const dst_n = @intFromEnum(dst);
    const r14_n = @intFromEnum(X86Reg.r14);
    var rex: u8 = 0x40;
    if (dst_n > 7) rex |= 0x04; // REX.R
    if (r14_n > 7) rex |= 0x01; // REX.B (always for R14=14)
    if (rex != 0x40) ctx.byte(rex);
    ctx.byte(0x8B);
    if (offset <= 127) {
        ctx.modrm(0b01, dst_n & 7, r14_n & 7);
        ctx.byte(@truncate(offset));
    } else {
        ctx.modrm(0b10, dst_n & 7, r14_n & 7);
        ctx.bytes(std.mem.asBytes(&@as(i32, @bitCast(@as(u32, offset)))));
    }
}

fn emitStoreToState(ctx: *EmitContext, src: X86Reg, offset: u16) void {
    // MOV [R14 + offset], src (32-bit store)
    const src_n = @intFromEnum(src);
    const r14_n = @intFromEnum(X86Reg.r14);
    var rex: u8 = 0x40;
    if (src_n > 7) rex |= 0x04; // REX.R
    if (r14_n > 7) rex |= 0x01; // REX.B (always for R14=14)
    if (rex != 0x40) ctx.byte(rex);
    ctx.byte(0x89);
    if (offset <= 127) {
        ctx.modrm(0b01, src_n & 7, r14_n & 7);
        ctx.byte(@truncate(offset));
    } else {
        ctx.modrm(0b10, src_n & 7, r14_n & 7);
        ctx.bytes(std.mem.asBytes(&@as(i32, @bitCast(@as(u32, offset)))));
    }
}

fn emitFpcrRead(ctx: *EmitContext, op: IROp) void {
    if (isXzr(op.dest)) return;
    const dst = mapReg(ctx.regmap, op.dest);
    emitLoadFromState(ctx, dst, STATE_FPCR_OFFSET);
}

fn emitFpcrWrite(ctx: *EmitContext, op: IROp) void {
    if (isXzr(op.src0)) return;
    const src = mapReg(ctx.regmap, op.src0);
    emitStoreToState(ctx, src, STATE_FPCR_OFFSET);
}

fn emitFpsrRead(ctx: *EmitContext, op: IROp) void {
    if (isXzr(op.dest)) return;
    const dst = mapReg(ctx.regmap, op.dest);
    emitLoadFromState(ctx, dst, STATE_FPSR_OFFSET);
}

fn emitFpsrWrite(ctx: *EmitContext, op: IROp) void {
    if (isXzr(op.src0)) return;
    const src = mapReg(ctx.regmap, op.src0);
    emitStoreToState(ctx, src, STATE_FPSR_OFFSET);
}

// ── Main dispatch ──────────────────────────────────────────────────

pub fn emitOp(ctx: *EmitContext, op: IROp) usize {
    const start = ctx.offset;
    switch (op.tag) {
        .add_i64 => emitAdd(ctx, op),
        .adc_i64 => emitAddCarry(ctx, op),
        .sub_i64 => emitSub(ctx, op),
        .sbc_i64 => emitSubBorrow(ctx, op),
        .mul_i64 => emitMul(ctx, op),
        .div_u64, .div_s64 => emitDiv(ctx, op, op.tag == .div_s64),
        .mul_hi_s64 => emitMulHiS(ctx, op),
        .mul_hi_u64 => emitMulHiU(ctx, op),
        .and_ => emitLogical(ctx, op, 0x21),
        .or_ => emitLogical(ctx, op, 0x09),
        .xor_ => emitLogical(ctx, op, 0x31),
        .not_ => emitNot(ctx, op),
        .neg_i64 => emitNeg(ctx, op),
        .mov_i64 => {
            if (isXzr(op.dest)) return ctx.offset - start;
            const d = mapReg(ctx.regmap, op.dest);
            const s = mapReg(ctx.regmap, op.src0);
            emitMovReg(ctx, d, s);
        },
        .lshl_i64 => emitShiftVar(ctx, op, 4),
        .lshr_i64 => emitShiftVar(ctx, op, 5),
        .ashr_i64 => emitShiftVar(ctx, op, 7),
        .lshl_i64_imm => emitShiftImm(ctx, op, 4),
        .lshr_i64_imm => emitShiftImm(ctx, op, 5),
        .ashr_i64_imm => emitShiftImm(ctx, op, 7),
        .clz => emitClz(ctx, op),
        .crc32 => emitCrc32(ctx, op),

        .load_u64 => emitLoad(ctx, op, 0x8B, true),
        .load_u32 => emitLoad(ctx, op, 0x8B, false),
        .load_u16 => {
            ctx.byte(0x66);
            emitLoad(ctx, op, 0x8B, false);
        },
        .load_u8 => emitLoad(ctx, op, 0x8A, false),

        .store_u64 => emitStore(ctx, op, 0x89, true),
        .store_u32 => emitStore(ctx, op, 0x89, false),
        .store_u16 => {
            ctx.byte(0x66);
            emitStore(ctx, op, 0x89, false);
        },
        .store_u8 => emitStore(ctx, op, 0x88, false),

        .br => emitBranch(ctx, op),
        .call => emitCall(ctx, op),
        .call_reg => emitCallReg(ctx, op),
        .ret_ => emitRet(ctx),
        .br_cond => emitBrCond(ctx, op),
        .ccmp => emitCCmp(ctx, op),

        .nzcv_update => emitNZCVUpdate(ctx, op),
        .nzcv_read => emitNZCVRead(ctx, op),

        .sp_get => {
            if (isXzr(op.dest)) return ctx.offset - start;
            // SP lives in R15 before block call; preserved by sp_put.
            const dst = mapReg(ctx.regmap, op.dest);
            emitMovReg(ctx, dst, .r15);
        },
        .sp_put => {
            const src = mapReg(ctx.regmap, op.src0);
            emitMovReg(ctx, .r15, src);
        },

        // ── Atomic / exclusive ─────────────────────────────────
        .load_excl => emitLoadExcl(ctx, op),
        .store_excl => emitStoreExcl(ctx, op),
        .atomic_add => emitAtomicAdd(ctx, op),
        .atomic_cas => emitAtomicCas(ctx, op),

        // ── FPCR / FPSR ────────────────────────────────────────
        .fpcr_read => emitFpcrRead(ctx, op),
        .fpcr_write => emitFpcrWrite(ctx, op),
        .fpsr_read => emitFpsrRead(ctx, op),
        .fpsr_write => emitFpsrWrite(ctx, op),

        else => ctx.byte(0xCC),
    }
    return ctx.offset - start;
}

pub fn emitBlock(buf: []u8, regmap: *const RegisterMap, ops: []const IROp) []u8 {
    var ctx = EmitContext.init(buf, regmap);
    for (ops) |op| _ = emitOp(&ctx, op);
    if (ctx.offset == 0) emitRet(&ctx);
    const optimized_len = Peephole.optimize(buf, ctx.offset);
    return buf[0..optimized_len];
}

// ── Trampoline ─────────────────────────────────────────────────────
// Generates a small stub that:
//   1. Loads mapped host regs from Arm64State (pointer in RDI)
//   2. Calls the translated block (address in RSI)
//   3. Stores host regs back to Arm64State
//   4. Returns

/// State offsets for Arm64State fields (x[0..7])
const state_x_offset = struct {
    fn get(i: u64) u64 { return i * 8; }
};

/// Buffer size needed for the trampoline.
pub const TRAMPOLINE_SIZE: usize = 256;

/// Emit the trampoline into `buf`. Must be at least TRAMPOLINE_SIZE bytes.
/// Returns the slice of emitted code.
pub fn emitTrampoline(buf: []u8) []u8 {
    var ctx = EmitContext.init(buf, &DefaultMapping);

    // Save callee-saved registers we use (with REX prefix for r8-r15)
    for ([_]X86Reg{ .rbx, .rbp, .r12, .r13, .r14, .r15 }) |reg| {
        const r: u8 = @intFromEnum(reg);
        if (r >= 8) ctx.byte(0x41); // REX.B
        ctx.byte(0x50 + (r & 7)); // PUSH reg
    }

    // mov r15, rdi  — save state pointer (r15 is callee-saved, not mapped)
    ctx.rex(true, @intFromEnum(X86Reg.rdi), 0, @intFromEnum(X86Reg.r14));
    ctx.byte(0x89);
    ctx.modrm(0b11, @intFromEnum(X86Reg.rdi), @intFromEnum(X86Reg.r14));

    // mov rax, rsi  — save block address
    ctx.rex(true, @intFromEnum(X86Reg.rsi), 0, @intFromEnum(X86Reg.rax));
    ctx.byte(0x89);
    ctx.modrm(0b11, @intFromEnum(X86Reg.rsi), @intFromEnum(X86Reg.rax));

    // Load mapped GPRs from state (x0-x7 → RDI, RSI, RDX, RCX, R8-R11)
    const mapped_regs = [_]X86Reg{ .rdi, .rsi, .rdx, .rcx, .r8, .r9, .r10, .r11 };
    inline for (mapped_regs, 0..) |reg, i| {
        const off: u8 = @intCast(i * 8);
        // mov reg, [r15 + off]
        ctx.rex(true, @intFromEnum(reg), 0, @intFromEnum(X86Reg.r14));
        if (off < 128) {
            ctx.byte(0x8B);
            ctx.modrm(0b01, @intFromEnum(reg), @intFromEnum(X86Reg.r14));
            ctx.byte(off);
        } else {
            ctx.byte(0x8B);
            ctx.modrm(0b10, @intFromEnum(reg), @intFromEnum(X86Reg.r14));
            ctx.disp32(@intCast(off));
        }
    }

    // call rax — translated block address is in rax
    // FF D0 = CALL RAX (indirect)
    ctx.byte(0xFF);
    ctx.modrm(0b11, 2, @intFromEnum(X86Reg.rax));

    // Store regs back to state
    inline for (mapped_regs, 0..) |reg, i| {
        const off: u8 = @intCast(i * 8);
        ctx.rex(true, @intFromEnum(reg), 0, @intFromEnum(X86Reg.r14));
        if (off < 128) {
            ctx.byte(0x89);
            ctx.modrm(0b01, @intFromEnum(reg), @intFromEnum(X86Reg.r14));
            ctx.byte(off);
        } else {
            ctx.byte(0x89);
            ctx.modrm(0b10, @intFromEnum(reg), @intFromEnum(X86Reg.r14));
            ctx.disp32(@intCast(off));
        }
    }

    // Restore callee-saved regs (reverse order, with REX prefix for r8-r15)
    for ([_]X86Reg{ .r15, .r14, .r13, .r12, .rbp, .rbx }) |reg| {
        const r: u8 = @intFromEnum(reg);
        if (r >= 8) ctx.byte(0x41); // REX.B
        ctx.byte(0x58 + (r & 7)); // POP reg
    }

    ctx.byte(0xC3); // RET
    return buf[0..ctx.offset];
}

// ── Tests ─────────────────────────────────────────────────────────

test "emit ADD immediate" {
    var code: [128]u8 = undefined;
    const op = IROp{ .tag = .add_i64, .dest = 0, .src0 = 0, .src1 = 0x1F, .flags = 0, .imm = 42 };
    const emitted = emitBlock(&code, &DefaultMapping, &.{op});
    try std.testing.expectEqual(@as(u8, 0x48), emitted[0]);
    try std.testing.expectEqual(@as(u8, 0x83), emitted[1]);
    try std.testing.expectEqual(@as(u8, 0xC7), emitted[2]);
    try std.testing.expectEqual(@as(u8, 42), emitted[3]);
}

test "emit 3-operand ADD (dst ≠ src0)" {
    var code: [128]u8 = undefined;
    // ADD X2, X0, X1 → RDX = RDI + RSI (needs MOV first since dst=RDX, src0=RDI)
    const op = IROp{ .tag = .add_i64, .dest = 2, .src0 = 0, .src1 = 1, .flags = 0, .imm = 0 };
    const emitted = emitBlock(&code, &DefaultMapping, &.{op});
    // Should emit: MOV RDX, RDI (48 89 FA) + ADD RDX, RSI (48 01 F2)
    try std.testing.expect(emitted.len >= 6);
    try std.testing.expectEqual(@as(u8, 0x48), emitted[0]);
    try std.testing.expectEqual(@as(u8, 0x89), emitted[1]); // MOV
    try std.testing.expectEqual(@as(u8, 0x48), emitted[3]); // REX.W (second insn)
    try std.testing.expectEqual(@as(u8, 0x01), emitted[4]); // ADD
}

test "emit LOAD" {
    var code: [128]u8 = undefined;
    const op = IROp{ .tag = .load_u64, .dest = 0, .src0 = 1, .src1 = 0, .flags = 0, .imm = 16 };
    const emitted = emitBlock(&code, &DefaultMapping, &.{op});
    try std.testing.expectEqual(@as(u8, 0x48), emitted[0]);
    try std.testing.expectEqual(@as(u8, 0x8B), emitted[1]);
    try std.testing.expectEqual(@as(u8, 0x7E), emitted[2]);
    try std.testing.expectEqual(@as(u8, 16), emitted[3]);
}

test "emit NOT" {
    var code: [128]u8 = undefined;
    const op = IROp{ .tag = .not_, .dest = 0, .src0 = 0, .src1 = 0, .flags = 0, .imm = 0 };
    const emitted = emitBlock(&code, &DefaultMapping, &.{op});
    try std.testing.expectEqual(@as(u8, 0x48), emitted[0]);
    try std.testing.expectEqual(@as(u8, 0xF7), emitted[1]);
}

test "emit CLZ (LZCNT r64)" {
    var code: [128]u8 = undefined;
    // CLZ X0, X1 → LZCNT RDI, RSI = F3 48 0F BD FE
    // (LZCNT encodes the dest in the ModRM reg field; objdump-verified)
    const op = IROp{ .tag = .clz, .dest = 0, .src0 = 1, .src1 = 0x1F, .flags = 3, .imm = 0 };
    const emitted = emitBlock(&code, &DefaultMapping, &.{op});
    try std.testing.expectEqual(@as(u8, 0xF3), emitted[0]);
    try std.testing.expectEqual(@as(u8, 0x48), emitted[1]);
    try std.testing.expectEqual(@as(u8, 0x0F), emitted[2]);
    try std.testing.expectEqual(@as(u8, 0xBD), emitted[3]);
    try std.testing.expectEqual(@as(u8, 0xFE), emitted[4]);
}

test "emit CLZ (LZCNT r32, no REX.W)" {
    var code: [128]u8 = undefined;
    // CLZ W0, W1 → LZCNT EDI, ESI = F3 0F BD FE
    const op = IROp{ .tag = .clz, .dest = 0, .src0 = 1, .src1 = 0x1F, .flags = 2, .imm = 0 };
    const emitted = emitBlock(&code, &DefaultMapping, &.{op});
    try std.testing.expectEqual(@as(u8, 0xF3), emitted[0]);
    try std.testing.expectEqual(@as(u8, 0x0F), emitted[1]);
    try std.testing.expectEqual(@as(u8, 0xBD), emitted[2]);
    try std.testing.expectEqual(@as(u8, 0xFE), emitted[3]);
}

test "emit CRC32 (32-bit data form)" {
    var code: [128]u8 = undefined;
    // dest ← CRC32(src0, src1) with 32-bit data → MOV RDI, RSI (48 89 F7) + CRC32 EDI, EDX (F2 0F 38 F1 FA)
    const op = IROp{ .tag = .crc32, .dest = 0, .src0 = 1, .src1 = 2, .flags = 2, .imm = 0 };
    const emitted = emitBlock(&code, &DefaultMapping, &.{op});
    try std.testing.expectEqual(@as(u8, 0x48), emitted[0]); // REX.W
    try std.testing.expectEqual(@as(u8, 0x89), emitted[1]); // MOV
    try std.testing.expectEqual(@as(u8, 0xF2), emitted[3]); // F2 prefix
    try std.testing.expectEqual(@as(u8, 0x0F), emitted[4]);
    try std.testing.expectEqual(@as(u8, 0x38), emitted[5]);
    try std.testing.expectEqual(@as(u8, 0xF1), emitted[6]);
    try std.testing.expectEqual(@as(u8, 0xFA), emitted[7]); // modrm: reg=RDI, rm=RDX
}
