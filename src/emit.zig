//! x86-64 machine code emitter.
//!
//! Translates IR ops into x86-64 machine code. Handles the 3-operand
//! to 2-operand mapping by emitting MOV+ALU when dst ≠ src0.

const std = @import("std");
const Ir = @import("ir.zig");
const IROp = Ir.IROp;
const Tag = Ir.Tag;

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
    return regmap[arm_reg] orelse .r11;
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
    _: u8,
    _: u8,
) void {
    if (dst == src0) return;
    emitMovReg(ctx, dst, src0);
}

// ── ALU emission ───────────────────────────────────────────────────

fn emitAdd(ctx: *EmitContext, op: IROp) void {
    const dst = mapReg(ctx.regmap, op.dest);
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
        } else {
            const src0 = mapReg(ctx.regmap, op.src0);
            threeOp(ctx, dst, src0, 0x01, 0x03);
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
        threeOp(ctx, dst, src0, 0x01, 0x03);
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
    const dst = mapReg(ctx.regmap, op.dest);
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
            threeOp(ctx, dst, src0, 0x29, 0x2B);
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
        threeOp(ctx, dst, src0, 0x29, 0x2B);
        const src1 = mapReg(ctx.regmap, op.src1);
        ctx.rex(true, @intFromEnum(src1), 0, @intFromEnum(dst));
        ctx.byte(0x29);
        ctx.modrm(0b11, @intFromEnum(src1), @intFromEnum(dst));
    } else if (!isXzr(op.dest)) {
        // src0 only (no immediate, no src1): just copy/move
        if (!src0_is_xzr) {
            const src0 = mapReg(ctx.regmap, op.src0);
            emitMovReg(ctx, dst, src0);
        } else {
            // 0 - 0 = 0
        }
    }
}

fn emitAddCarry(ctx: *EmitContext, op: IROp) void {
    // ADC: dst = src0 + src1 + CF (same as ADD but with carry-in)
    // x86 opcode: 11 /r (instead of ADD's 01 /r)
    const dst = mapReg(ctx.regmap, op.dest);
    const src0 = mapReg(ctx.regmap, op.src0);
    if (op.src1 != 0x1F) {
        const src1 = mapReg(ctx.regmap, op.src1);
        threeOp(ctx, dst, src0, 0x11, 0x13);
        ctx.rex(true, @intFromEnum(src1), 0, @intFromEnum(dst));
        ctx.byte(0x11);
        ctx.modrm(0b11, @intFromEnum(src1), @intFromEnum(dst));
    }
}

fn emitSubBorrow(ctx: *EmitContext, op: IROp) void {
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
        threeOp(ctx, dst, src0, 0x19, 0x1B);
        ctx.rex(true, @intFromEnum(src1), 0, @intFromEnum(dst));
        ctx.byte(0x19);
        ctx.modrm(0b11, @intFromEnum(src1), @intFromEnum(dst));
    }
}

fn emitMul(ctx: *EmitContext, op: IROp) void {
    const dst = mapReg(ctx.regmap, op.dest);
    const src0 = mapReg(ctx.regmap, op.src0);

    threeOp(ctx, dst, src0, 0, 0);
    const src1 = mapReg(ctx.regmap, op.src1);

    ctx.rex(true, @intFromEnum(src1), 0, @intFromEnum(dst));
    ctx.byte(0x0F);
    ctx.byte(0xAF);
    ctx.modrm(0b11, @intFromEnum(src1), @intFromEnum(dst));
}

fn emitMulHiS(ctx: *EmitContext, op: IROp) void {
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

fn emitLogical(ctx: *EmitContext, op: IROp, opcode_byte: u8, opcode_rev: u8) void {
    const dst = mapReg(ctx.regmap, op.dest);
    const src0 = mapReg(ctx.regmap, op.src0);

    threeOp(ctx, dst, src0, opcode_byte, opcode_rev);
    const src1 = mapReg(ctx.regmap, op.src1);

    ctx.rex(true, @intFromEnum(src1), 0, @intFromEnum(dst));
    ctx.byte(opcode_byte);
    ctx.modrm(0b11, @intFromEnum(src1), @intFromEnum(dst));
}

fn emitNeg(ctx: *EmitContext, op: IROp) void {
    const dst = mapReg(ctx.regmap, op.dest);
    ctx.rex(true, 0, 0, @intFromEnum(dst));
    ctx.byte(0xF7);
    ctx.modrm(0b11, 3, @intFromEnum(dst)); // NEG r/m64
}

fn emitNot(ctx: *EmitContext, op: IROp) void {
    const dst = mapReg(ctx.regmap, op.dest);
    ctx.rex(true, 0, 0, @intFromEnum(dst));
    ctx.byte(0xF7);
    ctx.modrm(0b11, 2, @intFromEnum(dst)); // NOT r/m64
}

fn emitDiv(ctx: *EmitContext, op: IROp, signed: bool) void {
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
    const dst = mapReg(ctx.regmap, op.dest);
    // Variable shift: count in CL
    ctx.rex(true, 0, 0, @intFromEnum(dst));
    ctx.byte(0xD3);
    ctx.modrm(0b11, shift_type, @intFromEnum(dst));
}

fn emitShiftImm(ctx: *EmitContext, op: IROp, shift_type: u4) void {
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

fn emitLoad(ctx: *EmitContext, op: IROp) void {
    const dst = mapReg(ctx.regmap, op.dest);
    const base = mapReg(ctx.regmap, op.src0);
    const offset = op.imm;

    if (offset == 0) {
        ctx.rex(true, @intFromEnum(dst), 0, @intFromEnum(base));
        ctx.byte(0x8B);
        // RBP/R13 with mod=00 encodes as RIP-relative on x86-64. Use mod=01 + 0-displacement.
        if (@intFromEnum(base) & 0x07 == 0b101) {
            ctx.modrm(0b01, @intFromEnum(dst), @intFromEnum(base));
            ctx.byte(0);
        } else {
            ctx.modrm(0b00, @intFromEnum(dst), @intFromEnum(base));
        }
    } else if (offset <= 0x7F) {
        ctx.rex(true, @intFromEnum(dst), 0, @intFromEnum(base));
        ctx.byte(0x8B);
        ctx.modrm(0b01, @intFromEnum(dst), @intFromEnum(base));
        ctx.byte(@truncate(offset));
    } else {
        ctx.rex(true, @intFromEnum(dst), 0, @intFromEnum(base));
        ctx.byte(0x8B);
        ctx.modrm(0b10, @intFromEnum(dst), @intFromEnum(base));
        ctx.bytes(std.mem.asBytes(&@as(i32, @bitCast(offset))));
    }
}

fn emitStore(ctx: *EmitContext, op: IROp) void {
    const base_ = mapReg(ctx.regmap, op.src0);
    const base: X86Reg = if (base_ == .rax) .r15 else base_;
    const src = mapReg(ctx.regmap, op.src1);
    const offset = op.imm;

    if (offset == 0) {
        ctx.rex(true, @intFromEnum(src), 0, @intFromEnum(base));
        ctx.byte(0x89);
        if (@intFromEnum(base) & 0x07 == 0b101) {
            ctx.modrm(0b01, @intFromEnum(src), @intFromEnum(base));
            ctx.byte(0);
        } else {
            ctx.modrm(0b00, @intFromEnum(src), @intFromEnum(base));
        }
    } else if (offset <= 0x7F) {
        ctx.rex(true, @intFromEnum(src), 0, @intFromEnum(base));
        ctx.byte(0x89);
        ctx.modrm(0b01, @intFromEnum(src), @intFromEnum(base));
        ctx.byte(@truncate(offset));
    } else {
        ctx.rex(true, @intFromEnum(src), 0, @intFromEnum(base));
        ctx.byte(0x89);
        ctx.modrm(0b10, @intFromEnum(src), @intFromEnum(base));
        ctx.bytes(std.mem.asBytes(&@as(i32, @bitCast(offset))));
    }
}

// ── Control flow ───────────────────────────────────────────────────

fn emitBranch(ctx: *EmitContext, op: IROp) void {
    if (op.flags == 0) {
        ctx.byte(0xE9);
        ctx.bytes(&[4]u8{ 0x00, 0x00, 0x00, 0x00 });
    } else {
        const t = mapReg(ctx.regmap, op.src0);
        ctx.byte(0xFF);
        ctx.modrm(0b11, 4, @intFromEnum(t));
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
    const t = mapReg(ctx.regmap, op.src0);
    ctx.byte(0xFF);
    ctx.modrm(0b11, 2, @intFromEnum(t));
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
    _ = ctx;
    _ = op;
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
    emitPopf(ctx); // pop into RAX
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
        .and_ => emitLogical(ctx, op, 0x21, 0x23),
        .or_ => emitLogical(ctx, op, 0x09, 0x0B),
        .xor_ => emitLogical(ctx, op, 0x31, 0x33),
        .not_ => emitNot(ctx, op),
        .neg_i64 => emitNeg(ctx, op),
        .mov_i64 => {
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

        .load_u64 => emitLoad(ctx, op),
        .store_u64 => emitStore(ctx, op),

        .br => emitBranch(ctx, op),
        .call => emitCall(ctx, op),
        .call_reg => emitCallReg(ctx, op),
        .ret_ => emitRet(ctx),
        .br_cond => emitBrCond(ctx, op),
        .ccmp => emitCCmp(ctx, op),

        .nzcv_update => emitNZCVUpdate(ctx, op),
        .nzcv_read => emitNZCVRead(ctx, op),

        .sp_get => {
            const dst = mapReg(ctx.regmap, op.dest);
            emitMovReg(ctx, dst, .r15);
        },
        .sp_put => {
            const src = mapReg(ctx.regmap, op.src0);
            emitMovReg(ctx, .r15, src);
        },

        else => ctx.byte(0xCC),
    }
    return ctx.offset - start;
}

pub fn emitBlock(buf: []u8, regmap: *const RegisterMap, ops: []const IROp) []u8 {
    var ctx = EmitContext.init(buf, regmap);
    for (ops) |op| _ = emitOp(&ctx, op);
    if (ctx.offset == 0) emitRet(&ctx);
    return buf[0..ctx.offset];
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
    const op = IROp{ .tag = .add_i64, .dest = 0, .src0 = 1, .src1 = 0x1F, .flags = 0, .imm = 42 };
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
    try std.testing.expectEqual(@as(u8, 0x46), emitted[2]);
    try std.testing.expectEqual(@as(u8, 16), emitted[3]);
}

test "emit NOT" {
    var code: [128]u8 = undefined;
    const op = IROp{ .tag = .not_, .dest = 0, .src0 = 0, .src1 = 0, .flags = 0, .imm = 0 };
    const emitted = emitBlock(&code, &DefaultMapping, &.{op});
    try std.testing.expectEqual(@as(u8, 0x48), emitted[0]);
    try std.testing.expectEqual(@as(u8, 0xF7), emitted[1]);
}
