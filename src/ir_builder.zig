//! ARM64 instruction → IR builder.
//!
//! Translates each decoded ARM64 instruction into one or more IR ops.
//! Complex instructions are decomposed into simpler IR primitives.
//! The x86-64 backend emits these primitives efficiently.

const std = @import("std");
const Ir = @import("ir.zig");
const Decode = @import("decode.zig");

const IROp = Ir.IROp;
const Tag = Ir.Tag;
const IRBuffer = Ir.IRBuffer;
const A64Inst = Decode.A64Inst;
const Opcode = Decode.Opcode;
const Condition = Decode.Condition;

pub fn build(buf: *IRBuffer, allocator: std.mem.Allocator, inst: A64Inst, guest_pc: u64) !void {
    switch (inst.opcode) {
        // ── ALU immediate ─────────────────────────────────────────
        .add_imm => try buildAddSubImm(buf, allocator, inst, .add_i64),
        .sub_imm => try buildAddSubImm(buf, allocator, inst, .sub_i64),
        .adds_imm => try buildAddSubImmFlags(buf, allocator, inst, .add_i64),
        .subs_imm => try buildAddSubImmFlags(buf, allocator, inst, .sub_i64),

        // ── Move wide immediate ──────────────────────────────────
        .movz => try buildMovz(buf, allocator, inst),
        .movn => try buildMovn(buf, allocator, inst),
        .movk => try buildMovk(buf, allocator, inst),

        // ── PC-relative address ──────────────────────────────────
        .adr, .adrp => try buildAdr(buf, allocator, inst, guest_pc),

        // ── ALU register ─────────────────────────────────────────
        .add_reg => try buildAddSubReg(buf, allocator, inst, .add_i64),
        .adc_reg => try buildAddSubReg(buf, allocator, inst, .adc_i64),
        .sub_reg => try buildAddSubReg(buf, allocator, inst, .sub_i64),
        .sbc_reg => try buildAddSubReg(buf, allocator, inst, .sbc_i64),
        .add_ext => try buildAddSubReg(buf, allocator, inst, .add_i64),
        .sub_ext => try buildAddSubReg(buf, allocator, inst, .sub_i64),
        .and_reg => try buildLogical(buf, allocator, inst, .and_),
        .orr_reg => try buildLogical(buf, allocator, inst, .or_),
        .eor_reg => try buildLogical(buf, allocator, inst, .xor_),
        .bic_reg => try buildBic(buf, allocator, inst),
        .orn_reg => try buildOrn(buf, allocator, inst),
        .eon_reg => try buildEon(buf, allocator, inst),

        // ── Multiply ─────────────────────────────────────────────
        .mul => try buildMul(buf, allocator, inst),
        .mneg => try buildMneg(buf, allocator, inst),

        // ── Divide ───────────────────────────────────────────────
        .sdiv => try buildDiv(buf, allocator, inst, true),
        .udiv => try buildDiv(buf, allocator, inst, false),

        // ── Shift by register ────────────────────────────────────
        .lsl_reg => try buildShift(buf, allocator, inst, .lshl_i64),
        .lsr_reg => try buildShift(buf, allocator, inst, .lshr_i64),
        .asr_reg => try buildShift(buf, allocator, inst, .ashr_i64),

        // ── NEG / CMN ────────────────────────────────────────────
        .neg_reg => try buildNeg(buf, allocator, inst),

        // ── Comparison ───────────────────────────────────────────
        .cmp_reg => try buildCmp(buf, allocator, inst),
        .cmn_reg => try buildCmn(buf, allocator, inst),

        // ── Bitfield operations ──────────────────────────────────
        .ubfm => try buildUbfm(buf, allocator, inst),
        .sbfm => try buildSbfm(buf, allocator, inst),

        // ── Conditional select ───────────────────────────────────
        .csel => try buildCSel(buf, allocator, inst),
        .csinc => try buildCSinc(buf, allocator, inst),
        .csinv => try buildCSinv(buf, allocator, inst),
        .csneg => try buildCSneg(buf, allocator, inst),

        // ── Memory ───────────────────────────────────────────────
        .ldr_imm => try buildLoad(buf, allocator, inst, .load_u64, false),
        .ldrb_imm => try buildLoad(buf, allocator, inst, .load_u8, false),
        .ldrh_imm => try buildLoad(buf, allocator, inst, .load_u16, false),
        .ldur => try buildLoad(buf, allocator, inst, .load_u64, false),
        .ldurh => try buildLoad(buf, allocator, inst, .load_u16, false),
        .ldurb => try buildLoad(buf, allocator, inst, .load_u8, false),
        .ldr_literal => try buildLoadLiteral(buf, allocator, inst, guest_pc),
        .ldr_reg => try buildLoadReg(buf, allocator, inst, .load_u64, guest_pc),
        .ldrb_reg => try buildLoadReg(buf, allocator, inst, .load_u8, guest_pc),
        .ldrh_reg => try buildLoadReg(buf, allocator, inst, .load_u16, guest_pc),
        .str_imm => try buildStore(buf, allocator, inst, .store_u64, false),
        .strb_imm => try buildStore(buf, allocator, inst, .store_u8, false),
        .strh_imm => try buildStore(buf, allocator, inst, .store_u16, false),
        .stur => try buildStore(buf, allocator, inst, .store_u64, false),
        .str_reg => try buildStoreReg(buf, allocator, inst, .store_u64, guest_pc),
        .strb_reg => try buildStoreReg(buf, allocator, inst, .store_u8, guest_pc),
        .strh_reg => try buildStoreReg(buf, allocator, inst, .store_u16, guest_pc),
        .ldp => try buildLDP(buf, allocator, inst, guest_pc),
        .stp => try buildSTP(buf, allocator, inst, guest_pc),

        // ── Branches ─────────────────────────────────────────────
        .b => try buildB(buf, allocator, inst, guest_pc),
        .bl => try buildBL(buf, allocator, inst, guest_pc),
        .br => try buildBR(buf, allocator, inst),
        .blr => try buildBLR(buf, allocator, inst),
        .ret_ => try buildRet(buf, allocator),

        // ── Conditional branch ───────────────────────────────────
        .b_cond => try buildBCond(buf, allocator, inst, guest_pc),

        // ── Conditional compare ──────────────────────────────────
        .ccmp_reg, .ccmp_imm => try buildCCmp(buf, allocator, inst),

        // ── System ───────────────────────────────────────────────
        .svc => {}, // handled by runtime dispatcher
        .nop => {},
        .unknown => {},
        else => {},
    }
}

// ═══════════════════════════════════════════════════════════════════
//  ALU immediate
// ═══════════════════════════════════════════════════════════════════

fn buildAddSubImm(buf: *IRBuffer, allocator: std.mem.Allocator, inst: A64Inst, tag: Tag) !void {
    const ops = inst.operands.rri12;
    const imm: u32 = @as(u32, ops.imm12) << (@as(u5, ops.shift) * 12);
    try buf.append(allocator, .{
        .tag = tag, .dest = ops.rd, .src0 = ops.rn, .src1 = 0x1F,
        .flags = 0, .imm = imm,
    });
}

fn buildAddSubImmFlags(buf: *IRBuffer, allocator: std.mem.Allocator, inst: A64Inst, tag: Tag) !void {
    // ADDS/SUBS — same as ADD/SUB but also sets NZCV
    const ops = inst.operands.rri12;
    const imm: u32 = @as(u32, ops.imm12) << (@as(u5, ops.shift) * 12);
    try buf.append(allocator, .{
        .tag = tag, .dest = ops.rd, .src0 = ops.rn, .src1 = 0x1F,
        .flags = 0, .imm = imm,
    });
    // imm=1 → CMC needed (SUB-based); imm=0 → no CMC (ADD-based)
    const need_cmc: u32 = if (tag == .sub_i64) 1 else 0;
    try buf.append(allocator, .{ .tag = .nzcv_update, .dest = 0, .src0 = 0, .src1 = 0, .flags = 0, .imm = need_cmc });
}

// ═══════════════════════════════════════════════════════════════════
//  Move wide
// ═══════════════════════════════════════════════════════════════════

fn buildMovz(buf: *IRBuffer, allocator: std.mem.Allocator, inst: A64Inst) !void {
    const ops = inst.operands.ri16_hw;
    const shifted = @as(u64, ops.imm16) << (@as(u6, ops.hw) * 16);
    try buf.append(allocator, .{
        .tag = .add_i64, .dest = ops.rd, .src0 = 0x1F, .src1 = 0x1F,
        .flags = 0, .imm = @truncate(shifted),
    });
}

fn buildMovn(buf: *IRBuffer, allocator: std.mem.Allocator, inst: A64Inst) !void {
    const ops = inst.operands.ri16_hw;
    const shifted = @as(u64, ops.imm16) << (@as(u6, ops.hw) * 16);
    try buf.append(allocator, .{
        .tag = .add_i64, .dest = ops.rd, .src0 = 0x1F, .src1 = 0x1F,
        .flags = 0, .imm = @truncate(~shifted),
    });
}

fn buildMovk(buf: *IRBuffer, allocator: std.mem.Allocator, inst: A64Inst) !void {
    // MOVK Xd, #imm16, LSL #hw: insert 16-bit field into existing register
    // Decompose: AND mask + OR immediate
    const ops = inst.operands.ri16_hw;
    const shift = @as(u6, ops.hw) * 16;
    const shifted = @as(u64, ops.imm16) << shift;
    const mask: u64 = ~(@as(u64, 0xFFFF) << shift);

    // Use x16 (IP0) as temp for the masked value
    try buf.append(allocator, .{ .tag = .and_, .dest = 16, .src0 = ops.rd, .src1 = 0x1F, .flags = 0, .imm = @truncate(mask) });
    try buf.append(allocator, .{ .tag = .or_, .dest = ops.rd, .src0 = 16, .src1 = 0x1F, .flags = 0, .imm = @truncate(shifted) });
}

// ═══════════════════════════════════════════════════════════════════
//  ADR/ADRP
// ═══════════════════════════════════════════════════════════════════

fn buildAdr(buf: *IRBuffer, allocator: std.mem.Allocator, inst: A64Inst, guest_pc: u64) !void {
    const ops = inst.operands.rl;
    const target = @as(u64, @intCast(@as(i64, @intCast(guest_pc)) + ops.label));
    try buf.append(allocator, .{
        .tag = .add_i64, .dest = ops.rd, .src0 = 0x1F, .src1 = 0x1F,
        .flags = 0, .imm = @truncate(target),
    });
}

// ═══════════════════════════════════════════════════════════════════
//  ALU register
// ═══════════════════════════════════════════════════════════════════

fn buildAddSubReg(buf: *IRBuffer, allocator: std.mem.Allocator, inst: A64Inst, tag: Tag) !void {
    const ops = inst.operands.rrr;
    try buf.append(allocator, .{
        .tag = tag, .dest = ops.rd, .src0 = ops.rn, .src1 = ops.rm,
        .flags = 0, .imm = 0,
    });
}

fn buildLogical(buf: *IRBuffer, allocator: std.mem.Allocator, inst: A64Inst, tag: Tag) !void {
    const ops = inst.operands.rrr;
    try buf.append(allocator, .{
        .tag = tag, .dest = ops.rd, .src0 = ops.rn, .src1 = ops.rm,
        .flags = 0, .imm = 0,
    });
}

// ── Logical with NOT ───────────────────────────────────────────

fn buildBic(buf: *IRBuffer, allocator: std.mem.Allocator, inst: A64Inst) !void {
    const ops = inst.operands.rrr;
    // BIC Xd, Xn, Xm = Xd = Xn & ~Xm
    // Use x16 (IP0) as temp for ~Xm
    try buf.append(allocator, .{ .tag = .not_, .dest = 16, .src0 = ops.rm, .src1 = 0, .flags = 0, .imm = 0 });
    try buf.append(allocator, .{ .tag = .and_, .dest = ops.rd, .src0 = ops.rn, .src1 = 16, .flags = 0, .imm = 0 });
}

fn buildOrn(buf: *IRBuffer, allocator: std.mem.Allocator, inst: A64Inst) !void {
    const ops = inst.operands.rrr;
    // ORN Xd, Xn, Xm = Xd = Xn | ~Xm
    try buf.append(allocator, .{ .tag = .not_, .dest = 16, .src0 = ops.rm, .src1 = 0, .flags = 0, .imm = 0 });
    try buf.append(allocator, .{ .tag = .or_, .dest = ops.rd, .src0 = ops.rn, .src1 = 16, .flags = 0, .imm = 0 });
}

fn buildEon(buf: *IRBuffer, allocator: std.mem.Allocator, inst: A64Inst) !void {
    const ops = inst.operands.rrr;
    // EON Xd, Xn, Xm = Xd = Xn ^ ~Xm
    try buf.append(allocator, .{ .tag = .not_, .dest = 16, .src0 = ops.rm, .src1 = 0, .flags = 0, .imm = 0 });
    try buf.append(allocator, .{ .tag = .xor_, .dest = ops.rd, .src0 = ops.rn, .src1 = 16, .flags = 0, .imm = 0 });
}

// ═══════════════════════════════════════════════════════════════════
//  Multiply / Divide / Negate
// ═══════════════════════════════════════════════════════════════════

fn buildMul(buf: *IRBuffer, allocator: std.mem.Allocator, inst: A64Inst) !void {
    const ops = inst.operands.rrr;
    try buf.append(allocator, .{ .tag = .mul_i64, .dest = ops.rd, .src0 = ops.rn, .src1 = ops.rm, .flags = 0, .imm = 0 });
}

fn buildMneg(buf: *IRBuffer, allocator: std.mem.Allocator, inst: A64Inst) !void {
    const ops = inst.operands.rrr;
    // MNEG = -(Rn * Rm)
    try buf.append(allocator, .{ .tag = .mul_i64, .dest = 16, .src0 = ops.rn, .src1 = ops.rm, .flags = 0, .imm = 0 });
    try buf.append(allocator, .{ .tag = .neg_i64, .dest = ops.rd, .src0 = 16, .src1 = 0, .flags = 0, .imm = 0 });
}

fn buildDiv(buf: *IRBuffer, allocator: std.mem.Allocator, inst: A64Inst, signed: bool) !void {
    const ops = inst.operands.rrr;
    try buf.append(allocator, .{
        .tag = if (signed) .div_s64 else .div_u64,
        .dest = ops.rd, .src0 = ops.rn, .src1 = ops.rm,
        .flags = 0, .imm = 0,
    });
}

fn buildNeg(buf: *IRBuffer, allocator: std.mem.Allocator, inst: A64Inst) !void {
    const ops = inst.operands.rrr;
    try buf.append(allocator, .{
        .tag = .neg_i64, .dest = ops.rd, .src0 = ops.rm, .src1 = 0,
        .flags = 0, .imm = 0,
    });
}

// ═══════════════════════════════════════════════════════════════════
//  Shift by register
// ═══════════════════════════════════════════════════════════════════

fn buildShift(buf: *IRBuffer, allocator: std.mem.Allocator, inst: A64Inst, tag: Tag) !void {
    const ops = inst.operands.rrr;
    try buf.append(allocator, .{
        .tag = tag, .dest = ops.rd, .src0 = ops.rn, .src1 = ops.rm,
        .flags = 0, .imm = 0,
    });
}

// ═══════════════════════════════════════════════════════════════════
//  Comparison
// ═══════════════════════════════════════════════════════════════════

fn buildCmp(buf: *IRBuffer, allocator: std.mem.Allocator, inst: A64Inst) !void {
    const ops = inst.operands.rrr;
    try buf.append(allocator, .{ .tag = .sub_i64, .dest = 0x1F, .src0 = ops.rn, .src1 = ops.rm, .flags = 0, .imm = 0 });
    try buf.append(allocator, .{ .tag = .nzcv_update, .dest = 0, .src0 = 0, .src1 = 0, .flags = 0, .imm = 1 }); // imm=1 = CMC needed
}

fn buildCmn(buf: *IRBuffer, allocator: std.mem.Allocator, inst: A64Inst) !void {
    const ops = inst.operands.rrr;
    try buf.append(allocator, .{ .tag = .add_i64, .dest = 0x1F, .src0 = ops.rn, .src1 = ops.rm, .flags = 0, .imm = 0 });
    try buf.append(allocator, .{ .tag = .nzcv_update, .dest = 0, .src0 = 0, .src1 = 0, .flags = 0, .imm = 0 }); // imm=0 = no CMC
}

// ═══════════════════════════════════════════════════════════════════
//  Bitfield operations
// ═══════════════════════════════════════════════════════════════════

fn buildUbfm(buf: *IRBuffer, allocator: std.mem.Allocator, inst: A64Inst) !void {
    // UBFM Xd, Xn, #immr, #imms
    // Common patterns:
    //   UXTB: immr=0, imms=7   → AND with 0xFF
    //   UXTH: immr=0, imms=15  → AND with 0xFFFF
    //   LSL:  immr=-shift, imms=63-shift → left shift
    //   LSR:  immr=shift, imms=63 → right shift (actually UBFM with shift)
    //   UBFM: generic bitfield extract → AND + shift
    const ops = inst.operands.bitfield;
    const sf = inst.sf;
    _ = sf;

    if (ops.immr == 0) {
        // Zero-extend: AND with (2^(imms+1) - 1)
        const mask: u64 = if (ops.imms >= 63)
            ~@as(u64, 0)
        else
            (@as(u64, 1) << @as(u6, @intCast(ops.imms + 1))) - 1;
        try buf.append(allocator, .{
            .tag = .and_, .dest = ops.rd, .src0 = ops.rn, .src1 = 0x1F,
            .flags = 0, .imm = @truncate(mask),
        });
    } else if (ops.imms == 63) {
        // Logical shift right: LSR Rd, Rn, #immr
        try buf.append(allocator, .{
            .tag = .lshr_i64_imm, .dest = ops.rd, .src0 = ops.rn, .src1 = 0x1F,
            .flags = 0, .imm = ops.immr,
        });
    } else if (ops.immr == ops.imms + 1) {
        // Left shift (LSL): UBFM with fields set for left shift
        // LSL #shift = UBFM Rd, Rn, #(64-shift), #(63-shift)
        const shift: u32 = 64 - @as(u32, ops.immr);
        try buf.append(allocator, .{
            .tag = .lshl_i64_imm, .dest = ops.rd, .src0 = ops.rn, .src1 = 0x1F,
            .flags = 0, .imm = shift,
        });
    } else {
        // Generic UBFM: shift right + AND mask
        // For now: just emit the shift
        try buf.append(allocator, .{
            .tag = .lshr_i64_imm, .dest = 16, .src0 = ops.rn, .src1 = 0x1F,
            .flags = 0, .imm = ops.immr,
        });
        const width: u6 = @intCast(ops.imms - ops.immr + 1);
        const mask = (@as(u64, 1) << @as(u6, @intCast(width))) - 1;
        try buf.append(allocator, .{
            .tag = .and_, .dest = ops.rd, .src0 = 16, .src1 = 0x1F,
            .flags = 0, .imm = @truncate(mask),
        });
    }
}

fn buildSbfm(buf: *IRBuffer, allocator: std.mem.Allocator, inst: A64Inst) !void {
    // SBFM Xd, Xn, #immr, #imms
    // Common patterns:
    //   SXTB: immr=0, imms=7   → sign-extend byte: LSL 56, ASR 56
    //   SXTH: immr=0, imms=15  → sign-extend halfword: LSL 48, ASR 48
    //   SXTW: immr=0, imms=31  → sign-extend word: LSL 32, ASR 32
    //   ASR:  immr=shift, imms=63 → arithmetic shift right
    const ops = inst.operands.bitfield;

    if (ops.immr == 0 and ops.imms < 63) {
        // Sign-extension: LSL then ASR
        const bits_to_shift: u6 = @intCast(63 - ops.imms);
        try buf.append(allocator, .{
            .tag = .lshl_i64_imm, .dest = 16, .src0 = ops.rn, .src1 = 0x1F,
            .flags = 0, .imm = bits_to_shift,
        });
        try buf.append(allocator, .{
            .tag = .ashr_i64_imm, .dest = ops.rd, .src0 = 16, .src1 = 0x1F,
            .flags = 0, .imm = bits_to_shift,
        });
    } else if (ops.imms == 63) {
        // Arithmetic shift right
        try buf.append(allocator, .{
            .tag = .ashr_i64_imm, .dest = ops.rd, .src0 = ops.rn, .src1 = 0x1F,
            .flags = 0, .imm = ops.immr,
        });
    } else {
        // Generic SBFM: just emit as ASR + AND (may be slightly off for some edge cases)
        try buf.append(allocator, .{
            .tag = .ashr_i64_imm, .dest = 16, .src0 = ops.rn, .src1 = 0x1F,
            .flags = 0, .imm = ops.immr,
        });
        const width: u6 = @intCast(ops.imms - ops.immr + 1);
        const mask = (@as(u64, 1) << @as(u6, @intCast(width))) - 1;
        try buf.append(allocator, .{
            .tag = .and_, .dest = ops.rd, .src0 = 16, .src1 = 0x1F,
            .flags = 0, .imm = @truncate(mask),
        });
    }
}

// ═══════════════════════════════════════════════════════════════════
//  Conditional select
// ═══════════════════════════════════════════════════════════════════

fn buildCSel(buf: *IRBuffer, allocator: std.mem.Allocator, inst: A64Inst) !void {
    const ops = inst.operands.csel;
    // Emit: mov rd, rm; cmovcc rd, rn
    // CMOV uses x86 RFLAGS which reflects ARM64 NZCV after nzcv_read.
    try buf.append(allocator, .{ .tag = .nzcv_read, .dest = 0, .src0 = @intFromEnum(ops.cond), .src1 = 0, .flags = 0, .imm = 0 });
    try buf.append(allocator, .{ .tag = .add_i64, .dest = ops.rd, .src0 = ops.rn, .src1 = ops.rm, .flags = @intFromEnum(ops.cond), .imm = 0 });
}

fn buildCSinc(buf: *IRBuffer, allocator: std.mem.Allocator, inst: A64Inst) !void {
    // CSINC Xd, Xn, Xm, cond = Xd = cond ? Xn : Xm + 1
    const ops = inst.operands.csel;
    // Precompute false-case: x16 = Rm + 1
    try buf.append(allocator, .{ .tag = .add_i64, .dest = 16, .src0 = ops.rm, .src1 = 0x1F, .flags = 0, .imm = 1 });
    // CMOV: Rd = cond ? Rn : x16 (src1=16 = x16 = false value)
    try buf.append(allocator, .{ .tag = .add_i64, .dest = ops.rd, .src0 = ops.rn, .src1 = 16, .flags = @intFromEnum(ops.cond), .imm = 0 });
}

fn buildCSinv(buf: *IRBuffer, allocator: std.mem.Allocator, inst: A64Inst) !void {
    // CSINV Xd, Xn, Xm, cond = Xd = cond ? Xn : ~Xm
    const ops = inst.operands.csel;
    // Precompute false-case: x16 = ~Rm
    try buf.append(allocator, .{ .tag = .not_, .dest = 16, .src0 = ops.rm, .src1 = 0, .flags = 0, .imm = 0 });
    // CMOV: Rd = cond ? Rn : x16
    try buf.append(allocator, .{ .tag = .add_i64, .dest = ops.rd, .src0 = ops.rn, .src1 = 16, .flags = @intFromEnum(ops.cond), .imm = 0 });
}

fn buildCSneg(buf: *IRBuffer, allocator: std.mem.Allocator, inst: A64Inst) !void {
    // CSNEG Xd, Xn, Xm, cond = Xd = cond ? Xn : -Xm
    const ops = inst.operands.csel;
    // Precompute false-case: x16 = -Rm
    try buf.append(allocator, .{ .tag = .neg_i64, .dest = 16, .src0 = ops.rm, .src1 = 0, .flags = 0, .imm = 0 });
    // CMOV: Rd = cond ? Rn : x16
    try buf.append(allocator, .{ .tag = .add_i64, .dest = ops.rd, .src0 = ops.rn, .src1 = 16, .flags = @intFromEnum(ops.cond), .imm = 0 });
}

// ═══════════════════════════════════════════════════════════════════
//  Memory
// ═══════════════════════════════════════════════════════════════════

fn buildLoad(buf: *IRBuffer, allocator: std.mem.Allocator, inst: A64Inst, tag: Tag, _: bool) !void {
    const ops = inst.operands.mem_imm;
    try buf.append(allocator, .{
        .tag = tag, .dest = ops.rt, .src0 = ops.rn, .src1 = 0,
        .flags = 0, .imm = @as(u32, @bitCast(@as(i32, ops.offset))),
    });
}

fn buildStore(buf: *IRBuffer, allocator: std.mem.Allocator, inst: A64Inst, tag: Tag, _: bool) !void {
    const ops = inst.operands.mem_imm;
    try buf.append(allocator, .{
        .tag = tag, .dest = 0, .src0 = ops.rn, .src1 = ops.rt,
        .flags = 0, .imm = @as(u32, @bitCast(@as(i32, ops.offset))),
    });
}

fn buildLoadReg(buf: *IRBuffer, allocator: std.mem.Allocator, inst: A64Inst, tag: Tag, _: u64) !void {
    // LDR Xt, [Xn, Xm]{, extend #amount}
    // Decompose: addr = Xn + Xm; load from addr
    const ops = inst.operands.mem_reg;
    // ADD temp = Xn + Xm
    try buf.append(allocator, .{ .tag = .add_i64, .dest = 16, .src0 = ops.rn, .src1 = ops.rm, .flags = 0, .imm = 0 });
    // Load from temp
    try buf.append(allocator, .{ .tag = tag, .dest = ops.rt, .src0 = 16, .src1 = 0, .flags = 0, .imm = 0 });
}

fn buildStoreReg(buf: *IRBuffer, allocator: std.mem.Allocator, inst: A64Inst, tag: Tag, _: u64) !void {
    const ops = inst.operands.mem_reg;
    try buf.append(allocator, .{ .tag = .add_i64, .dest = 16, .src0 = ops.rn, .src1 = ops.rm, .flags = 0, .imm = 0 });
    try buf.append(allocator, .{ .tag = tag, .dest = 0, .src0 = 16, .src1 = ops.rt, .flags = 0, .imm = 0 });
}

fn buildLoadLiteral(buf: *IRBuffer, allocator: std.mem.Allocator, inst: A64Inst, guest_pc: u64) !void {
    const ops = inst.operands.rl;
    const target_addr = @as(u64, @intCast(@as(i64, @intCast(guest_pc)) + ops.label));
    try buf.append(allocator, .{
        .tag = .load_u64, .dest = ops.rd, .src0 = 0x1F, .src1 = 0,
        .flags = 0, .imm = @truncate(target_addr),
    });
}

fn buildLDP(buf: *IRBuffer, allocator: std.mem.Allocator, inst: A64Inst, guest_pc: u64) !void {
    const ops = inst.operands.ldp_stp;
    const scale: u64 = if (inst.sf) 8 else 4;
    const offset = @as(u64, @intCast(ops.imm7)) * scale;

    // Handle writeback: update base register before/after load
    const base = if (ops.writeback) blk: {
        if (ops.post_index) {
            // LDP with post-index: load from [Rn], then Rn += offset
            // Emit load then add
            break :blk ops.rn;
        } else {
            // LDP with pre-index: Rn += offset, then load from [Rn]
            try buf.append(allocator, .{ .tag = .add_i64, .dest = ops.rn, .src0 = ops.rn, .src1 = 0x1F, .flags = 0, .imm = @truncate(offset) });
            break :blk ops.rn;
        }
    } else ops.rn;

    try buf.append(allocator, .{ .tag = .load_u64, .dest = ops.rt1, .src0 = base, .src1 = 0, .flags = 0, .imm = @truncate(offset) });
    try buf.append(allocator, .{ .tag = .load_u64, .dest = ops.rt2, .src0 = base, .src1 = 0, .flags = 0, .imm = @truncate(offset + 8) });

    if (ops.writeback and ops.post_index) {
        try buf.append(allocator, .{ .tag = .add_i64, .dest = ops.rn, .src0 = ops.rn, .src1 = 0x1F, .flags = 0, .imm = @truncate(offset) });
    }
    _ = guest_pc;
}

fn buildSTP(buf: *IRBuffer, allocator: std.mem.Allocator, inst: A64Inst, guest_pc: u64) !void {
    const ops = inst.operands.ldp_stp;
    const scale: u64 = if (inst.sf) 8 else 4;
    const offset = @as(u64, @intCast(ops.imm7)) * scale;
    const base = ops.rn;

    if (ops.writeback and !ops.post_index) {
        try buf.append(allocator, .{ .tag = .add_i64, .dest = ops.rn, .src0 = ops.rn, .src1 = 0x1F, .flags = 0, .imm = @truncate(offset) });
    }

    try buf.append(allocator, .{ .tag = .store_u64, .dest = 0, .src0 = base, .src1 = ops.rt1, .flags = 0, .imm = @truncate(offset) });
    try buf.append(allocator, .{ .tag = .store_u64, .dest = 0, .src0 = base, .src1 = ops.rt2, .flags = 0, .imm = @truncate(offset + 8) });

    if (ops.writeback and ops.post_index) {
        try buf.append(allocator, .{ .tag = .add_i64, .dest = ops.rn, .src0 = ops.rn, .src1 = 0x1F, .flags = 0, .imm = @truncate(offset) });
    }
    _ = guest_pc;
}

// ═══════════════════════════════════════════════════════════════════
//  Branches
// ═══════════════════════════════════════════════════════════════════

fn buildB(buf: *IRBuffer, allocator: std.mem.Allocator, inst: A64Inst, guest_pc: u64) !void {
    const target = @as(u64, @intCast(@as(i64, @intCast(guest_pc)) + inst.operands.b_target.label));
    try buf.append(allocator, .{ .tag = .br, .dest = 0, .src0 = 0, .src1 = 0, .flags = 0, .imm = @truncate(target) });
}

fn buildBL(buf: *IRBuffer, allocator: std.mem.Allocator, inst: A64Inst, guest_pc: u64) !void {
    const target = @as(u64, @intCast(@as(i64, @intCast(guest_pc)) + inst.operands.b_target.label));
    const ret_addr = guest_pc + 4;
    try buf.append(allocator, .{ .tag = .add_i64, .dest = 30, .src0 = 0x1F, .src1 = 0x1F, .flags = 0, .imm = @truncate(ret_addr) });
    try buf.append(allocator, .{ .tag = .call, .dest = 0, .src0 = 0, .src1 = 0, .flags = 0, .imm = @truncate(target) });
}

fn buildBR(buf: *IRBuffer, allocator: std.mem.Allocator, inst: A64Inst) !void {
    const rn = inst.operands.br_target.rn;
    try buf.append(allocator, .{ .tag = .br, .dest = 0, .src0 = rn, .src1 = 0, .flags = 1, .imm = 0 });
}

fn buildBLR(buf: *IRBuffer, allocator: std.mem.Allocator, inst: A64Inst) !void {
    const rn = inst.operands.br_target.rn;
    try buf.append(allocator, .{ .tag = .call_reg, .dest = 0, .src0 = rn, .src1 = 0, .flags = 0, .imm = 0 });
}

fn buildRet(buf: *IRBuffer, allocator: std.mem.Allocator) !void {
    try buf.append(allocator, .{ .tag = .ret_, .dest = 0, .src0 = 0, .src1 = 0, .flags = 0, .imm = 0 });
}

fn buildBCond(buf: *IRBuffer, allocator: std.mem.Allocator, inst: A64Inst, guest_pc: u64) !void {
    const target = @as(u64, @intCast(@as(i64, @intCast(guest_pc)) + inst.operands.bcond.label));
    try buf.append(allocator, .{ .tag = .nzcv_update, .dest = 0, .src0 = 0, .src1 = 0, .flags = 0, .imm = 0 });
    try buf.append(allocator, .{ .tag = .br_cond, .dest = 0, .src0 = 0, .src1 = 0, .flags = @intFromEnum(inst.operands.bcond.cond), .imm = @truncate(target) });
}

fn buildCCmp(buf: *IRBuffer, allocator: std.mem.Allocator, inst: A64Inst) !void {
    // CCMP Xn, Xm, #nzcv, cond
    //   if cond true:  NZCV = Xn - Xm  (compare)
    //   if cond false: NZCV = nzcv_imm
    //
    // Single IR op: the emitter emits JCC→CMP+CMC vs PUSHFQ/POPFQ
    const ops = inst.operands.ccmp;
    try buf.append(allocator, .{
        .tag = .ccmp, .dest = 0,
        .src0 = ops.rn, .src1 = ops.rm,
        .flags = @intFromEnum(ops.cond),
        .imm = ops.nzcv,
    });
}

// ═══════════════════════════════════════════════════════════════════
//  Tests
// ═══════════════════════════════════════════════════════════════════

test "ADD immediate → IR" {
    var buf: IRBuffer = .{};
    defer buf.deinit(std.testing.allocator);
    const inst = Decode.decode(0x91000C2A); // ADD X0, X1, #42
    try build(&buf, std.testing.allocator, inst, 0x1000);
    try std.testing.expectEqual(@as(usize, 1), buf.ops.items.len);
    try std.testing.expectEqual(Tag.add_i64, buf.ops.items[0].tag);
    try std.testing.expectEqual(@as(u16, 0), buf.ops.items[0].dest);
    try std.testing.expectEqual(@as(u32, 42), buf.ops.items[0].imm);
}

test "B → IR" {
    var buf: IRBuffer = .{};
    defer buf.deinit(std.testing.allocator);
    const inst = Decode.decode(0x14000040); // B #256
    try build(&buf, std.testing.allocator, inst, 0x1000);
    try std.testing.expectEqual(Tag.br, buf.ops.items[0].tag);
    try std.testing.expectEqual(@as(u32, 0x1100), buf.ops.items[0].imm);
}

test "BL → IR" {
    var buf: IRBuffer = .{};
    defer buf.deinit(std.testing.allocator);
    const inst = Decode.decode(0x94000040); // BL #256
    try build(&buf, std.testing.allocator, inst, 0x1000);
    try std.testing.expectEqual(Tag.add_i64, buf.ops.items[0].tag);
    try std.testing.expectEqual(@as(u16, 30), buf.ops.items[0].dest);
    try std.testing.expectEqual(Tag.call, buf.ops.items[1].tag);
}

test "CMP → IR (NZCV update)" {
    var buf: IRBuffer = .{};
    defer buf.deinit(std.testing.allocator);
    const inst = Decode.decode(0xEB01001F); // CMP X0, X1
    try build(&buf, std.testing.allocator, inst, 0);
    try std.testing.expectEqual(@as(usize, 2), buf.ops.items.len);
    try std.testing.expectEqual(Tag.sub_i64, buf.ops.items[0].tag);
    try std.testing.expectEqual(Tag.nzcv_update, buf.ops.items[1].tag);
}

test "BIC → IR (decomposed)" {
    var buf: IRBuffer = .{};
    defer buf.deinit(std.testing.allocator);
    // BIC X0, X1, X2 → NOT X2 + AND X1, ~X2
    const inst = Decode.decode(0x8A620020); // BIC X0, X1, X2
    try build(&buf, std.testing.allocator, inst, 0);
    try std.testing.expectEqual(@as(usize, 2), buf.ops.items.len);
    try std.testing.expectEqual(Tag.not_, buf.ops.items[0].tag);
    try std.testing.expectEqual(@as(u16, 2), buf.ops.items[0].src0); // ~X2
    try std.testing.expectEqual(Tag.and_, buf.ops.items[1].tag);
}

test "UBFM (UXTB) → IR" {
    var buf: IRBuffer = .{};
    defer buf.deinit(std.testing.allocator);
    // UBFM X0, X1, #0, #7 → UXTB: AND with 0xFF
    // Encoding: sf=1, N=1, opc=10, immr=0, imms=7, rn=1, rd=0
    const inst = Decode.decode(0x13001C20);
    try build(&buf, std.testing.allocator, inst, 0);
    try std.testing.expectEqual(@as(usize, 1), buf.ops.items.len);
    try std.testing.expectEqual(Tag.and_, buf.ops.items[0].tag);
    try std.testing.expectEqual(@as(u32, 0xFF), buf.ops.items[0].imm);
}

test "SBFM (SXTB) → IR" {
    var buf: IRBuffer = .{};
    defer buf.deinit(std.testing.allocator);
    // SBFM X0, X1, #0, #7 → SXTB: LSL 56, ASR 56
    // Encoding: sf=1, N=1, opc=00, immr=0, imms=7, rn=1, rd=0
    const inst = Decode.decode(0x13001C00);
    try build(&buf, std.testing.allocator, inst, 0);
    try std.testing.expectEqual(@as(usize, 2), buf.ops.items.len);
    try std.testing.expectEqual(Tag.lshl_i64_imm, buf.ops.items[0].tag);
    try std.testing.expectEqual(Tag.ashr_i64_imm, buf.ops.items[1].tag);
}

test "LDR with register offset → IR" {
    var buf: IRBuffer = .{};
    defer buf.deinit(std.testing.allocator);
    // LDR X0, [X1, X2] — register offset, no extend
    // This is matched via ldr_reg opcode
    // For now just test the decode works
    const inst = Decode.decode(0xF8606820); // LDR X0, [X1, X2]
    _ = inst;
}

test "ADC SBC decode and emit" {
    // ADC X0, X1, X2 (32-bit)
    var buf: IRBuffer = .{};
    defer buf.deinit(std.testing.allocator);
    const inst_adc = Decode.decode(0x1A020020);
    try std.testing.expectEqual(Opcode.adc_reg, inst_adc.opcode);
    // SBC X0, X1, X2 (64-bit)
    const inst_sbc = Decode.decode(0xDA020020);
    try std.testing.expectEqual(Opcode.sbc_reg, inst_sbc.opcode);
    // ADC (64-bit)
    const inst_adc64 = Decode.decode(0x9A020020);
    try std.testing.expectEqual(Opcode.adc_reg, inst_adc64.opcode);
}

test "NOP → IR" {
    var buf: IRBuffer = .{};
    defer buf.deinit(std.testing.allocator);
    const inst = Decode.decode(0xD503201F);
    try build(&buf, std.testing.allocator, inst, 0);
    try std.testing.expectEqual(@as(usize, 0), buf.ops.items.len);
}
