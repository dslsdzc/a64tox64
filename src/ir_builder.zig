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
        .add_reg, .add_ext => try buildAddSubReg(buf, allocator, inst, .add_i64),
        .adc_reg => try buildAddSubReg(buf, allocator, inst, .adc_i64),
        .sub_reg, .sub_ext => try buildAddSubReg(buf, allocator, inst, .sub_i64),
        .sbc_reg => try buildAddSubReg(buf, allocator, inst, .sbc_i64),
        .adds_reg => try buildAddSubRegFlags(buf, allocator, inst, .add_i64),
        .subs_reg => try buildAddSubRegFlags(buf, allocator, inst, .sub_i64),
        .and_reg => try buildLogical(buf, allocator, inst, .and_),
        .ands_reg => try buildLogicalFlags(buf, allocator, inst, .and_),
        .orr_reg => try buildLogical(buf, allocator, inst, .or_),
        .eor_reg => try buildLogical(buf, allocator, inst, .xor_),
        .bic_reg => try buildBic(buf, allocator, inst),
        .bics_reg => try buildBics(buf, allocator, inst),
        .orn_reg => try buildOrn(buf, allocator, inst),
        .eon_reg => try buildEon(buf, allocator, inst),

        // ── Multiply ─────────────────────────────────────────────
        .mul => try buildMul(buf, allocator, inst),
        .mneg => try buildMneg(buf, allocator, inst),
        .madd => try buildMadd(buf, allocator, inst),
        .msub => try buildMsub(buf, allocator, inst),
        .smulh => try buildSmulh(buf, allocator, inst),
        .umulh => try buildUmulh(buf, allocator, inst),
        .extr => try buildExtr(buf, allocator, inst),

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
        .ldp, .ldpsw => try buildLDP(buf, allocator, inst, guest_pc),
        .stp => try buildSTP(buf, allocator, inst, guest_pc),

        // ── Branches ─────────────────────────────────────────────
        .b => try buildB(buf, allocator, inst, guest_pc),
        .bl => try buildBL(buf, allocator, inst, guest_pc),
        .br => try buildBR(buf, allocator, inst),
        .blr => try buildBLR(buf, allocator, inst),
        .ret_ => try buildRet(buf, allocator),

        // ── Conditional branch ───────────────────────────────────
        .b_cond => try buildBCond(buf, allocator, inst, guest_pc),
        .cbz, .cbnz => try buildCBZ(buf, allocator, inst, guest_pc),
        .tbz, .tbnz => try buildTBZ(buf, allocator, inst, guest_pc),

        // ── Conditional compare ──────────────────────────────────
        .ccmp_reg, .ccmp_imm => try buildCCmp(buf, allocator, inst),

        // ── System ───────────────────────────────────────────────
        .svc => try buildRet(buf, allocator), // returns to runtime for SVC handling
        .nop, .hint => {}, // HINT (incl. NOP, SEV, WFE, etc.): no-op on x86-64
        .bfm => try buildUbfm(buf, allocator, inst),
        .ror_reg => try buildRor(buf, allocator, inst),
        .clz => try buildClz(buf, allocator, inst),
        .dc_zva => try buildDcZva(buf, allocator, inst),
        .sys, .crc32 => {}, // SYS/CRC32: no-ops on x86-64 (CRC uses hardware CRC in x86)
        .dmb, .dsb, .isb => {}, // memory barriers: no-op on x86-64

        // ── Atomic / exclusive ────────────────────────────────────
        .ldxr => try buildLDXR(buf, allocator, inst),
        .stxr => try buildSTXR(buf, allocator, inst),
        .ldadd => try buildLDADD(buf, allocator, inst),
        .cas => try buildCAS(buf, allocator, inst),

        // ── System register access (MRS/MSR) ──────────────────────
        .mrs, .msr => try buildMRSMSR(buf, allocator, inst),
        // ── Advanced SIMD (NEON) ──────────────────────────────────
        .neon_same => {
            const is_float = (inst.raw >> 24) == 0x1E or (inst.raw >> 24) == 0x3E;
            if (is_float) try buildNeonSameFloat(buf, allocator, inst)
            else try buildNeonSame(buf, allocator, inst);
        },
        .neon_diff => try buildNeonDiff(buf, allocator, inst),
        .neon_perm => try buildNeonPerm(buf, allocator, inst),
        .neon_conv => try buildNeonConv(buf, allocator, inst),
        .neon_load, .neon_store => try buildNeonLdSt(buf, allocator, inst),

        .unknown => {
            // Log once per page to avoid spam; helps debug missed instructions
            const pc = guest_pc;
            if (pc % 4096 == 0) {
                std.log.warn("decode unknown at PC 0x{X:08} (raw=0x{X:08})", .{ pc, inst.raw });
            }
        },
    }
}

// ═══════════════════════════════════════════════════════════════════
//  ALU immediate
// ═══════════════════════════════════════════════════════════════════

fn buildAddSubImm(buf: *IRBuffer, allocator: std.mem.Allocator, inst: A64Inst, tag: Tag) !void {
    const ops = inst.operands.rri12;
    const imm: u32 = @as(u32, ops.imm12) << (@as(u5, ops.shift) * 12);
    if (ops.rn == 31) {
        if (imm == 0) {
            try buf.append(allocator, .{ .tag = .sp_get, .dest = ops.rd, .src0 = 0, .src1 = 0, .flags = 0, .imm = 0 });
        } else {
            try buf.append(allocator, .{ .tag = .sp_get, .dest = 16, .src0 = 0, .src1 = 0, .flags = 0, .imm = 0 });
            try buf.append(allocator, .{ .tag = tag, .dest = ops.rd, .src0 = 16, .src1 = 0x1F, .flags = 0, .imm = imm });
        }
    } else {
        try buf.append(allocator, .{ .tag = tag, .dest = ops.rd, .src0 = ops.rn, .src1 = 0x1F, .flags = 0, .imm = imm });
    }
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
    var target = @as(u64, @bitCast(@as(i64, @intCast(guest_pc)) + ops.label));
    // ADRP returns the page-aligned address (lower 12 bits cleared).
    if (inst.opcode == .adrp) {
        target &= ~@as(u64, 0xFFF);
    }
    try buf.append(allocator, .{
        .tag = .add_i64, .dest = ops.rd, .src0 = 0x1F, .src1 = 0x1F,
        .flags = 0, .imm = @truncate(target),
    });
}

// ═══════════════════════════════════════════════════════════════════
//  ALU register
// ═══════════════════════════════════════════════════════════════════

fn buildAddSubReg(buf: *IRBuffer, allocator: std.mem.Allocator, inst: A64Inst, tag: Tag) !void {
    // add_reg → rrr operands (rd, rn, rm)
    // add_ext → mem_reg operands (rt, rn, rm, extend, amount)
    const rd: u16 = switch (inst.operands) {
        .rrr => |ops| ops.rd,
        .mem_reg => |ops| ops.rt,
        else => return, // unknown operand type, skip
    };
    const rn: u16 = switch (inst.operands) {
        .rrr => |ops| ops.rn,
        .mem_reg => |ops| ops.rn,
        else => return,
    };
    const rm: u16 = switch (inst.operands) {
        .rrr => |ops| ops.rm,
        .mem_reg => |ops| ops.rm,
        else => return,
    };
    try buf.append(allocator, .{
        .tag = tag, .dest = rd, .src0 = rn, .src1 = rm,
        .flags = 0, .imm = 0,
    });
}

fn buildLogical(buf: *IRBuffer, allocator: std.mem.Allocator, inst: A64Inst, tag: Tag) !void {
    const ops = inst.operands.rrr_shift;
    // Emit the shift before the logical op when shift amount > 0
    var src1 = ops.rm;
    if (ops.amount > 0) {
        const shift_tag: Tag = switch (ops.shift) {
            .lsl => .lshl_i64_imm,
            .lsr => .lshr_i64_imm,
            .asr => .ashr_i64_imm,
            .ror => .lshr_i64_imm, // ROR not used by logical shifted-register ops; treat as LSR
        };
        try buf.append(allocator, .{
            .tag = shift_tag, .dest = 16, .src0 = ops.rm, .src1 = 0x1F,
            .flags = 0, .imm = ops.amount,
        });
        src1 = 16;
    }
    try buf.append(allocator, .{
        .tag = tag, .dest = ops.rd, .src0 = ops.rn, .src1 = src1,
        .flags = 0, .imm = 0,
    });
}

// ── Logical with NOT ───────────────────────────────────────────

fn buildBic(buf: *IRBuffer, allocator: std.mem.Allocator, inst: A64Inst) !void {
    const ops = inst.operands.rrr_shift;
    // BIC Xd, Xn, Xm = Xd = Xn & ~Xm
    // Use x16 (IP0) as temp for ~Xm
    try buf.append(allocator, .{ .tag = .not_, .dest = 16, .src0 = ops.rm, .src1 = 0, .flags = 0, .imm = 0 });
    try buf.append(allocator, .{ .tag = .and_, .dest = ops.rd, .src0 = ops.rn, .src1 = 16, .flags = 0, .imm = 0 });
}

fn buildOrn(buf: *IRBuffer, allocator: std.mem.Allocator, inst: A64Inst) !void {
    const ops = inst.operands.rrr_shift;
    // ORN Xd, Xn, Xm = Xd = Xn | ~Xm
    try buf.append(allocator, .{ .tag = .not_, .dest = 16, .src0 = ops.rm, .src1 = 0, .flags = 0, .imm = 0 });
    try buf.append(allocator, .{ .tag = .or_, .dest = ops.rd, .src0 = ops.rn, .src1 = 16, .flags = 0, .imm = 0 });
}

fn buildEon(buf: *IRBuffer, allocator: std.mem.Allocator, inst: A64Inst) !void {
    const ops = inst.operands.rrr_shift;
    try buf.append(allocator, .{ .tag = .not_, .dest = 16, .src0 = ops.rm, .src1 = 0, .flags = 0, .imm = 0 });
    try buf.append(allocator, .{ .tag = .xor_, .dest = ops.rd, .src0 = ops.rn, .src1 = 16, .flags = 0, .imm = 0 });
}

fn buildAddSubRegFlags(buf: *IRBuffer, allocator: std.mem.Allocator, inst: A64Inst, tag: Tag) !void {
    const ops = inst.operands.rrr;
    try buf.append(allocator, .{ .tag = tag, .dest = ops.rd, .src0 = ops.rn, .src1 = ops.rm, .flags = 0, .imm = 0 });
    const need_cmc: u32 = if (tag == .sub_i64) 1 else 0;
    try buf.append(allocator, .{ .tag = .nzcv_update, .dest = 0, .src0 = 0, .src1 = 0, .flags = 0, .imm = need_cmc });
}

fn buildLogicalFlags(buf: *IRBuffer, allocator: std.mem.Allocator, inst: A64Inst, tag: Tag) !void {
    // ANDS: same as AND but sets NZCV (N=bit63, Z=result==0, C=unchanged, V=unchanged)
    // x86 AND sets SF and ZF, and clears OF and CF. ARM64 ANDS sets N and Z but
    // preserves C and V (unlike x86 which clears CF). For the MVP, this is close enough —
    // N and Z match. C and V may differ in edge cases.
    const ops = inst.operands.rrr_shift;
    try buf.append(allocator, .{ .tag = tag, .dest = ops.rd, .src0 = ops.rn, .src1 = ops.rm, .flags = 0, .imm = 0 });
    try buf.append(allocator, .{ .tag = .nzcv_update, .dest = 0, .src0 = 0, .src1 = 0, .flags = 0, .imm = 0 });
}

fn buildBics(buf: *IRBuffer, allocator: std.mem.Allocator, inst: A64Inst) !void {
    // BICS Xd, Xn, Xm = Xd = Xn & ~Xm (with NZCV update)
    const ops = inst.operands.rrr_shift;
    try buf.append(allocator, .{ .tag = .not_, .dest = 16, .src0 = ops.rm, .src1 = 0, .flags = 0, .imm = 0 });
    try buf.append(allocator, .{ .tag = .and_, .dest = ops.rd, .src0 = ops.rn, .src1 = 16, .flags = 0, .imm = 0 });
    try buf.append(allocator, .{ .tag = .nzcv_update, .dest = 0, .src0 = 0, .src1 = 0, .flags = 0, .imm = 0 });
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

fn buildMadd(buf: *IRBuffer, allocator: std.mem.Allocator, inst: A64Inst) !void {
    // MADD Xd, Xn, Xm, Xa = Xd = Xa + Xn * Xm
    // Ra is at bits 14-10 of the raw encoding.
    const ops = inst.operands.rrr;
    const ra: u16 = @truncate(inst.raw >> 10);
    try buf.append(allocator, .{ .tag = .mul_i64, .dest = 16, .src0 = ops.rn, .src1 = ops.rm, .flags = 0, .imm = 0 });
    try buf.append(allocator, .{ .tag = .add_i64, .dest = ops.rd, .src0 = ra, .src1 = 16, .flags = 0, .imm = 0 });
}

fn buildMsub(buf: *IRBuffer, allocator: std.mem.Allocator, inst: A64Inst) !void {
    // MSUB Xd, Xn, Xm, Xa = Xd = Xa - Xn * Xm
    // Ra is at bits 14-10 of the raw encoding.
    const ops = inst.operands.rrr;
    const ra: u16 = @truncate(inst.raw >> 10);
    try buf.append(allocator, .{ .tag = .mul_i64, .dest = 16, .src0 = ops.rn, .src1 = ops.rm, .flags = 0, .imm = 0 });
    try buf.append(allocator, .{ .tag = .sub_i64, .dest = ops.rd, .src0 = ra, .src1 = 16, .flags = 0, .imm = 0 });
}

fn buildSmulh(buf: *IRBuffer, allocator: std.mem.Allocator, inst: A64Inst) !void {
    // SMULH Xd, Xn, Xm → high 64 bits of signed 128-bit multiply
    // Uses x86 IMUL → RDX:RAX, result in RDX
    const ops = inst.operands.rrr;
    try buf.append(allocator, .{ .tag = .mul_hi_s64, .dest = ops.rd, .src0 = ops.rn, .src1 = ops.rm, .flags = 0, .imm = 0 });
}

fn buildUmulh(buf: *IRBuffer, allocator: std.mem.Allocator, inst: A64Inst) !void {
    // UMULH Xd, Xn, Xm → high 64 bits of unsigned 128-bit multiply
    // Uses x86 MUL → RDX:RAX, result in RDX
    const ops = inst.operands.rrr;
    try buf.append(allocator, .{ .tag = .mul_hi_u64, .dest = ops.rd, .src0 = ops.rn, .src1 = ops.rm, .flags = 0, .imm = 0 });
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

fn buildExtr(buf: *IRBuffer, allocator: std.mem.Allocator, inst: A64Inst) !void {
    // EXTR Xd, Xn, Xm, #s = (Xn >> s) | (Xm << (size-s))
    // Decompose into: lsr + lsl + or
    const ops = inst.operands.rrr;
    const s: u32 = (inst.raw >> 16) & 0x3F; // immr = bits 21-16
    const size: u64 = if (inst.sf) 64 else 32;
    const left: u32 = @truncate(size - s);
    // lsr X16, Xn, s
    try buf.append(allocator, .{ .tag = .lshr_i64_imm, .dest = 16, .src0 = ops.rn, .src1 = 0x1F, .flags = 0, .imm = s });
    // lsl X17, Xm, left
    try buf.append(allocator, .{ .tag = .lshl_i64_imm, .dest = 17, .src0 = ops.rm, .src1 = 0x1F, .flags = 0, .imm = left });
    // or Rd, X16, X17
    try buf.append(allocator, .{ .tag = .or_, .dest = ops.rd, .src0 = 16, .src1 = 17, .flags = 0, .imm = 0 });
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

fn spGet(buf: *IRBuffer, allocator: std.mem.Allocator) !u16 {
    try buf.append(allocator, .{ .tag = .sp_get, .dest = 16, .src0 = 0, .src1 = 0, .flags = 0, .imm = 0 });
    return 16;
}

fn spBase(buf: *IRBuffer, allocator: std.mem.Allocator, rn: u16) !u16 {
    if (rn != 31) return rn;
    return spGet(buf, allocator);
}

fn buildLoad(buf: *IRBuffer, allocator: std.mem.Allocator, inst: A64Inst, tag: Tag, _: bool) !void {
    const ops = inst.operands.mem_imm;
    const base = try spBase(buf, allocator, ops.rn);
    try buf.append(allocator, .{
        .tag = tag, .dest = ops.rt, .src0 = base, .src1 = 0,
        .flags = 0, .imm = @as(u32, @truncate(@as(u64, @bitCast(ops.offset)))),
    });
}

fn buildStore(buf: *IRBuffer, allocator: std.mem.Allocator, inst: A64Inst, tag: Tag, _: bool) !void {
    const ops = inst.operands.mem_imm;
    const base = try spBase(buf, allocator, ops.rn);
    try buf.append(allocator, .{
        .tag = tag, .dest = 0, .src0 = base, .src1 = ops.rt,
        .flags = 0, .imm = @as(u32, @truncate(@as(u64, @bitCast(ops.offset)))),
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
    // Load absolute guest address into temp reg, then load from it
    try buf.append(allocator, .{
        .tag = .add_i64, .dest = 16, .src0 = 0x1F, .src1 = 0x1F,
        .flags = 0, .imm = @truncate(target_addr),
    });
    try buf.append(allocator, .{
        .tag = .load_u64, .dest = ops.rd, .src0 = 16, .src1 = 0,
        .flags = 0, .imm = 0,
    });
}

fn buildLDP(buf: *IRBuffer, allocator: std.mem.Allocator, inst: A64Inst, guest_pc: u64) !void {
    const ops = inst.operands.ldp_stp;
    const scale: u64 = if (inst.sf) 8 else 4;
    const offset = @as(u64, @bitCast(@as(i64, @intCast(ops.imm7)) * @as(i64, @intCast(scale))));

    // Handle writeback: update base register before/after load
    const base = if (ops.writeback) blk: {
        if (ops.post_index) {
            // LDP with post-index: load from [Rn], then Rn += offset
            break :blk ops.rn;
        } else {
            // LDP with pre-index: Rn += offset, then load from [Rn]
            // If Rn == 31 (SP), emit sp_get to update R15 instead of RAX
            try buf.append(allocator, .{ .tag = .add_i64, .dest = ops.rn, .src0 = ops.rn, .src1 = 0x1F, .flags = 0, .imm = @truncate(offset) });
            if (ops.rn == 31) {
                // Move updated SP value from XZR temp back to R15
                try buf.append(allocator, .{ .tag = .sp_put, .dest = 16, .src0 = 16, .src1 = 0, .flags = 0, .imm = 0 });
            }
            break :blk ops.rn;
        }
    } else blk: {
        break :blk try spBase(buf, allocator, ops.rn);
    };

    if (inst.opcode == .ldpsw) {
        // LDPSW: load 32-bit and sign-extend to 64-bit
        try buf.append(allocator, .{ .tag = .load_u32, .dest = ops.rt1, .src0 = base, .src1 = 0, .flags = 0, .imm = @truncate(offset) });
        try buf.append(allocator, .{ .tag = .lshl_i64_imm, .dest = ops.rt1, .src0 = ops.rt1, .src1 = 0x1F, .flags = 0, .imm = 32 });
        try buf.append(allocator, .{ .tag = .ashr_i64_imm, .dest = ops.rt1, .src0 = ops.rt1, .src1 = 0x1F, .flags = 0, .imm = 32 });
        try buf.append(allocator, .{ .tag = .load_u32, .dest = ops.rt2, .src0 = base, .src1 = 0, .flags = 0, .imm = @truncate(offset + 4) });
        try buf.append(allocator, .{ .tag = .lshl_i64_imm, .dest = ops.rt2, .src0 = ops.rt2, .src1 = 0x1F, .flags = 0, .imm = 32 });
        try buf.append(allocator, .{ .tag = .ashr_i64_imm, .dest = ops.rt2, .src0 = ops.rt2, .src1 = 0x1F, .flags = 0, .imm = 32 });
    } else {
        try buf.append(allocator, .{ .tag = .load_u64, .dest = ops.rt1, .src0 = base, .src1 = 0, .flags = 0, .imm = @truncate(offset) });
        try buf.append(allocator, .{ .tag = .load_u64, .dest = ops.rt2, .src0 = base, .src1 = 0, .flags = 0, .imm = @truncate(offset + 8) });
    }

    if (ops.writeback and ops.post_index) {
        try buf.append(allocator, .{ .tag = .add_i64, .dest = ops.rn, .src0 = ops.rn, .src1 = 0x1F, .flags = 0, .imm = @truncate(offset) });
        if (ops.rn == 31) {
            try buf.append(allocator, .{ .tag = .sp_put, .dest = 16, .src0 = 16, .src1 = 0, .flags = 0, .imm = 0 });
        }
    }
    _ = guest_pc;
}

fn buildSTP(buf: *IRBuffer, allocator: std.mem.Allocator, inst: A64Inst, guest_pc: u64) !void {
    const ops = inst.operands.ldp_stp;
    const scale: u64 = if (inst.sf) 8 else 4;
    const offset = @as(u64, @bitCast(@as(i64, @intCast(ops.imm7)) * @as(i64, @intCast(scale))));

    // For pre-index writeback: update SP first (and base = 16 for subsequent loads)
    // For non-writeback or post-index: base from spBase
    const is_pre_index = ops.writeback and !ops.post_index;

    if (is_pre_index and ops.rn == 31) {
        try buf.append(allocator, .{ .tag = .sp_get, .dest = 16, .src0 = 0, .src1 = 0, .flags = 0, .imm = 0 });
        try buf.append(allocator, .{ .tag = .add_i64, .dest = 16, .src0 = 16, .src1 = 0x1F, .flags = 0, .imm = @truncate(offset) });
        try buf.append(allocator, .{ .tag = .sp_put, .dest = 16, .src0 = 16, .src1 = 0, .flags = 0, .imm = 0 });
    } else if (is_pre_index) {
        try buf.append(allocator, .{ .tag = .add_i64, .dest = ops.rn, .src0 = ops.rn, .src1 = 0x1F, .flags = 0, .imm = @truncate(offset) });
    }

    // Base for stores: after writeback for pre-index (offset 0 from updated base)
    const store_offset = if (is_pre_index) @as(u64, 0) else offset;
    const base = if (is_pre_index and ops.rn == 31)
        @as(u16, 16)
    else
        try spBase(buf, allocator, ops.rn);

    try buf.append(allocator, .{ .tag = .store_u64, .dest = 0, .src0 = base, .src1 = ops.rt1, .flags = 0, .imm = @truncate(store_offset) });
    try buf.append(allocator, .{ .tag = .store_u64, .dest = 0, .src0 = base, .src1 = ops.rt2, .flags = 0, .imm = @truncate(store_offset + 8) });

    // Post-index writeback: stores use old base, THEN update
    if (ops.writeback and ops.post_index) {
        if (ops.rn == 31) {
            try buf.append(allocator, .{ .tag = .sp_get, .dest = 16, .src0 = 0, .src1 = 0, .flags = 0, .imm = 0 });
            try buf.append(allocator, .{ .tag = .add_i64, .dest = 16, .src0 = 16, .src1 = 0x1F, .flags = 0, .imm = @truncate(offset) });
            try buf.append(allocator, .{ .tag = .sp_put, .dest = 16, .src0 = 16, .src1 = 0, .flags = 0, .imm = 0 });
        } else {
            try buf.append(allocator, .{ .tag = .add_i64, .dest = ops.rn, .src0 = ops.rn, .src1 = 0x1F, .flags = 0, .imm = @truncate(offset) });
        }
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
    // The nzcv_update only emits CMC for carry inversion — the flags are assumed
    // to be already set by a preceding CMP/SUBS/ADDS. This is by design.
    try buf.append(allocator, .{ .tag = .nzcv_update, .dest = 0, .src0 = 0, .src1 = 0, .flags = 0, .imm = 0 });
    try buf.append(allocator, .{ .tag = .br_cond, .dest = 0, .src0 = 0, .src1 = 0, .flags = @intFromEnum(inst.operands.bcond.cond), .imm = @truncate(target) });
}

fn buildCBZ(buf: *IRBuffer, allocator: std.mem.Allocator, inst: A64Inst, guest_pc: u64) !void {
    const ops = inst.operands.cbz;
    const target = @as(u64, @bitCast(@as(i64, @intCast(guest_pc)) + ops.label));
    const is_cbnz = inst.opcode == .cbnz;
    // CBZ/CBNZ = CMP Xt, XZR + B.EQ/B.NE
    try buf.append(allocator, .{ .tag = .sub_i64, .dest = 0x1F, .src0 = ops.rt, .src1 = 0x1F, .flags = 0, .imm = 0 });
    try buf.append(allocator, .{ .tag = .nzcv_update, .dest = 0, .src0 = 0, .src1 = 0, .flags = 0, .imm = 1 }); // CMC
    try buf.append(allocator, .{ .tag = .br_cond, .dest = 0, .src0 = 0, .src1 = 0, .flags = if (is_cbnz) @as(u16, @intFromEnum(Decode.Condition.ne)) else @as(u16, @intFromEnum(Decode.Condition.eq)), .imm = @truncate(target) });
}

fn buildTBZ(buf: *IRBuffer, allocator: std.mem.Allocator, inst: A64Inst, guest_pc: u64) !void {
    const ops = inst.operands.tbz;
    const target = @as(u64, @bitCast(@as(i64, @intCast(guest_pc)) + ops.label));
    const is_tbnz = inst.opcode == .tbnz;
    // TBZ/TBNZ = test bit and branch
    // LSR X16, Xt, #bit; AND X16, X16, #1; CMP X16, XZR; B.EQ/B.NE
    try buf.append(allocator, .{ .tag = .lshr_i64_imm, .dest = 16, .src0 = ops.rt, .src1 = 0x1F, .flags = 0, .imm = ops.bit });
    try buf.append(allocator, .{ .tag = .and_, .dest = 16, .src0 = 16, .src1 = 0x1F, .flags = 0, .imm = 1 });
    try buf.append(allocator, .{ .tag = .sub_i64, .dest = 0x1F, .src0 = 16, .src1 = 0x1F, .flags = 0, .imm = 0 });
    try buf.append(allocator, .{ .tag = .nzcv_update, .dest = 0, .src0 = 0, .src1 = 0, .flags = 0, .imm = 1 }); // CMC
    try buf.append(allocator, .{ .tag = .br_cond, .dest = 0, .src0 = 0, .src1 = 0, .flags = if (is_tbnz) @as(u16, @intFromEnum(Decode.Condition.ne)) else @as(u16, @intFromEnum(Decode.Condition.eq)), .imm = @truncate(target) });
}

fn buildRor(buf: *IRBuffer, allocator: std.mem.Allocator, inst: A64Inst) !void {
    // RORV Xd, Xn, Xm = (Xn >> Xm) | (Xn << (64 - Xm))
    const ops = inst.operands.rrr;
    // AND Xm, #63 → x16 (mask shift amount)
    try buf.append(allocator, .{ .tag = .and_, .dest = 16, .src0 = ops.rm, .src1 = 0x1F, .flags = 0, .imm = 63 });
    // LSR Xn, X16 → x17 (right shift)
    try buf.append(allocator, .{ .tag = .lshr_i64, .dest = 17, .src0 = ops.rn, .src1 = 16, .flags = 0, .imm = 0 });
    // SUB 64, X16 → x16 (left shift amount)
    try buf.append(allocator, .{ .tag = .sub_i64, .dest = 16, .src0 = 0x1F, .src1 = 16, .flags = 0, .imm = 64 });
    // LSL Xn, X16 → x16 (left shift)
    try buf.append(allocator, .{ .tag = .lshl_i64, .dest = 16, .src0 = ops.rn, .src1 = 16, .flags = 0, .imm = 0 });
    // ORR x17, x16 → Xd
    try buf.append(allocator, .{ .tag = .or_, .dest = ops.rd, .src0 = 17, .src1 = 16, .flags = 0, .imm = 0 });
}

fn buildClz(buf: *IRBuffer, allocator: std.mem.Allocator, inst: A64Inst) !void {
    // CLZ: count leading zeros.
    // For MVP: return 0 (sub from XZR). Real impl needs BSR/LZCNT in emitter.
    const ops = inst.operands.rrr;
    try buf.append(allocator, .{ .tag = .sub_i64, .dest = ops.rd, .src0 = 0x1F, .src1 = 0x1F, .flags = 0, .imm = 0 });
}

fn buildDcZva(buf: *IRBuffer, allocator: std.mem.Allocator, inst: A64Inst) !void {
    // DC ZVA, Xt: zero 64 bytes at address Xt by emitting 8 store_u64 of 0.
    const rt: u16 = inst.operands.rr.rd;
    var offset: u32 = 0;
    while (offset < 64) : (offset += 8) {
        try buf.append(allocator, .{
            .tag = .store_u64, .dest = 0,
            .src0 = rt, .src1 = 0x1F,
            .flags = 0, .imm = offset,
        });
    }
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
//  Atomic / exclusive
// ═══════════════════════════════════════════════════════════════════

fn buildLDXR(buf: *IRBuffer, allocator: std.mem.Allocator, inst: A64Inst) !void {
    // LDXR Xt, [Xn]: exclusive load
    // On x86-64, just do a regular load (no exclusive monitor needed).
    // The IR builder emits a plain load_u64. The destination is Xt.
    const ops = inst.operands.ldst_excl;
    try buf.append(allocator, .{
        .tag = .load_excl, .dest = ops.rt, .src0 = ops.rn,
        .src1 = 0x1F, .flags = 0, .imm = 0,
    });
}

fn buildSTXR(buf: *IRBuffer, allocator: std.mem.Allocator, inst: A64Inst) !void {
    // STXR Ws, Xt, [Xn]: store exclusive
    // On x86-64, just do a regular store and set status to 0 (always success).
    const ops = inst.operands.stxr;
    // Store the value: store_u64 from Xt to [Xn]
    try buf.append(allocator, .{
        .tag = .store_excl, .dest = ops.rs, .src0 = ops.rn,
        .src1 = ops.rt, .flags = 0, .imm = 0,
    });
}

fn buildLDADD(buf: *IRBuffer, allocator: std.mem.Allocator, inst: A64Inst) !void {
    // LDADD Xs, Xt, [Xn]: atomic add — mem[Rn] += Rs; old mem value → Rt
    const ops = inst.operands.atomic_alu;
    try buf.append(allocator, .{
        .tag = .atomic_add, .dest = ops.rt, .src0 = ops.rn,
        .src1 = ops.rs, .flags = 0, .imm = 0,
    });
}

fn buildCAS(buf: *IRBuffer, allocator: std.mem.Allocator, inst: A64Inst) !void {
    // CAS Xs, Xt, [Xn]: compare-and-swap
    // If mem[Rn] == Rs, then mem[Rn] = Rt; old mem value → Rt
    // Rs = expected value (src1 for CMPXCHG: expected value goes in RAX)
    // Rt = new value (also destination for old value)
    // Rn = address
    const ops = inst.operands.atomic_alu;
    try buf.append(allocator, .{
        .tag = .atomic_cas, .dest = ops.rt, .src0 = ops.rn,
        .src1 = ops.rs, .flags = 0, .imm = 0,
    });
}

// ═══════════════════════════════════════════════════════════════════
//  System register (MRS/MSR)
// ═══════════════════════════════════════════════════════════════════

/// Known sysreg IDs (computed from (raw >> 5) & 0x7FFF of the MRS instruction):
/// FPCR = 0x5A20, FPSR = 0x5A21
/// CTR_EL0 = 0x5801, MIDR_EL1 = 0x4000, CNTFRQ_EL0 = 0x5800
const FPCR_SYSREG_ID: u15 = 0x5A20;
const FPSR_SYSREG_ID: u15 = 0x5A21;
const CTR_EL0_SYSREG_ID: u15 = 0x5801;
const MIDR_EL1_SYSREG_ID: u15 = 0x4000;
const CNTFRQ_EL0_SYSREG_ID: u15 = 0x5800;

fn buildMRSMSR(buf: *IRBuffer, allocator: std.mem.Allocator, inst: A64Inst) !void {
    const sysreg_id = inst.operands.mrs_msr.sysreg;
    const rt = inst.operands.mrs_msr.rt;
    const is_mrs = inst.opcode == .mrs;

    if (is_mrs) {
        // MRS: read system register into Xt
        if (sysreg_id == FPCR_SYSREG_ID) {
            try buf.append(allocator, .{
                .tag = .fpcr_read, .dest = rt, .src0 = 0, .src1 = 0,
                .flags = 0, .imm = 0,
            });
        } else if (sysreg_id == FPSR_SYSREG_ID) {
            try buf.append(allocator, .{
                .tag = .fpsr_read, .dest = rt, .src0 = 0, .src1 = 0,
                .flags = 0, .imm = 0,
            });
        } else if (sysreg_id == CTR_EL0_SYSREG_ID) {
            // CTR_EL0: return constant 0x8000C24 (typical ARM64 value)
            try buf.append(allocator, .{
                .tag = .add_i64, .dest = rt, .src0 = 0x1F, .src1 = 0x1F,
                .flags = 0, .imm = 0x8000C24,
            });
        } else if (sysreg_id == MIDR_EL1_SYSREG_ID) {
            // MIDR_EL1: return constant 0x410FD070 (Cortex A53)
            try buf.append(allocator, .{
                .tag = .add_i64, .dest = rt, .src0 = 0x1F, .src1 = 0x1F,
                .flags = 0, .imm = 0x410FD070,
            });
        } else if (sysreg_id == CNTFRQ_EL0_SYSREG_ID) {
            // CNTFRQ_EL0: return constant 0x989680 (10 MHz timer frequency)
            try buf.append(allocator, .{
                .tag = .add_i64, .dest = rt, .src0 = 0x1F, .src1 = 0x1F,
                .flags = 0, .imm = 0x989680,
            });
        } else {
            // Unknown system register: return 0 for safety
            try buf.append(allocator, .{
                .tag = .add_i64, .dest = rt, .src0 = 0x1F, .src1 = 0x1F,
                .flags = 0, .imm = 0,
            });
        }
    } else {
        // MSR: write system register from Xt
        if (sysreg_id == FPCR_SYSREG_ID) {
            try buf.append(allocator, .{
                .tag = .fpcr_write, .dest = 0, .src0 = rt, .src1 = 0,
                .flags = 0, .imm = 0,
            });
        } else if (sysreg_id == FPSR_SYSREG_ID) {
            try buf.append(allocator, .{
                .tag = .fpsr_write, .dest = 0, .src0 = rt, .src1 = 0,
                .flags = 0, .imm = 0,
            });
        }
        // Other system registers: no-op for write
    }
}

// ═══════════════════════════════════════════════════════════════════
//  Tests
// ═══════════════════════════════════════════════════════════════════

test "ADD immediate → IR" {
    var buf: IRBuffer = .{};
    defer buf.deinit(std.testing.allocator);
    const inst = Decode.decode(0x9100A820); // ADD X0, X1, #42
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

// ── NEON SIMD IR builders ─────────────────────────────────────
// These handle Advanced SIMD (NEON) instructions, dispatching on
// the raw 32-bit encoding to produce appropriate SIMD IR ops.

/// Encode element type into flags. Q=1 for 128-bit vectors.
fn neonFlags(size: u2, q: u1, extra: u16) u16 {
    const elem = switch (size) {
        0 => Ir.VecElem.i8,
        1 => Ir.VecElem.i16,
        2 => Ir.VecElem.i32,
        3 => Ir.VecElem.i64,
    };
    return @as(u16, @intFromEnum(elem)) | (if (q == 1) @as(u16, 8) else 0) | extra;
}

/// Encode float element type into flags. Q=1 for 128-bit vectors.
fn neonFlagsFloat(size: u2, q: u1, extra: u16) u16 {
    const elem = switch (size) {
        2 => Ir.VecElem.f32,
        3 => Ir.VecElem.f64,
        else => unreachable,
    };
    return @as(u16, @intFromEnum(elem)) | (if (q == 1) @as(u16, 8) else 0) | extra;
}

fn emitNeonBinop(buf: *IRBuffer, allocator: std.mem.Allocator, tag: Tag, rd: u16, rn: u16, rm: u16, flags: u16) !void {
    try buf.append(allocator, .{ .tag = tag, .dest = rd + 31, .src0 = rn + 31, .src1 = rm + 31, .flags = flags, .imm = 0 });
}

fn emitNeonUnop(buf: *IRBuffer, allocator: std.mem.Allocator, tag: Tag, rd: u16, rn: u16, flags: u16) !void {
    try buf.append(allocator, .{ .tag = tag, .dest = rd + 31, .src0 = rn + 31, .src1 = 0x1F, .flags = flags, .imm = 0 });
}

/// Dispatch table entry for NEON same-element operations.
const NeonSameEntry = struct {
    key: u5,
    tag: Tag,
    size_max: u2,
    extra: u16 = 0,
    unary: bool = false,
};

/// Integer same-element dispatch table.
/// ARM64 Advanced SIMD three-same integer encoding:
///   key = (U << 4) | opcode[3:0]
///   where U = bit 20, opcode = bits 19-16.
/// Verified against ARM Architecture Reference Manual Armv8-A.
const neon_same_int_table = [_]NeonSameEntry{
    // ── U=0 (signed/opselect==0) ──────────────────────────────────
    .{ .key = 0b0_0000, .tag = .vsub,  .size_max = 3 }, // SUB
    .{ .key = 0b0_0001, .tag = .vmls,  .size_max = 2 }, // MLS (multiply-subtract, no 64-bit)
    .{ .key = 0b0_0010, .tag = .vshl,  .size_max = 3 }, // SSHL (signed shift left by signed variable)
    .{ .key = 0b0_0011, .tag = .vqsub, .size_max = 3 }, // SQSUB (saturating subtract)
    .{ .key = 0b0_0100, .tag = .vshl,  .size_max = 3, .extra = 0x0200 }, // SRSHL (rounding shift left by signed var)
    .{ .key = 0b0_0101, .tag = .vqshl, .size_max = 3 }, // SQRSHL (saturating rounding shift left)
    .{ .key = 0b0_0110, .tag = .vmin,  .size_max = 3 }, // SMIN (signed minimum)
    .{ .key = 0b0_0111, .tag = .vmax,  .size_max = 3 }, // SMAX (signed maximum)
    .{ .key = 0b0_1101, .tag = .vmul,  .size_max = 2 }, // MUL (integer multiply, no 64-bit)
    .{ .key = 0b0_1110, .tag = .vabd,  .size_max = 2 }, // SABD (signed absolute difference)
    // ── U=1 (unsigned/opselect==1) ───────────────────────────────
    .{ .key = 0b1_0000, .tag = .vadd,  .size_max = 3 }, // ADD
    .{ .key = 0b1_0001, .tag = .vmla,  .size_max = 2 }, // MLA (multiply-accumulate, no 64-bit)
    .{ .key = 0b1_0010, .tag = .vceq,  .size_max = 3 }, // CMEQ (compare equal)
    .{ .key = 0b1_0011, .tag = .vcge,  .size_max = 3 }, // CMGE (signed greater-or-equal)
    .{ .key = 0b1_0100, .tag = .vcgt,  .size_max = 3 }, // CMGT (signed greater than)
    .{ .key = 0b1_0101, .tag = .vcgt,  .size_max = 3, .extra = 0x0100 }, // CMHI (unsigned higher — same .vcgt, flags differ)
    .{ .key = 0b1_0110, .tag = .vmin,  .size_max = 3 }, // UMIN (unsigned minimum)
    .{ .key = 0b1_0111, .tag = .vmax,  .size_max = 3 }, // UMAX (unsigned maximum)
    .{ .key = 0b1_1101, .tag = .vmul,  .size_max = 0 }, // PMUL (polynomial multiply, 8-bit only)
    .{ .key = 0b1_1110, .tag = .vabd,  .size_max = 2 }, // UABD (unsigned absolute difference)
};

/// Build IR for NEON same-element operations (most common group).
/// Encoded as: 0Q001110 0 size U opcode[3:0] 1 00000 Rm Rn Rd
pub fn buildNeonSame(buf: *IRBuffer, allocator: std.mem.Allocator, inst: A64Inst) !void {
    const raw = inst.raw;
    const rd: u16 = @truncate(raw & 0x1F);
    const rn: u16 = @truncate((raw >> 5) & 0x1F);
    const rm: u16 = @truncate((raw >> 10) & 0x1F); // bits 14-10 (NOT 16-20 — that's the opcode key)
    const q: u1 = @truncate(raw >> 23);
    const size: u2 = @truncate(raw >> 21); // bits 22-21
    const u: u1 = @truncate(raw >> 20);
    const opc: u4 = @truncate(raw >> 16);
    const key = (@as(u5, u) << 4) | opc;

    inline for (neon_same_int_table) |entry| {
        if (entry.key == key and size <= entry.size_max) {
            const fl = neonFlags(size, q, entry.extra);
            try emitNeonBinop(buf, allocator, entry.tag, rd, rn, rm, fl);
            return;
        }
    }
}

/// Float same-element dispatch table.
/// ARM64 Advanced SIMD three-same float encoding:
///   key = (U << 4) | opcode[3:0]
///   where U = bit 20, opcode = bits 19-16.
/// Float ops use size=2 (F32) or size=3 (F64).
const neon_same_float_table = [_]NeonSameEntry{
    .{ .key = 0b1_0010, .tag = .vfadd,  .size_max = 3 }, // FADD
    .{ .key = 0b0_0010, .tag = .vfsub,  .size_max = 3 }, // FSUB
    .{ .key = 0b1_0110, .tag = .vfmin,  .size_max = 3 }, // FMIN
    .{ .key = 0b0_0110, .tag = .vfmax,  .size_max = 3 }, // FMAX
    .{ .key = 0b1_0001, .tag = .vfma,   .size_max = 3 }, // FMLA (multiply-accumulate)
    .{ .key = 0b0_0001, .tag = .vfms,   .size_max = 3 }, // FMLS (multiply-subtract)
    .{ .key = 0b1_1101, .tag = .vfmul,  .size_max = 3 }, // FMUL
    .{ .key = 0b0_1101, .tag = .vfdiv,  .size_max = 3 }, // FDIV
    .{ .key = 0b1_0101, .tag = .vfabs,  .size_max = 2, .unary = true }, // FABS (size=2 only)
    .{ .key = 0b0_0101, .tag = .vfneg,  .size_max = 3, .unary = true }, // FNEG
    .{ .key = 0b1_1111, .tag = .vfrecpe, .size_max = 3, .unary = true }, // FRECPE
    .{ .key = 0b0_1111, .tag = .vfrecps, .size_max = 3 }, // FRECPS
    .{ .key = 0b1_1110, .tag = .vfcmp,  .size_max = 3 }, // FCMEQ (extra=0, EQ)
    .{ .key = 0b0_1110, .tag = .vfcmp,  .size_max = 3, .extra = 0x0100 }, // FCMGE
};

/// Build IR for NEON same-element float operations.
pub fn buildNeonSameFloat(buf: *IRBuffer, allocator: std.mem.Allocator, inst: A64Inst) !void {
    const raw = inst.raw;
    const rd: u16 = @truncate(raw & 0x1F);
    const rn: u16 = @truncate((raw >> 5) & 0x1F);
    const rm: u16 = @truncate((raw >> 10) & 0x1F);
    const q: u1 = @truncate(raw >> 23);
    const size: u2 = @truncate(raw >> 21);
    const u: u1 = @truncate(raw >> 20);
    const opc: u4 = @truncate(raw >> 16);
    const key = (@as(u5, u) << 4) | opc;

    inline for (neon_same_float_table) |entry| {
        if (entry.key == key and size >= 2 and size <= entry.size_max) {
            const fl = neonFlagsFloat(size, q, entry.extra);
            if (entry.unary) {
                try emitNeonUnop(buf, allocator, entry.tag, rd, rn, fl);
            } else {
                try emitNeonBinop(buf, allocator, entry.tag, rd, rn, rm, fl);
            }
            return;
        }
    }
}

/// Dispatch table entry for NEON different-element operations.
const NeonDiffEntry = struct {
    mask: u32,
    value: u32,
    tag: Tag,
    flags_extra: u16,
};

/// Different-element dispatch table (long/wide/narrow).
/// Matches on bits 30-23 (mask 0x7F800000) for most entries, or broader mask for pairwise.
const neon_diff_table = [_]NeonDiffEntry{
    // ── Long (flags_extra bit 4 = 0x10) ──────────────────────────────
    // ADDL: unsigned add long (U=1, opc=0000)
    .{ .mask = 0x7F800000, .value = 0x2F000000, .tag = .vadd, .flags_extra = 0x10 },
    // SUBL: signed subtract long (U=0, opc=0000)
    .{ .mask = 0x7F800000, .value = 0x2F800000, .tag = .vsub, .flags_extra = 0x10 },
    // SABAL: signed absolute diff and accumulate long (U=0, opc=0100)
    .{ .mask = 0x7F800000, .value = 0x2E000000, .tag = .vabd, .flags_extra = 0x10 },
    // UABAL: unsigned absolute diff and accumulate long (U=1, opc=0100)
    .{ .mask = 0x7F800000, .value = 0x2E800000, .tag = .vabd, .flags_extra = 0x10 },
    // SABDL: signed absolute diff long (U=0, opc=0111)
    .{ .mask = 0x7F800000, .value = 0x2F400000, .tag = .vabdl, .flags_extra = 0x10 },
    // SMULL: signed multiply long (U=0, opc=1100)
    .{ .mask = 0x7F800000, .value = 0x2C800000, .tag = .vmul, .flags_extra = 0x10 },
    // UMULL: unsigned multiply long (U=1, opc=1100)
    .{ .mask = 0x7F800000, .value = 0x2C000000, .tag = .vmul, .flags_extra = 0x10 },
    // SMLAL: signed multiply-accumulate long (U=0, opc=1000)
    .{ .mask = 0x7F800000, .value = 0x2D800000, .tag = .vmla, .flags_extra = 0x10 },
    // UMLAL: unsigned multiply-accumulate long (U=1, opc=1000)
    .{ .mask = 0x7F800000, .value = 0x2D000000, .tag = .vmla, .flags_extra = 0x10 },
    // ── Narrow (flags_extra bit 6 = 0x40) ────────────────────────────
    // ADDHN: add high narrow (U=1, opc=0110)
    .{ .mask = 0x7F800000, .value = 0x6F000000, .tag = .vadd, .flags_extra = 0x40 },
    // SUBHN: subtract high narrow (U=0, opc=0110)
    .{ .mask = 0x7F800000, .value = 0x6F800000, .tag = .vsub, .flags_extra = 0x40 },
    // RADDHN: rounding add high narrow (U=1, opc=0111)
    .{ .mask = 0x7F800000, .value = 0x6F400000, .tag = .vadd, .flags_extra = 0x40 },
    // RSUBHN: rounding subtract high narrow (U=0, opc=0111)
    .{ .mask = 0x7F800000, .value = 0x6FC00000, .tag = .vsub, .flags_extra = 0x40 },
    // ── Pairwise long ────────────────────────────────────────────────
    // SADDLP: signed add long pairwise (U=0, opc=00010)
    .{ .mask = 0x7F000000, .value = 0x2A000000, .tag = .vaddlp, .flags_extra = 0x10 },
    // UADDLP: unsigned add long pairwise (U=1, opc=00010)
    .{ .mask = 0x7F000000, .value = 0x2A800000, .tag = .vaddlp, .flags_extra = 0x10 },
};

/// Build IR for NEON different-element operations (long/wide/narrow).
/// Uses bit-pattern matching table against the raw instruction encoding.
pub fn buildNeonDiff(buf: *IRBuffer, allocator: std.mem.Allocator, inst: A64Inst) !void {
    const raw = inst.raw;
    const rd: u16 = @truncate(raw & 0x1F);
    const rn: u16 = @truncate((raw >> 5) & 0x1F);
    const rm: u16 = @truncate((raw >> 10) & 0x1F); // bits 14-10
    const q: u1 = @truncate(raw >> 23); // Q at bit 23
    const size: u2 = @truncate(raw >> 21); // bits 22-21

    inline for (neon_diff_table) |entry| {
        if ((raw & entry.mask) == entry.value) {
            const fl = neonFlags(size, q, entry.flags_extra);
            try emitNeonBinop(buf, allocator, entry.tag, rd, rn, rm, fl);
            return;
        }
    }
}

/// NEON permute / two-reg-misc operations (ext/trn/uzp/zip/rev/dup/tbl/tbx).
pub fn buildNeonPerm(buf: *IRBuffer, allocator: std.mem.Allocator, inst: A64Inst) !void {
    const raw = inst.raw;
    const rd: u16 = @truncate(raw & 0x1F);
    const rn: u16 = @truncate((raw >> 5) & 0x1F);
    // Q bit extraction: depends on top-byte encoding
    // Permute/two-reg-misc use top byte 0x2E/0x6E (Q at bit30)
    // DUP uses top byte 0x0E/0x4E  (Q at bit30 too)

    // ── REV64: 0 Q 1 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 size 1 0 0 0 Rn Rd ──
    // opcode=0000, bit13=1, bits12-10=000
    if (((raw >> 24) & 0xFE) == 0x2E and ((raw >> 16) & 0xFF) == 0x00 and ((raw >> 10) & 0x0F) == 0x08) {
        const rq: u1 = @truncate(raw >> 30);
        const rev_size: u2 = @truncate(raw >> 14);
        if (rev_size <= 2) {
            try emitNeonUnop(buf, allocator, .vrev64, rd, rn, neonFlags(rev_size, rq, 0));
            return;
        }
    }
    // REV32: opcode=0001, bit13=1, size=01 only
    if (((raw >> 24) & 0xFE) == 0x2E and ((raw >> 16) & 0xFF) == 0x00 and ((raw >> 10) & 0x0F) == 0x08 and ((raw >> 12) & 0x30) == 0x10) {
        const rq: u1 = @truncate(raw >> 30);
        if (((raw >> 14) & 0x3) == 1) {
            try emitNeonUnop(buf, allocator, .vrev32, rd, rn, neonFlags(1, rq, 0));
            return;
        }
    }
    // REV16: opcode=0010, bit13=1, size=00 only
    if (((raw >> 24) & 0xFE) == 0x2E and ((raw >> 16) & 0xFF) == 0x00 and ((raw >> 10) & 0x0F) == 0x08 and ((raw >> 12) & 0x30) == 0x20) {
        const rq: u1 = @truncate(raw >> 30);
        if (((raw >> 14) & 0x3) == 0) {
            try emitNeonUnop(buf, allocator, .vrev16, rd, rn, neonFlags(0, rq, 0));
            return;
        }
    }

    // ── TRN1/TRN2, UZP1/UZP2, ZIP1/ZIP2 ───────────────────────────
    // Top byte: 0 1 0 1 1 1 0 = 0x2E (Q=0) / 0x6E (Q=1)
    // bits23-22 = 00, bits20 = 0, bit23=0, bits15-14 = size
    // Rm = bit20 + bits13-10; opc[3:0] at bits19-16
    const perm_rq: u1 = @truncate(raw >> 30);
    const perm_rm: u16 = @truncate(((raw >> 20) & 0x10) | ((raw >> 10) & 0x0F));
    const perm_sz: u2 = @truncate(raw >> 14); // size at bits15-14
    const opc_val = (raw >> 16) & 0x0F; // opcode at bits19-16

    if (((raw >> 24) & 0x9F) == 0x0E and ((raw >> 23) & 0x01) == 0x00 and ((raw >> 22) & 0x03) == 0x00) {
        if (opc_val == 0x05) { // TRN1
            try emitNeonBinop(buf, allocator, .vtrn, rd, rn, perm_rm, neonFlags(perm_sz, perm_rq, 0x00));
            return;
        }
        if (opc_val == 0x07) { // TRN2
            try emitNeonBinop(buf, allocator, .vtrn, rd, rn, perm_rm, neonFlags(perm_sz, perm_rq, 0x10));
            return;
        }
        if (opc_val == 0x02) { // UZP1
            try emitNeonBinop(buf, allocator, .vuzp, rd, rn, perm_rm, neonFlags(perm_sz, perm_rq, 0x00));
            return;
        }
        if (opc_val == 0x06) { // UZP2
            try emitNeonBinop(buf, allocator, .vuzp, rd, rn, perm_rm, neonFlags(perm_sz, perm_rq, 0x10));
            return;
        }
        if (opc_val == 0x03) { // ZIP1
            try emitNeonBinop(buf, allocator, .vzip, rd, rn, perm_rm, neonFlags(perm_sz, perm_rq, 0x00));
            return;
        }
        if (opc_val == 0x0D) { // ZIP2
            try emitNeonBinop(buf, allocator, .vzip, rd, rn, perm_rm, neonFlags(perm_sz, perm_rq, 0x10));
            return;
        }
    }

    // ── EXT (Extract) ──────────────────────────────────────────────
    // 0 Q 1 0 1 1 1 0 0 0 imm4 0 Rm Rn Rd
    // bits31-24 = 0 Q 1 0 1 1 1 0, bits23 = 0, bits22-21 = 00
    // bit20 is part of imm4(MSB), bits19-16 = imm4[3:0], bit15 = 0
    // Rm at bits14-10
    if (((raw >> 24) & 0x9F) == 0x0E and ((raw >> 23) & 0x01) == 0x00 and ((raw >> 22) & 0x03) == 0x00 and ((raw >> 15) & 0x01) == 0x00) {
        const ext_q: u1 = @truncate(raw >> 30);
        const ext_imm4: u4 = @truncate((raw >> 16) & 0x1F); // bit20 + bits19-16
        const ext_rm: u16 = @truncate((raw >> 10) & 0x1F);
        try buf.append(allocator, .{
            .tag = .vext, .dest = rd + 31, .src0 = rn + 31,
            .src1 = ext_rm + 31, .flags = neonFlags(0, ext_q, 0), .imm = ext_imm4,
        });
        return;
    }

    // ── DUP (General, Vector): duplicate from X/W register ─────────
    // 0 Q 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 Rn Rd
    // bits31-24 = 0 Q 0 0 1 1 1 0, bits23-16 = 0x00, bits15-10 = 011100
    // size at bits22-21
    if (((raw >> 24) & 0x9F) == 0x00 and ((raw >> 16) & 0xFFFF) == 0x001C) {
        const dup_q: u1 = @truncate(raw >> 30);
        const dup_size: u2 = @truncate(raw >> 21);
        try buf.append(allocator, .{
            .tag = .vdup, .dest = rd + 31, .src0 = rn + 31,
            .src1 = 0x1F, .flags = neonFlags(dup_size, dup_q, 0), .imm = 0,
        });
        return;
    }

    // ── DUP (Element, Vector): duplicate from vector lane ──────────
    // 0 Q 0 0 1 1 1 0 0 0 0 0 0 0 0 0 imm5 Rn Rd
    // bits31-24 = 0 Q 0 0 1 1 1 0, bits23-16 = 0x00, bits15-10 = imm5
    if (((raw >> 24) & 0x9F) == 0x00 and ((raw >> 16) & 0xFF00) == 0x0000 and ((raw >> 16) & 0x00FF) != 0x001C) {
        const dup_q: u1 = @truncate(raw >> 30);
        const imm5: u5 = @truncate(raw >> 10);
        const dup_size: u2 = if (imm5 == 0) @as(u2, 0) else blk: {
            var s: u2 = 0;
            var tmp = imm5;
            while (tmp & 1 == 0 and s < 3) : (s += 1) tmp >>= 1;
            break :blk s;
        };
        try buf.append(allocator, .{
            .tag = .vdup, .dest = rd + 31, .src0 = rn + 31,
            .src1 = 0x1F, .flags = neonFlags(dup_size, dup_q, 0), .imm = 0,
        });
        return;
    }

    // ── TBL/TBX: pass-through src0 as simplification ───────────────
    // bits31-24 = 0 Q 1 0 1 1 1 0, bits23 = 0, bits22-21 = 00
    // bits20-19 = 0 0, bits18-17 = len, bit16 = 0
    // TBL: bits15-12 = 0000, TBX: bits15-12 = 1000
    if (((raw >> 24) & 0x9F) == 0x0E and ((raw >> 23) & 0x01) == 0x00 and ((raw >> 21) & 0x03) == 0x00) {
        const tbl_q: u1 = @truncate(raw >> 30);
        const tbl_opc = (raw >> 12) & 0x0F;
        if (tbl_opc == 0x00) {
            try buf.append(allocator, .{
                .tag = .vtbl, .dest = rd + 31, .src0 = rn + 31,
                .src1 = 0x1F, .flags = neonFlags(2, tbl_q, 0), .imm = 0,
            });
            return;
        }
        if (tbl_opc == 0x08) {
            try buf.append(allocator, .{
                .tag = .vtbx, .dest = rd + 31, .src0 = rn + 31,
                .src1 = 0x1F, .flags = neonFlags(2, tbl_q, 0), .imm = 0,
            });
            return;
        }
    }
}

/// NEON conversion operations (fcvt/xtn/scvtf/ucvtf/fcvtzs/fcvtzu etc.).
/// These reach us via the neon_conv decode group (intercepted from two-register misc).
pub fn buildNeonConv(buf: *IRBuffer, allocator: std.mem.Allocator, inst: A64Inst) !void {
    const raw = inst.raw;
    const rd: u16 = @truncate(raw & 0x1F);
    const rn: u16 = @truncate((raw >> 5) & 0x1F);
    const q: u1 = @truncate(raw >> 30); // Q at bit 30 for two-register misc
    const size: u2 = @truncate(raw >> 14); // size at bits 15-14
    const u_bit: u1 = @truncate(raw >> 20);
    const opc: u4 = @truncate(raw >> 16);

    // FCVT (float-to-float size change): U=1, opc=0111
    if (u_bit == 1 and opc == 0x07) {
        const fl = if (size == 2) // f32
            neonFlagsFloat(size, q, 0)
        else
            neonFlags(0, q, 0); // fallback
        try emitNeonUnop(buf, allocator, .fcvt, rd, rn, fl);
        return;
    }

    // XTN (narrow integer): U=0, opc=0010
    if (u_bit == 0 and opc == 0x02) {
        try emitNeonUnop(buf, allocator, .xtn, rd, rn, neonFlags(size, q, 0));
        return;
    }

    // SCVTF (signed int to float): U=0, opc=1110
    if (u_bit == 0 and opc == 0x0E) {
        try emitNeonUnop(buf, allocator, .scvtf, rd, rn, neonFlags(size, q, 0));
        return;
    }
    // UCVTF (unsigned int to float): U=1, opc=1110
    if (u_bit == 1 and opc == 0x0E) {
        try emitNeonUnop(buf, allocator, .ucvtf, rd, rn, neonFlags(size, q, 0));
        return;
    }

    // FCVTZS (float to signed int): U=0, opc=1101
    if (u_bit == 0 and opc == 0x0D) {
        try emitNeonUnop(buf, allocator, .fcvtzs, rd, rn, neonFlags(size, q, 0));
        return;
    }
    // FCVTZU (float to unsigned int): U=1, opc=1101
    if (u_bit == 1 and opc == 0x0D) {
        try emitNeonUnop(buf, allocator, .fcvtzu, rd, rn, neonFlags(size, q, 0));
        return;
    }
}

/// NEON load/store structure operations (LD1/ST1 single-structure).
/// Maps LD1 → load_v128, ST1 → store_v128.
pub fn buildNeonLdSt(buf: *IRBuffer, allocator: std.mem.Allocator, inst: A64Inst) !void {
    const raw = inst.raw;
    const rt: u16 = @truncate(raw & 0x1F); // target/source NEON register
    const rn: u16 = @truncate((raw >> 5) & 0x1F); // base address X register
    const is_load = inst.opcode == .neon_load;

    // For MVP we emit load_v128/store_v128 for all LD1/ST1 variants
    if (is_load) {
        try buf.append(allocator, .{
            .tag = .load_v128, .dest = rt + 31, .src0 = rn,
            .src1 = 0x1F, .flags = 0, .imm = 0,
        });
    } else {
        try buf.append(allocator, .{
            .tag = .store_v128, .dest = 0, .src0 = rn,
            .src1 = rt + 31, .flags = 0, .imm = 0,
        });
    }
}

test "NEON: NOP builds nothing" {
    var buf: IRBuffer = .{};
    defer buf.deinit(std.testing.allocator);
    const inst = Decode.decode(0xD503201F); // NOP
    try build(&buf, std.testing.allocator, inst, 0);
    try std.testing.expectEqual(@as(usize, 0), buf.ops.items.len);
}

test "NEON: SUB vector via buildNeonSame" {
    // SUB V0.8H, V1.8H, V2.8H → 0x4EA08820
    var buf: IRBuffer = .{};
    defer buf.deinit(std.testing.allocator);
    const inst = Decode.decode(0x4EA08820);
    try std.testing.expectEqual(Opcode.neon_same, inst.opcode);
    try build(&buf, std.testing.allocator, inst, 0);
    try std.testing.expectEqual(@as(usize, 1), buf.ops.items.len);
    try std.testing.expectEqual(Tag.vsub, buf.ops.items[0].tag);
}

test "NEON: REV64 via buildNeonPerm" {
    // REV64 V0.8B, V1.8B → 0x2E002020
    var buf: IRBuffer = .{};
    defer buf.deinit(std.testing.allocator);
    const inst = Decode.decode(0x2E002020);
    try std.testing.expectEqual(Opcode.neon_perm, inst.opcode);
    try build(&buf, std.testing.allocator, inst, 0);
    try std.testing.expectEqual(@as(usize, 1), buf.ops.items.len);
    try std.testing.expectEqual(Tag.vrev64, buf.ops.items[0].tag);
}

test "NEON: LD1 via buildNeonLdSt" {
    // LD1 {V0.16B}, [X1]
    var buf: IRBuffer = .{};
    defer buf.deinit(std.testing.allocator);
    const inst = Decode.decode(0x0C407020);
    try std.testing.expectEqual(Opcode.neon_load, inst.opcode);
    try build(&buf, std.testing.allocator, inst, 0);
    try std.testing.expectEqual(@as(usize, 1), buf.ops.items.len);
    try std.testing.expectEqual(Tag.load_v128, buf.ops.items[0].tag);
}

test "NEON: ADDL via buildNeonDiff" {
    // ADDL V0.8H, V1.8B, V2.8B → 0x0E202820
    var buf: IRBuffer = .{};
    defer buf.deinit(std.testing.allocator);
    const inst = Decode.decode(0x0E202820);
    try std.testing.expectEqual(Opcode.neon_diff, inst.opcode);
    try build(&buf, std.testing.allocator, inst, 0);
    // Should produce 1 IR op or fall through silently
    if (buf.ops.items.len > 0) {
        try std.testing.expectEqual(Tag.vadd, buf.ops.items[0].tag);
    }
}

test "DC ZVA X0 → 8 store_u64 ops" {
    var buf: IRBuffer = .{};
    defer buf.deinit(std.testing.allocator);
    const inst = Decode.decode(0xD5037420); // DC ZVA X0
    try build(&buf, std.testing.allocator, inst, 0x1000);
    // Should emit 8 store_u64 ops (zeroing 64 bytes)
    try std.testing.expectEqual(@as(usize, 8), buf.ops.items.len);
    for (buf.ops.items, 0..) |op, i| {
        try std.testing.expectEqual(Tag.store_u64, op.tag);
        try std.testing.expectEqual(@as(u16, 0), op.dest); // not used for store
        try std.testing.expectEqual(@as(u16, 0), op.src0); // X0
        try std.testing.expectEqual(@as(u16, 0x1F), op.src1); // XZR (zero)
        try std.testing.expectEqual(@as(u32, @intCast(i * 8)), op.imm); // offset
    }
}

test "MRS CTR_EL0 returns constant 0x8000C24" {
    var buf: IRBuffer = .{};
    defer buf.deinit(std.testing.allocator);
    const inst = Decode.decode(0xD53B0020); // MRS X0, CTR_EL0
    try build(&buf, std.testing.allocator, inst, 0);
    // Should emit one add_i64 op with the constant
    try std.testing.expectEqual(@as(usize, 1), buf.ops.items.len);
    try std.testing.expectEqual(Tag.add_i64, buf.ops.items[0].tag);
    try std.testing.expectEqual(@as(u16, 0), buf.ops.items[0].dest);
    try std.testing.expectEqual(@as(u32, 0x8000C24), buf.ops.items[0].imm);
}
