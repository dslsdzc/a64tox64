//! ARM64 (AArch64) instruction decoder.
//!
//! Decodes 32-bit ARM64 instructions into a structured representation
//! using bit-pattern matching. The decoder covers the Phase 1 instruction
//! set: integer ALU, load/store, branches, and system instructions.

const std = @import("std");
const assert = std.debug.assert;

// ── Opcode enumeration ─────────────────────────────────────────────

pub const Opcode = enum(u16) {
    // Data processing — immediate
    add_imm,
    adds_imm,
    sub_imm,
    subs_imm,
    movz,
    movk,
    movn,
    adr,
    adrp,

    // Data processing — register
    add_reg,
    sub_reg,
    add_ext,
    sub_ext,
    mul,
    mneg,
    and_reg,
    bic_reg,
    orr_reg,
    orn_reg,
    eor_reg,
    eon_reg,
    lsl_reg,
    lsr_reg,
    asr_reg,
    ror_reg,
    cmp_reg,
    cmn_reg,
    neg_reg,
    sdiv,
    udiv,

    // Data processing — wide immediate
    ubfm, // unsigned bitfield move (includes LSL, LSR, UXTW)
    sbfm, // signed bitfield move (includes ASR, SXTW)
    bfm,  // bitfield move

    // Loads/stores
    ldr_imm,
    ldr_reg,
    ldr_literal,
    ldrb_imm,
    ldrb_reg,
    ldrh_imm,
    ldrh_reg,
    ldur,
    ldurh,
    ldurb,
    ldp,
    ldpsw,
    str_imm,
    str_reg,
    strb_imm,
    strb_reg,
    strh_imm,
    strh_reg,
    stur,
    stp,

    // Branches
    b,
    bl,
    br,
    blr,
    ret_,

    // Conditional
    b_cond,
    csel,
    csinc,
    csinv,
    csneg,
    ccmp_reg,
    ccmp_imm,

    // System
    svc,
    nop,

    // Unknown / unallocated
    unknown,
};

// ── Condition codes ────────────────────────────────────────────────

pub const Condition = enum(u4) {
    eq = 0b0000,
    ne = 0b0001,
    cs = 0b0010, // hs (same encoding)
    cc = 0b0011, // lo (same encoding)
    mi = 0b0100,
    pl = 0b0101,
    vs = 0b0110,
    vc = 0b0111,
    hi = 0b1000,
    ls = 0b1001,
    ge = 0b1010,
    lt = 0b1011,
    gt = 0b1100,
    le = 0b1101,
    al = 0b1110,
    nv = 0b1111,

    pub fn fromU4(v: u4) Condition {
        return @enumFromInt(v);
    }
};

// ── Shift/extend specifiers ───────────────────────────────────────

pub const ShiftType = enum(u2) {
    lsl = 0b00,
    lsr = 0b01,
    asr = 0b10,
    ror = 0b11,
};

pub const ExtendType = enum(u3) {
    uxtb = 0b000,
    uxth = 0b001,
    uxtw = 0b010,
    uxtx = 0b011,
    sxtb = 0b100,
    sxth = 0b101,
    sxtw = 0b110,
    sxtx = 0b111,
};

// ── Operands union ─────────────────────────────────────────────────

pub const Operands = union(enum) {
    none,
    rd: struct { rd: u5 },
    rn: struct { rn: u5 },
    rr: struct { rd: u5, rn: u5 },
    rrr: struct { rd: u5, rn: u5, rm: u5 },
    rrr_shift: struct { rd: u5, rn: u5, rm: u5, shift: ShiftType, amount: u6 },
    rri12: struct { rd: u5, rn: u5, imm12: u12, shift: u1 },
    ri16: struct { rd: u5, imm16: u16 },
    ri16_hw: struct { rd: u5, imm16: u16, hw: u2 },
    rl: struct { rd: u5, label: i64 },
    mem_imm: struct { rt: u5, rn: u5, offset: i9, size: u2 },
    mem_reg: struct { rt: u5, rn: u5, rm: u5, extend: ExtendType, amount: u3 },
    ldp_stp: struct { rt1: u5, rt2: u5, rn: u5, imm7: i7, load: bool, post_index: bool, writeback: bool },
    b_target: struct { label: i64 },
    br_target: struct { rn: u5 },
    bcond: struct { label: i64, cond: Condition },
    csel: struct { rd: u5, rn: u5, rm: u5, cond: Condition },
    ccmp: struct { rn: u5, rm: u5, cond: Condition, nzcv: u4 },
    svc_op: struct { imm16: u16 },
    bitfield: struct { rd: u5, rn: u5, immr: u6, imms: u6 },
};

// ── Decoded instruction ────────────────────────────────────────────

pub const A64Inst = struct {
    opcode: Opcode,
    operands: Operands,
    raw: u32,
    sf: bool,

    pub fn is64bit(self: A64Inst) bool {
        return self.sf;
    }
};

// ── Main decode function ───────────────────────────────────────────

pub fn decode(raw: u32) A64Inst {
    var inst = A64Inst{
        .opcode = .unknown,
        .operands = .none,
        .raw = raw,
        .sf = (raw >> 31) & 1 == 1,
    };

    inst.opcode = decodeOpcode(raw);
    inst.operands = extractOperands(raw, inst.opcode);
    return inst;
}

// ── Opcode decode dispatch ─────────────────────────────────────────

fn decodeOpcode(raw: u32) Opcode {
    // Try structured groups first, fall back to flat table
    const group_bits = raw >> 24;

    // Unconditional branches (B, BL)
    if (group_bits == 0x14) return .b;
    if (group_bits == 0x94) return .bl;

    // Conditional branches (B.cond)
    if ((raw & 0xFF000010) == 0x54000000) return .b_cond;

    // Exception generation (SVC)
    if ((raw & 0xFF000000) == 0xD4000000) return .svc;

    // NOP (hint encoding)
    if (raw == 0xD503201F) return .nop;

    // Main dispatch by bit pattern table
    inline for (&opcode_table) |entry| {
        if ((raw & entry.mask) == entry.value) {
            return entry.opcode;
        }
    }

    return .unknown;
}

const OpcodeEntry = struct { mask: u32, value: u32, opcode: Opcode };

const opcode_table = [_]OpcodeEntry{
    // ── Data processing — immediate ─────────────────────────────
    // Note: mask includes bit 31 (sf flag) to distinguish 32-bit vs 64-bit
    .{ .mask = 0xFF800000, .value = 0x11000000, .opcode = .add_imm },  // ADD (immediate, 32-bit)
    .{ .mask = 0xFF800000, .value = 0x51000000, .opcode = .sub_imm },  // SUB (immediate, 32-bit)
    .{ .mask = 0xFF800000, .value = 0x91000000, .opcode = .add_imm },  // ADD (immediate, 64-bit)
    .{ .mask = 0xFF800000, .value = 0xD1000000, .opcode = .sub_imm },  // SUB (immediate, 64-bit)
    .{ .mask = 0xFF800000, .value = 0x31000000, .opcode = .adds_imm },  // ADDS (immediate, 32-bit)
    .{ .mask = 0xFF800000, .value = 0x71000000, .opcode = .subs_imm },  // SUBS (immediate, 32-bit)
    .{ .mask = 0xFF800000, .value = 0xB1000000, .opcode = .adds_imm },  // ADDS (immediate, 64-bit)
    .{ .mask = 0xFF800000, .value = 0xF1000000, .opcode = .subs_imm },  // SUBS (immediate, 64-bit)
    .{ .mask = 0xFF800000, .value = 0x12800000, .opcode = .movn },     // MOVN (32-bit)
    .{ .mask = 0xFF800000, .value = 0x92800000, .opcode = .movn },     // MOVN (64-bit)
    .{ .mask = 0xFF800000, .value = 0x52800000, .opcode = .movz },     // MOVZ (32-bit)
    .{ .mask = 0xFF800000, .value = 0xD2800000, .opcode = .movz },     // MOVZ (64-bit)
    .{ .mask = 0xFF800000, .value = 0x72800000, .opcode = .movk },     // MOVK (32-bit)
    .{ .mask = 0xFF800000, .value = 0xF2800000, .opcode = .movk },     // MOVK (64-bit)
    .{ .mask = 0x9F000000, .value = 0x10000000, .opcode = .adr },      // ADR
    .{ .mask = 0x9F000000, .value = 0x90000000, .opcode = .adrp },     // ADRP

    // ── Data processing — register ──────────────────────────────
    .{ .mask = 0x7FE00000, .value = 0x0B000000, .opcode = .add_reg },  // ADD (register, 32-bit)
    .{ .mask = 0x7FE00000, .value = 0x8B000000, .opcode = .add_reg },  // ADD (register, 64-bit)
    .{ .mask = 0x7FE00000, .value = 0x4B000000, .opcode = .sub_reg },  // SUB (register, 32-bit)
    .{ .mask = 0x7FE00000, .value = 0xCB000000, .opcode = .sub_reg },  // SUB (register, 64-bit)
    .{ .mask = 0x7FE00000, .value = 0x2B000000, .opcode = .add_ext },  // ADD (extend, 32-bit)
    .{ .mask = 0x7FE00000, .value = 0xAB000000, .opcode = .add_ext },  // ADD (extend, 64-bit)
    .{ .mask = 0x7FE00000, .value = 0x6B000000, .opcode = .sub_ext },  // SUB (extend, 32-bit)
    .{ .mask = 0x7FE00000, .value = 0xEB000000, .opcode = .sub_ext },  // SUB (extend, 64-bit)
    .{ .mask = 0x7FE00000, .value = 0x0A000000, .opcode = .and_reg },  // AND (register, 32-bit)
    .{ .mask = 0x7FE00000, .value = 0x8A000000, .opcode = .and_reg },  // AND (register, 64-bit)
    .{ .mask = 0x7FE00000, .value = 0x0A200000, .opcode = .bic_reg },  // BIC (32-bit)
    .{ .mask = 0x7FE00000, .value = 0x8A200000, .opcode = .bic_reg },  // BIC (64-bit)
    .{ .mask = 0x7FE00000, .value = 0x2A000000, .opcode = .orr_reg },  // ORR (32-bit)
    .{ .mask = 0x7FE00000, .value = 0xAA000000, .opcode = .orr_reg },  // ORR (64-bit)
    .{ .mask = 0x7FE00000, .value = 0x2A200000, .opcode = .orn_reg },  // ORN (32-bit)
    .{ .mask = 0x7FE00000, .value = 0xAA200000, .opcode = .orn_reg },  // ORN (64-bit)
    .{ .mask = 0x7FE00000, .value = 0x4A000000, .opcode = .eor_reg },  // EOR (32-bit)
    .{ .mask = 0x7FE00000, .value = 0xCA000000, .opcode = .eor_reg },  // EOR (64-bit)
    .{ .mask = 0x7FE00000, .value = 0x4A200000, .opcode = .eon_reg },  // EON (32-bit)
    .{ .mask = 0x7FE00000, .value = 0xCA200000, .opcode = .eon_reg },  // EON (64-bit)
    .{ .mask = 0x7FE00000, .value = 0x1B000000, .opcode = .mul },      // MUL (32-bit)
    .{ .mask = 0x7FE00000, .value = 0x9B000000, .opcode = .mul },      // MUL (64-bit)
    .{ .mask = 0x7FE00000, .value = 0x1B000800, .opcode = .mneg },     // MNEG (32-bit)
    .{ .mask = 0x7FE00000, .value = 0x9B000800, .opcode = .mneg },     // MNEG (64-bit)
    .{ .mask = 0x7FE00000, .value = 0x1AC00C00, .opcode = .sdiv },     // SDIV (32-bit)
    .{ .mask = 0x7FE00000, .value = 0x9AC00C00, .opcode = .sdiv },     // SDIV (64-bit)
    .{ .mask = 0x7FE00000, .value = 0x1AC00800, .opcode = .udiv },     // UDIV (32-bit)
    .{ .mask = 0x7FE00000, .value = 0x9AC00800, .opcode = .udiv },     // UDIV (64-bit)
    .{ .mask = 0x7FE00000, .value = 0x6B000000, .opcode = .cmp_reg },  // CMP (32-bit, SUBS XZR)
    .{ .mask = 0x7FE00000, .value = 0xEB000000, .opcode = .cmp_reg },  // CMP (64-bit, SUBS XZR)

    // ── Shift by register ──────────────────────────────────────
    .{ .mask = 0x7FE00000, .value = 0x1AC02000, .opcode = .lsl_reg },  // LSLV (32-bit)
    .{ .mask = 0x7FE00000, .value = 0x9AC02000, .opcode = .lsl_reg },  // LSLV (64-bit)
    .{ .mask = 0x7FE00000, .value = 0x1AC02400, .opcode = .lsr_reg },  // LSRV (32-bit)
    .{ .mask = 0x7FE00000, .value = 0x9AC02400, .opcode = .lsr_reg },  // LSRV (64-bit)
    .{ .mask = 0x7FE00000, .value = 0x1AC02800, .opcode = .asr_reg },  // ASRV (32-bit)
    .{ .mask = 0x7FE00000, .value = 0x9AC02800, .opcode = .asr_reg },  // ASRV (64-bit)
    .{ .mask = 0x7FE00000, .value = 0x1AC02C00, .opcode = .ror_reg },  // RORV (32-bit)
    .{ .mask = 0x7FE00000, .value = 0x9AC02C00, .opcode = .ror_reg },  // RORV (64-bit)

    // ── Bitfield ───────────────────────────────────────────────
    .{ .mask = 0xFF800000, .value = 0x13000000, .opcode = .ubfm },     // UBFM (32-bit)
    .{ .mask = 0xFF800000, .value = 0x93400000, .opcode = .ubfm },     // UBFM (64-bit)
    .{ .mask = 0xFF800000, .value = 0x13000000, .opcode = .sbfm },     // SBFM (32-bit)
    .{ .mask = 0xFF800000, .value = 0x93400000, .opcode = .sbfm },     // SBFM (64-bit)

    // ── Conditional select ──────────────────────────────────────
    .{ .mask = 0x7FE00000, .value = 0x1A800000, .opcode = .csel },     // CSEL (32-bit)
    .{ .mask = 0x7FE00000, .value = 0x9A800000, .opcode = .csel },     // CSEL (64-bit)
    .{ .mask = 0x7FE00000, .value = 0x1A840000, .opcode = .csinc },    // CSINC (32-bit)
    .{ .mask = 0x7FE00000, .value = 0x9A840000, .opcode = .csinc },    // CSINC (64-bit)
    .{ .mask = 0x7FE00000, .value = 0x5A800000, .opcode = .csinv },    // CSINV (32-bit)
    .{ .mask = 0x7FE00000, .value = 0xDA800000, .opcode = .csinv },    // CSINV (64-bit)
    .{ .mask = 0x7FE00000, .value = 0x5A840000, .opcode = .csneg },    // CSNEG (32-bit)
    .{ .mask = 0x7FE00000, .value = 0xDA840000, .opcode = .csneg },    // CSNEG (64-bit)

    // ── Conditional compare ─────────────────────────────────────
    .{ .mask = 0x7FE00000, .value = 0x5A400000, .opcode = .ccmp_reg }, // CCMP (32-bit, register)
    .{ .mask = 0x7FE00000, .value = 0xDA400000, .opcode = .ccmp_reg }, // CCMP (64-bit, register)
    .{ .mask = 0x7FE00000, .value = 0x5A400800, .opcode = .ccmp_imm }, // CCMP (32-bit, immediate)
    .{ .mask = 0x7FE00000, .value = 0xDA400800, .opcode = .ccmp_imm }, // CCMP (64-bit, immediate)

    // ── Loads/stores ────────────────────────────────────────────
    // LDR/STR (immediate, unsigned offset)
    .{ .mask = 0xFFC00000, .value = 0xB9000000, .opcode = .str_imm },  // STR (32-bit)
    .{ .mask = 0xFFC00000, .value = 0xF9000000, .opcode = .str_imm },  // STR (64-bit)
    .{ .mask = 0xFFC00000, .value = 0xB9400000, .opcode = .ldr_imm },  // LDR (32-bit)
    .{ .mask = 0xFFC00000, .value = 0xF9400000, .opcode = .ldr_imm },  // LDR (64-bit)
    // LDR/STR (immediate, scaled) — 8-bit
    .{ .mask = 0xFFC00000, .value = 0x39000000, .opcode = .strb_imm }, // STRB
    .{ .mask = 0xFFC00000, .value = 0x39400000, .opcode = .ldrb_imm }, // LDRB
    // LDR/STR (immediate, scaled) — 16-bit
    .{ .mask = 0xFFC00000, .value = 0x79000000, .opcode = .strh_imm }, // STRH
    .{ .mask = 0xFFC00000, .value = 0x79400000, .opcode = .ldrh_imm }, // LDRH
    // LDR (literal)
    .{ .mask = 0xFF000000, .value = 0x18000000, .opcode = .ldr_literal },
    // LDRSW (literal)
    .{ .mask = 0xFF000000, .value = 0x98000000, .opcode = .ldr_literal },
    // PRFM (literal) — skip
    // LDP/STP
    .{ .mask = 0x7FC00000, .value = 0x29400000, .opcode = .ldp },      // LDP (32-bit, signed offset)
    .{ .mask = 0x7FC00000, .value = 0xA9400000, .opcode = .ldp },      // LDP (64-bit, signed offset)
    .{ .mask = 0x7FC00000, .value = 0x29800000, .opcode = .stp },      // STP (32-bit, signed offset)
    .{ .mask = 0x7FC00000, .value = 0xA9800000, .opcode = .stp },      // STP (64-bit, signed offset)
    .{ .mask = 0x7FC00000, .value = 0x28400000, .opcode = .ldp },      // LDP (32-bit, pre-index)
    .{ .mask = 0x7FC00000, .value = 0xA8400000, .opcode = .ldp },      // LDP (64-bit, pre-index)
    .{ .mask = 0x7FC00000, .value = 0x28800000, .opcode = .stp },      // STP (32-bit, pre-index)
    .{ .mask = 0x7FC00000, .value = 0xA8800000, .opcode = .stp },      // STP (64-bit, pre-index)
    .{ .mask = 0x7FC00000, .value = 0x28400000, .opcode = .ldp },      // LDP (32-bit, post-index)
    .{ .mask = 0x7FC00000, .value = 0xA8400000, .opcode = .ldp },      // LDP (64-bit, post-index)
    // LDUR/STUR
    .{ .mask = 0x3B200000, .value = 0x38400000, .opcode = .ldurb },    // LDURB
    .{ .mask = 0x3B200000, .value = 0x78400000, .opcode = .ldurh },    // LDURH
    .{ .mask = 0x3B200000, .value = 0xB8400000, .opcode = .ldur },     // LDUR (32-bit)
    .{ .mask = 0x3B200000, .value = 0xF8400000, .opcode = .ldur },     // LDUR (64-bit)

    // ── Branches (register) ─────────────────────────────────────
    .{ .mask = 0xFFFFFC1F, .value = 0xD61F0000, .opcode = .br },
    .{ .mask = 0xFFFFFC1F, .value = 0xD63F0000, .opcode = .blr },
    .{ .mask = 0xFFFFFC1F, .value = 0xD65F0000, .opcode = .ret_ },
};

// ── Operand extraction ────────────────────────────────────────────

fn extractOperands(raw: u32, opcode: Opcode) Operands {
    return switch (opcode) {
        .add_imm, .sub_imm => extractRRI12(raw),
        .movz, .movk, .movn => extractMovImm(raw),
        .adr, .adrp => extractADR(raw),
        .add_reg, .sub_reg => extractRRR(raw),
        .add_ext, .sub_ext => extractExtend(raw),
        .and_reg, .bic_reg, .orr_reg, .orn_reg, .eor_reg, .eon_reg => extractRRRShift(raw),
        .mul, .mneg, .sdiv, .udiv => extractRRR(raw),
        .lsl_reg, .lsr_reg, .asr_reg, .ror_reg => extractRRR(raw),
        .cmp_reg, .cmn_reg, .neg_reg => extractCmp(raw),
        .ubfm, .sbfm, .bfm => extractBitfield(raw),
        .csel, .csinc, .csinv, .csneg => extractCSel(raw),
        .ccmp_reg => extractCCmpReg(raw),
        .ccmp_imm => extractCCmpImm(raw),
        .ldr_imm, .str_imm, .ldrb_imm, .strb_imm, .ldrh_imm, .strh_imm => extractMemImm(raw),
        .ldr_literal => extractLiteral(raw),
        .ldp, .stp => extractLDP_STP(raw),
        .ldur, .ldurh, .ldurb, .stur => extractLDUR(raw),
        .b, .bl => extractBranch(raw),
        .br, .blr, .ret_ => extractBranchReg(raw),
        .b_cond => extractBCond(raw),
        .svc => extractSVC(raw),
        .nop => Operands{ .none = {} },
        else => Operands{ .none = {} },
    };
}

// ── Operand extraction helpers ─────────────────────────────────────

fn extractRRI12(raw: u32) Operands {
    const rd: u5 = @truncate(raw);
    const rn: u5 = @truncate(raw >> 5);
    const imm12: u12 = @truncate(raw >> 10);
    const shift: u1 = @truncate(raw >> 22);
    return .{ .rri12 = .{ .rd = rd, .rn = rn, .imm12 = imm12, .shift = shift } };
}

fn extractMovImm(raw: u32) Operands {
    const rd: u5 = @truncate(raw);
    const imm16: u16 = @truncate(raw >> 5);
    const hw: u2 = @truncate(raw >> 21);
    return .{ .ri16_hw = .{ .rd = rd, .imm16 = imm16, .hw = hw } };
}

fn extractADR(raw: u32) Operands {
    const rd: u5 = @truncate(raw);
    const imm_low: u21 = @truncate((raw >> 3) & 0x1FFFFF);
    const imm_high: u2 = @truncate(raw >> 29);
    // Reconstruct signed 21-bit label
    const imm: u23 = @as(u23, @intCast(imm_high)) << 21 | imm_low;
    const label: i64 = @as(i64, @as(i23, @bitCast(imm)));
    return .{ .rl = .{ .rd = rd, .label = label } };
}

fn extractRRR(raw: u32) Operands {
    const rd: u5 = @truncate(raw);
    const rn: u5 = @truncate(raw >> 5);
    const rm: u5 = @truncate(raw >> 16);
    return .{ .rrr = .{ .rd = rd, .rn = rn, .rm = rm } };
}

fn extractRRRShift(raw: u32) Operands {
    const rd: u5 = @truncate(raw);
    const rn: u5 = @truncate(raw >> 5);
    const rm: u5 = @truncate(raw >> 16);
    const shift_type: ShiftType = @enumFromInt(@as(u2, @truncate(raw >> 22)));
    const amount: u6 = @truncate(raw >> 10);
    return .{ .rrr_shift = .{ .rd = rd, .rn = rn, .rm = rm, .shift = shift_type, .amount = amount } };
}

fn extractExtend(raw: u32) Operands {
    const rd: u5 = @truncate(raw);
    const rn: u5 = @truncate(raw >> 5);
    const rm: u5 = @truncate(raw >> 16);
    const extend_type: ExtendType = @enumFromInt(@as(u3, @truncate(raw >> 13)));
    const amount: u3 = @truncate(raw >> 10);
    return .{ .mem_reg = .{ .rt = rd, .rn = rn, .rm = rm, .extend = extend_type, .amount = amount } };
}

fn extractCmp(raw: u32) Operands {
    const rn: u5 = @truncate(raw >> 5);
    return .{ .rr = .{ .rd = 0x1F, .rn = rn } }; // XZR destination
}

fn extractBitfield(raw: u32) Operands {
    const rd: u5 = @truncate(raw);
    const rn: u5 = @truncate(raw >> 5);
    const immr: u6 = @truncate(raw >> 16);
    const imms: u6 = @truncate(raw >> 10);
    return .{ .bitfield = .{ .rd = rd, .rn = rn, .immr = immr, .imms = imms } };
}

fn extractCSel(raw: u32) Operands {
    const rd: u5 = @truncate(raw);
    const rn: u5 = @truncate(raw >> 5);
    const rm: u5 = @truncate(raw >> 16);
    const cond: Condition = @enumFromInt(@as(u4, @truncate(raw >> 12)));
    return .{ .csel = .{ .rd = rd, .rn = rn, .rm = rm, .cond = cond } };
}

fn extractCCmpReg(raw: u32) Operands {
    const rn: u5 = @truncate(raw >> 5);
    const rm: u5 = @truncate(raw >> 16);
    const cond: Condition = @enumFromInt(@as(u4, @truncate(raw >> 4)));
    const nzcv: u4 = @truncate(raw);
    return .{ .ccmp = .{ .rn = rn, .rm = rm, .cond = cond, .nzcv = nzcv } };
}

fn extractCCmpImm(raw: u32) Operands {
    const rn: u5 = @truncate(raw >> 5);
    const imm5: u5 = @truncate(raw >> 16);
    const cond: Condition = @enumFromInt(@as(u4, @truncate(raw >> 4)));
    const nzcv: u4 = @truncate(raw);
    return .{ .ccmp = .{ .rn = rn, .rm = @truncate(imm5), .cond = cond, .nzcv = nzcv } };
}

fn extractMemImm(raw: u32) Operands {
    const rt: u5 = @truncate(raw);
    const rn: u5 = @truncate(raw >> 5);
    const imm12: u12 = @truncate(raw >> 10);
    const size: u2 = @truncate(raw >> 30);
    const scale: u9 = @as(u9, 1) << @intCast(size);
    const offset: i9 = @as(i9, @intCast(imm12 * scale));
    return .{ .mem_imm = .{ .rt = rt, .rn = rn, .offset = offset, .size = size } };
}

fn extractLiteral(raw: u32) Operands {
    const rt: u5 = @truncate(raw);
    const imm19: i64 = @as(i64, @as(i21, @bitCast(@as(u21, @intCast((raw >> 3) & 0x1FFFFF)))));
    const label: i64 = imm19 * 4; // word-aligned offset
    return .{ .rl = .{ .rd = rt, .label = label } };
}

fn extractLDP_STP(raw: u32) Operands {
    const rt1: u5 = @truncate(raw);
    const rn: u5 = @truncate(raw >> 5);
    const rt2: u5 = @truncate(raw >> 10);
    const imm7: i7 = @bitCast(@as(u7, @truncate(raw >> 15)));
    const load: bool = ((raw >> 22) & 1) == 0;
    // Bit 23 indicates pre/post indexing
    const writeback: bool = ((raw >> 23) & 1) == 1;
    const post_index: bool = writeback and ((raw >> 24) & 1) == 0;
    return .{ .ldp_stp = .{
        .rt1 = rt1,
        .rt2 = rt2,
        .rn = rn,
        .imm7 = imm7,
        .load = load,
        .post_index = post_index,
        .writeback = writeback,
    } };
}

fn extractLDUR(raw: u32) Operands {
    const rt: u5 = @truncate(raw);
    const rn: u5 = @truncate(raw >> 5);
    const imm9: i9 = @bitCast(@as(u9, @truncate(raw >> 12)));
    return .{ .mem_imm = .{ .rt = rt, .rn = rn, .offset = imm9, .size = 3 } };
}

fn extractBranch(raw: u32) Operands {
    const imm26: u26 = @truncate(raw & 0x03FFFFFF);
    const offset: i64 = @as(i64, @as(i28, @bitCast(@as(u28, @intCast(imm26 << 2)))));
    return .{ .b_target = .{ .label = offset } };
}

fn extractBranchReg(raw: u32) Operands {
    const rn: u5 = @truncate(raw >> 5);
    return .{ .br_target = .{ .rn = rn } };
}

fn extractBCond(raw: u32) Operands {
    const imm19: u19 = @truncate((raw >> 5) & 0x7FFFF);
    const cond: u4 = @truncate(raw & 0xF);
    const offset: i64 = @as(i64, @as(i21, @bitCast(@as(u21, @intCast(imm19 << 2)))));
    return .{ .bcond = .{ .label = offset, .cond = @enumFromInt(cond) } };
}

fn extractSVC(raw: u32) Operands {
    const imm16: u16 = @truncate((raw >> 5) & 0xFFFF);
    return .{ .svc_op = .{ .imm16 = imm16 } };
}

// ── Tests ─────────────────────────────────────────────────────────

test "decode ADD immediate (32-bit)" {
    // ADD W0, W1, #42 → 0x11000C2A
    const inst = decode(0x11000C2A);
    try std.testing.expectEqual(Opcode.add_imm, inst.opcode);
    try std.testing.expectEqual(false, inst.sf);
    try std.testing.expectEqual(@as(u5, 0), inst.operands.rri12.rd);
    try std.testing.expectEqual(@as(u5, 1), inst.operands.rri12.rn);
    try std.testing.expectEqual(@as(u12, 42), inst.operands.rri12.imm12);
}

test "decode ADD immediate (64-bit)" {
    // ADD X0, X1, #42 → 0x91000C2A
    const inst = decode(0x91000C2A);
    try std.testing.expectEqual(Opcode.add_imm, inst.opcode);
    try std.testing.expectEqual(true, inst.sf);
    try std.testing.expectEqual(@as(u5, 0), inst.operands.rri12.rd);
    try std.testing.expectEqual(@as(u5, 1), inst.operands.rri12.rn);
    try std.testing.expectEqual(@as(u12, 42), inst.operands.rri12.imm12);
}

test "decode SUB immediate" {
    // SUB X2, X3, #0xFF → 0xD1007C62
    const inst = decode(0xD1007C62);
    try std.testing.expectEqual(Opcode.sub_imm, inst.opcode);
    try std.testing.expectEqual(@as(u5, 2), inst.operands.rri12.rd);
    try std.testing.expectEqual(@as(u5, 3), inst.operands.rri12.rn);
    try std.testing.expectEqual(@as(u12, 0xFF), inst.operands.rri12.imm12);
}

test "decode MOVZ (64-bit)" {
    // MOVZ X0, #0x42 → 0xD2800080
    // Encoding: sf=1, opc=10, hw=00, imm16=0x0042, rd=0
    const inst = decode(0xD2800080);
    try std.testing.expectEqual(Opcode.movz, inst.opcode);
    try std.testing.expectEqual(@as(u5, 0), inst.operands.ri16_hw.rd);
    try std.testing.expectEqual(@as(u16, 0x0042), inst.operands.ri16_hw.imm16);
}

test "decode unconditional branch B" {
    // B #256 → 0x14000040 (offset = 256/4 = 64 = 0x40)
    const inst = decode(0x14000040);
    try std.testing.expectEqual(Opcode.b, inst.opcode);
    try std.testing.expectEqual(@as(i64, 256), inst.operands.b_target.label);
}

test "decode BL" {
    // BL #0x1000 → offset = 0x1000/4 = 1024 = 0x400
    const inst = decode(0x94000400);
    try std.testing.expectEqual(Opcode.bl, inst.opcode);
    try std.testing.expectEqual(@as(i64, 0x1000), inst.operands.b_target.label);
}

test "decode BLR X0" {
    const inst = decode(0xD63F0000);
    try std.testing.expectEqual(Opcode.blr, inst.opcode);
    try std.testing.expectEqual(@as(u5, 0), inst.operands.br_target.rn);
}

test "decode RET" {
    const inst = decode(0xD65F0000);
    try std.testing.expectEqual(Opcode.ret_, inst.opcode);
}

test "decode NOP" {
    const inst = decode(0xD503201F);
    try std.testing.expectEqual(Opcode.nop, inst.opcode);
}

test "decode B.EQ" {
    // B.EQ #32 → offset = 32/4 = 8, cond=EQ(0)
    const inst = decode(0x54000100);
    try std.testing.expectEqual(Opcode.b_cond, inst.opcode);
    try std.testing.expectEqual(@as(i64, 32), inst.operands.bcond.label);
    try std.testing.expectEqual(Condition.eq, inst.operands.bcond.cond);
}

test "decode CSEL X0, X1, X2, EQ" {
    // CSEL X0, X1, X2, EQ → 0x9A820020
    const inst = decode(0x9A820020);
    try std.testing.expectEqual(Opcode.csel, inst.opcode);
    try std.testing.expectEqual(@as(u5, 0), inst.operands.csel.rd);
    try std.testing.expectEqual(@as(u5, 1), inst.operands.csel.rn);
    try std.testing.expectEqual(@as(u5, 2), inst.operands.csel.rm);
    try std.testing.expectEqual(Condition.eq, inst.operands.csel.cond);
}

test "decode MUL X0, X1, X2" {
    // MUL X0, X1, X2 → 0x9B007C20
    const inst = decode(0x9B007C20);
    try std.testing.expectEqual(Opcode.mul, inst.opcode);
    try std.testing.expectEqual(@as(u5, 0), inst.operands.rrr.rd);
    try std.testing.expectEqual(@as(u5, 1), inst.operands.rrr.rn);
    try std.testing.expectEqual(@as(u5, 2), inst.operands.rrr.rm);
}

test "decode SVC #0" {
    const inst = decode(0xD4000001);
    try std.testing.expectEqual(Opcode.svc, inst.opcode);
}

test "decode unknown instruction" {
    // An unallocated encoding
    const inst = decode(0x00000000);
    try std.testing.expectEqual(Opcode.unknown, inst.opcode);
}

test "decode LDR X0, [X1, #16]" {
    // LDR X0, [X1, #16] → scaled offset: imm12=2 (because 16/8=2), size=3 (64-bit)
    // 0xF9400020
    const inst = decode(0xF9400020);
    try std.testing.expectEqual(Opcode.ldr_imm, inst.opcode);
    try std.testing.expectEqual(@as(u5, 0), inst.operands.mem_imm.rt);
    try std.testing.expectEqual(@as(u5, 1), inst.operands.mem_imm.rn);
}

test "decode STP X0, X1, [SP]" {
    // STP X0, X1, [SP] → signed offset, imm7=0
    // 0xA98007E0
    const inst = decode(0xA98007E0);
    try std.testing.expectEqual(Opcode.stp, inst.opcode);
    try std.testing.expectEqual(@as(u5, 0), inst.operands.ldp_stp.rt1);
    try std.testing.expectEqual(@as(u5, 1), inst.operands.ldp_stp.rt2);
}
