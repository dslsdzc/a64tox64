//! a64tox64 Intermediate Representation.
//! Fixed-size ops with predictable layout.
//! Each op encodes a single machine-level operation in an
//! architecture-neutral form. ARM64 frontend produces these;
//! x86-64 backend consumes them.

const std = @import("std");
const assert = std.debug.assert;

pub const Tag = enum(u16) {
    // Note: _i32 variants are defined for completeness but ir_builder currently
    // emits only _i64 variants. The 32-bit tags exist for future use (e.g., when
    // the decoder extracts the sf=0 encoding for explicit 32-bit operations).
    // ── ALU integer ─────────────────────────────────────────────────
    add_i32,
    add_i64,
    sub_i32,
    sub_i64,
    mul_i32,
    mul_i64,
    and_,
    or_,
    xor_,
    lshl_i32,
    lshl_i64,
    lshr_i32,
    lshr_i64,
    ashr_i32,
    ashr_i64,

    // ── Condition flags ──────────────────────────────────────────────
    nzcv_read,
    nzcv_update,

    // ── Memory ───────────────────────────────────────────────────────
    load_u8,
    load_u16,
    load_u32,
    load_u64,
    load_v128,
    store_u8,
    store_u16,
    store_u32,
    store_u64,
    store_v128,

    // ── ALU extra ───────────────────────────────────────────────────
    not_,
    neg_i64,
    div_u64,
    div_s64,
    mul_hi_s64,
    mul_hi_u64,
    mov_i64, // reg-reg copy
    adc_i64, // add with carry
    sbc_i64, // subtract with carry
    lshl_i64_imm,
    lshr_i64_imm,
    ashr_i64_imm,

    // ── Control flow ─────────────────────────────────────────────────
    br,
    br_cond,
    call,
    call_reg,
    ret_,
    ccmp,

    // ── SIMD / FP ────────────────────────────────────────────────────
    // Integer vector arithmetic
    vadd,    // with long/wide/narrow/saturating variant in flags
    vsub,
    vmul,
    vmla,    // multiply-accumulate
    vmls,    // multiply-subtract
    vabs,    // absolute value
    vneg,
    vmin,
    vmax,
    vabd,    // absolute difference
    vpadd,   // pairwise add
    vpmin,
    vpmax,
    vqadd,   // saturating add
    vqsub,   // saturating sub
    vaddlp,  // pairwise add long
    vaddlv,  // pairwise add long across vector
    vabdl,   // absolute difference long

    // Float vector arithmetic
    vfadd,
    vfsub,
    vfmul,
    vfdiv,
    vfma,    // fused multiply-add
    vfms,    // fused multiply-sub
    vfmin,
    vfmax,
    vfabs,
    vfneg,
    vfrecpe, // reciprocal estimate
    vfrecps, // reciprocal step
    vfsqrt,
    vfmulx,  // multiply extended

    // Integer compare
    vceq,
    vcgt,
    vcge,

    // Float compare
    vfcmp,

    // Vector shift
    vshl,    // with saturating/rounding variant in flags
    vshr,
    vsra,    // shift right and accumulate
    vsri,    // shift right and insert
    vsli,    // shift left and insert
    vqshl,   // saturating shift left
    vqshlu,  // saturating shift left unsigned (for signed->unsigned)
    vrshr,   // rounding shift right
    vshrn,   // shift right and narrow
    vrshrn,  // rounding shift right narrow

    // Conversion
    fcvt,    // float-to-float size change
    fcvtzs,  // float to signed int
    fcvtzu,  // float to unsigned int
    scvtf,   // signed int to float
    ucvtf,   // unsigned int to float
    fcvtl,   // float long (f16→f32, f32→f64)
    fcvtn,   // float narrow (f32→f16, f64→f32)
    xtn,     // narrow integer (i16→i8, i32→i16, i64→i32)
    uxtl,    // unsigned extend long
    sxtl,    // signed extend long

    // Permute / data manipulation
    vext,    // extract vector from pair
    vtrn,    // transpose
    vuzp,    // de-interleave
    vzip,    // interleave
    vrev16,  // reverse within 16-bit elements
    vrev32,  // reverse within 32-bit elements
    vrev64,  // reverse within 64-bit elements
    vdup,    // broadcast scalar to all lanes
    vtbl,    // table lookup (1-4 registers)
    vtbx,    // table lookup with background

    // Logical
    vand,
    vorr,
    veor,
    vbic,    // bitwise clear
    vorn,    // bitwise nor
    vbsl,    // bitwise select

    // Load / store extra
    load_v64,   // 64-bit vector load (D-register)
    store_v64,  // 64-bit vector store

    // ── Atomic / exclusive ───────────────────────────────────────────
    load_excl,   // load with exclusive monitor (on x86 just a regular load)
    store_excl,  // store exclusive; dst=status (0=success, always success on x86)
    atomic_add,  // atomic add: mem[src0] += src1; uses LOCK XADD on x86
    atomic_cas,  // atomic CAS: if mem[src0]==src1 then mem[src0]=dest; uses LOCK CMPXCHG

    // ── FPCR / FPSR ─────────────────────────────────────────────────
    fpcr_read,   // read FPCR into dest register
    fpcr_write,  // write FPCR from src0
    fpsr_read,   // read FPSR into dest register
    fpsr_write,  // write FPSR from src0

    // ── Meta ─────────────────────────────────────────────────────────
    sp_get, // R15 → dst
    sp_put, // src → R15
    entry_point,
    block_start,
    block_end,
};

/// Fixed-size IR operation: 16 bytes.
/// `imm` is 32-bit; 64-bit immediates use a constant-pool reference.
pub const IROp = packed struct {
    tag: Tag,
    dest: u16,
    src0: u16,
    src1: u16,
    flags: u16,
    imm: u32,
    _pad: u16 = 0,

    comptime {
        assert(@sizeOf(IROp) == 16);
    }
};

// ── SIMD element type encoding (stored in flags bits 0-2) ─────
pub const VecElem = enum(u3) {
    i8 = 0,
    i16 = 1,
    i32 = 2,
    i64 = 3,
    f16 = 4,
    f32 = 5,
    f64 = 6,
};

/// SIMD flags helpers.
pub const VecFlags = struct {
    /// Extract element type from flags.
    pub fn elemType(flags: u16) VecElem {
        return @enumFromInt(@as(u3, @truncate(flags & 7)));
    }
    /// Encode element type into flags (preserves other bits).
    pub fn withElem(flags: u16, et: VecElem) u16 {
        return (flags & ~@as(u16, 7)) | @as(u16, @intFromEnum(et));
    }
    /// Check if operation variant is "long" (flags bit 4).
    pub fn isLong(flags: u16) bool { return flags & 0x10 != 0; }
    pub fn isWide(flags: u16) bool { return flags & 0x20 != 0; }
    pub fn isNarrow(flags: u16) bool { return flags & 0x40 != 0; }
};

/// A growable buffer of IR ops.
pub const IRBuffer = struct {
    ops: std.ArrayListUnmanaged(IROp) = .{
        .items = &.{},
        .capacity = 0,
    },

    pub fn append(buf: *IRBuffer, allocator: std.mem.Allocator, op: IROp) !void {
        try buf.ops.append(allocator, op);
    }

    pub fn clear(buf: *IRBuffer) void {
        buf.ops.clearRetainingCapacity();
    }

    pub fn deinit(buf: *IRBuffer, allocator: std.mem.Allocator) void {
        buf.ops.deinit(allocator);
    }
};

test "IROp size and layout" {
    try std.testing.expectEqual(@as(usize, 16), @sizeOf(IROp));
}

test "IRBuffer append and clear" {
    var buf: IRBuffer = .{};
    defer buf.deinit(std.testing.allocator);

    try buf.append(std.testing.allocator, .{
        .tag = .add_i64,
        .dest = 0,
        .src0 = 1,
        .src1 = 2,
        .flags = 0,
        .imm = 42,
    });

    try std.testing.expectEqual(@as(usize, 1), buf.ops.items.len);
    try std.testing.expectEqual(Tag.add_i64, buf.ops.items[0].tag);

    buf.clear();
    try std.testing.expectEqual(@as(usize, 0), buf.ops.items.len);
}
