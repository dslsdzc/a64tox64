//! a64tox64 Intermediate Representation.
//! Fixed-size ops with predictable layout.
//! Each op encodes a single machine-level operation in an
//! architecture-neutral form. ARM64 frontend produces these;
//! x86-64 backend consumes them.

const std = @import("std");
const assert = std.debug.assert;

pub const Tag = enum(u16) {
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
    mov_i64, // reg-reg copy
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
    vadd,
    vsub,
    vmul,
    vfadd,
    vfmul,
    vshl,
    vshr,
    fcvt,

    // ── Meta ─────────────────────────────────────────────────────────
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
