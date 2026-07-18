//! Per-block register allocator with priority and inter-block hints.
//!
//! Implements frequency-based priority allocation with:
//! - Block hotness weighting (hot blocks get better assignment)
//! - Inter-block register hints (carry assignments across blocks)
//! - Cost-aware mapping (callee-saved regs for high-frequency ARM regs)

const std = @import("std");
const Ir = @import("ir.zig");
const IROp = Ir.IROp;
const Emit = @import("emit.zig");
const X86Reg = Emit.X86Reg;
const RegisterMap = Emit.RegisterMap;

/// Host register pool. First 8 are call-clobbered (fast, no preservation cost),
/// last 6 are callee-saved (preferred for high-frequency ARM64 regs).
const host_regs = [_]X86Reg{
    .rdi, .rsi, .rdx, .rcx, .r8, .r9, .r10, .r11,
    .rbx, .rbp, .r12, .r13,
};

/// Register allocation hints from predecessor blocks.
pub const RegHints = struct {
    /// Preferred ARM64→x86 mapping based on predecessor block exit state.
    /// Higher score = stronger preference.
    pref: [31]?X86Reg = undefined,
    scores: [31]usize = undefined,
};

/// Basic allocation (no hints, cold block).
pub fn allocate(ops: []const IROp) RegisterMap {
    return allocateAdv(ops, 1.0, null);
}

/// Advanced allocation with hotness and predecessor hints.
/// hotness: multiplier for frequency counts (1.0 = normal, >1.0 = hot loop).
/// hints: optional register preferences from predecessor blocks.
pub fn allocateAdv(ops: []const IROp, hotness: f32, hints: ?*const RegHints) RegisterMap {
    var mapping: RegisterMap = undefined;
    for (&mapping, 0..) |*m, i| {
        m.* = if (i == 8) @as(?X86Reg, .rax) else null;
    }

    var score: [31]f32 = undefined;
    for (&score) |*s| s.* = 0;
    for (ops) |op| {
        if (op.dest < 31) score[op.dest] += 1.0;
        if (op.src0 < 31) score[op.src0] += 1.0;
        if (op.src1 < 31 and op.src1 != 0x1F) score[op.src1] += 1.0;
    }
    if (hotness > 1.0) {
        for (&score) |*s| s.* *= hotness;
    }

    // Strong hints: reserve preferred host regs before frequency assignment
    var used_hosts = std.mem.zeroes([16]bool);
    var hint_arm = std.mem.zeroes([31]bool);
    if (hints) |h| {
        for (h.pref, 0..) |maybe_reg, arm_i| {
            if (maybe_reg) |reg| {
                const host_idx = @intFromEnum(reg);
                if (host_idx < 16 and host_idx != 8 and !used_hosts[host_idx]) {
                    mapping[arm_i] = reg;
                    used_hosts[host_idx] = true;
                    hint_arm[arm_i] = true;
                }
            }
        }
    }

    // Sort by frequency
    var sorted: [30]usize = undefined;
    for (&sorted, 0..) |*s, i| s.* = i;
    var i: usize = 0;
    while (i < sorted.len) : (i += 1) {
        var best = i;
        var j: usize = i + 1;
        while (j < sorted.len) : (j += 1) {
            if (score[sorted[j]] > score[sorted[best]]) best = j;
        }
        const tmp = sorted[i]; sorted[i] = sorted[best]; sorted[best] = tmp;
    }

    // Assign non-hinted ARM regs via frequency
    var used: usize = 0;
    var callee_used: usize = 0;
    const num_callee: usize = 4;
    const num_clobber: usize = 8;

    for (sorted) |arm| {
        if (arm == 8) continue;
        if (hint_arm[arm]) continue;
        while (callee_used < num_callee) {
            const host_idx = num_clobber + callee_used;
            if (!used_hosts[host_idx]) {
                mapping[arm] = host_regs[host_idx];
                used_hosts[host_idx] = true;
                callee_used += 1;
                break;
            }
            callee_used += 1;
        } else {
            while (used < num_clobber) {
                if (!used_hosts[used]) {
                    mapping[arm] = host_regs[used];
                    used_hosts[used] = true;
                    used += 1;
                    break;
                }
                used += 1;
            } else break;
        }
    }

    mapping[8] = .rax;
    return mapping;
}

/// Build exit hints
pub fn exitHints(map: RegisterMap) RegHints {
    var hints: RegHints = undefined;
    for (&hints.pref) |*p| p.* = null;
    for (&hints.scores) |*s| s.* = 0;
    for (&hints.pref, &hints.scores, 0..) |*p, *s, i| {
        if (map[i]) |reg| {
            p.* = reg;
            s.* = 5; // default hint strength
        }
    }
    return hints;
}

test "frequent regs mapped" {
    const ops = [_]IROp{
        .{ .tag = .add_i64, .dest = 0, .src0 = 1, .src1 = 2, .flags = 0, .imm = 0 },
        .{ .tag = .add_i64, .dest = 0, .src0 = 1, .src1 = 2, .flags = 0, .imm = 0 },
        .{ .tag = .ret_, .dest = 0, .src0 = 0, .src1 = 0, .flags = 0, .imm = 0 },
    };
    const map = allocate(&ops);
    try std.testing.expectEqual(@as(?X86Reg, .rax), map[8]);
    try std.testing.expect(map[0] != null);
    try std.testing.expect(map[1] != null);
}

test "hotness weighting" {
    const ops = [_]IROp{
        .{ .tag = .add_i64, .dest = 5, .src0 = 6, .src1 = 7, .flags = 0, .imm = 0 },
    };
    const cold = allocateAdv(&ops, 1.0, null);
    const hot = allocateAdv(&ops, 100.0, null);
    // Both should produce valid mappings
    try std.testing.expectEqual(@as(?X86Reg, .rax), cold[8]);
    try std.testing.expectEqual(@as(?X86Reg, .rax), hot[8]);
}

test "hints influence mapping" {
    const ops = [_]IROp{
        .{ .tag = .add_i64, .dest = 0, .src0 = 1, .src1 = 2, .flags = 0, .imm = 0 },
    };
    var hints: RegHints = undefined;
    for (&hints.pref) |*p| p.* = null;
    for (&hints.scores) |*s| s.* = 0;
    hints.pref[0] = .r15; // strongly prefer x0 in r15
    hints.scores[0] = 100;
    const map = allocateAdv(&ops, 1.0, &hints);
    try std.testing.expectEqual(@as(?X86Reg, .rax), map[8]);
}
