//! Per-block register allocator using frequency-based priority.

const std = @import("std");
const Ir = @import("ir.zig");
const IROp = Ir.IROp;
const Emit = @import("emit.zig");
const X86Reg = Emit.X86Reg;
const RegisterMap = Emit.RegisterMap;

const host_regs = [_]X86Reg{
    .rdi, .rsi, .rdx, .rcx, .r8, .r9, .r10, .r11,
    .rbx, .rbp, .r12, .r13, .r14, .r15,
};

pub fn allocate(ops: []const IROp) RegisterMap {
    var mapping: RegisterMap = undefined;
    for (&mapping, 0..) |*m, i| {
        // x8 → RAX always; others start unmapped
        m.* = if (i == 8) @as(?X86Reg, .rax) else null;
    }

    var freq: [31]usize = undefined;
    @memset(&freq, 0);
    for (ops) |op| {
        if (op.dest < 31) freq[op.dest] += 1;
        if (op.src0 < 31) freq[op.src0] += 1;
        if (op.src1 < 31 and op.src1 != 0x1F) freq[op.src1] += 1;
    }

    var sorted: [30]usize = undefined;
    for (&sorted, 0..) |*s, i| s.* = i;

    var i: usize = 0;
    while (i < sorted.len) : (i += 1) {
        var best = i;
        var j: usize = i + 1;
        while (j < sorted.len) : (j += 1) {
            if (freq[sorted[j]] > freq[sorted[best]]) best = j;
        }
        const tmp = sorted[i];
        sorted[i] = sorted[best];
        sorted[best] = tmp;
    }

    var used: usize = 0;
    for (sorted) |arm| {
        if (arm == 8) continue;
        if (used >= host_regs.len) break;
        mapping[arm] = host_regs[used];
        used += 1;
    }

    mapping[8] = .rax;
    return mapping;
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

test "empty block" {
    const ops: [0]IROp = .{};
    const map = allocate(&ops);
    try std.testing.expectEqual(@as(?X86Reg, .rax), map[8]);
}
