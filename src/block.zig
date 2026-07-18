//! Translation block — a single-entry, multiple-exit translated code region.

const std = @import("std");
const Ir = @import("ir.zig");
const IROp = Ir.IROp;

pub const TranslationBlock = struct {
    guest_pc: u64,
    host_addr: []u8,
    ir_ops: []const IROp,
    link_target: ?*TranslationBlock,
    pending_target: u64,
    generation: u32,
    has_indirect: bool,

    pub fn init(guest_pc: u64, host_addr: []u8, ir_ops: []const IROp) TranslationBlock {
        return .{
            .guest_pc = guest_pc,
            .host_addr = host_addr,
            .ir_ops = ir_ops,
            .link_target = null,
            .pending_target = 0,
            .generation = 0,
            .has_indirect = false,
        };
    }
};

test "TranslationBlock init" {
    var host_addr: [4]u8 = .{ 0xC3, 0, 0, 0 };
    const ir_ops: [0]IROp = .{};
    const tb = TranslationBlock.init(0x1000, host_addr[0..], &ir_ops);
    try std.testing.expectEqual(@as(u64, 0x1000), tb.guest_pc);
    try std.testing.expect(tb.link_target == null);
}
