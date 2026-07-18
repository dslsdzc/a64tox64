//! Translation block — a single-entry, multiple-exit translated code region.

pub const ChainType = enum(u8) {
    none,
    direct,  // B (unconditional branch to known target)
    cond,    // B.cond (conditional branch, two successors)
    call,    // BL (call to known target)
};

pub const TranslationBlock = struct {
    guest_pc: u64,
    host_addr: []u8,
    chain_type: ChainType,
    chain_target: u64, // guest PC of the branch target
    fallthrough_pc: u64, // next PC after block (for untaken branches)

    pub fn init(guest_pc: u64, host_addr: []u8) TranslationBlock {
        return .{
            .guest_pc = guest_pc,
            .host_addr = host_addr,
            .chain_type = .none,
            .chain_target = 0,
            .fallthrough_pc = 0,
        };
    }
};

test "TranslationBlock init" {
    const std = @import("std");
    var host_addr: [4]u8 = .{ 0xC3, 0, 0, 0 };
    const tb = TranslationBlock.init(0x1000, host_addr[0..]);
    try std.testing.expectEqual(@as(u64, 0x1000), tb.guest_pc);
    try std.testing.expectEqual(ChainType.none, tb.chain_type);
}
