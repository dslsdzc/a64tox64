//! ARM64 guest architectural state.
//!
//! Holds the full register file and system state
//! of the emulated ARM64 CPU. The JIT compiler reads from and
//! writes to this state both directly (via register mapping)
//! and indirectly (when spilling or syncing).

const std = @import("std");

/// Complete ARM64 architectural state.
pub const Arm64State = struct {
    /// General-purpose registers x0-x30
    x: [31]u64,

    /// Stack pointer
    sp: u64,

    /// Program counter
    pc: u64,

    /// Condition flags (PSTATE NZCV)
    nzcv: u32,

    /// Floating-point control register
    fpcr: u32,

    /// Floating-point status register
    fpsr: u32,

    /// SIMD/FP registers v0-v31 (128-bit NEON)
    v: [32]u128,

    pub fn init() Arm64State {
        return std.mem.zeroes(Arm64State);
    }
};

test "Arm64State zero-initialized" {
    const state = Arm64State.init();
    try std.testing.expectEqual(@as(u64, 0), state.x[0]);
    try std.testing.expectEqual(@as(u64, 0), state.sp);
    try std.testing.expectEqual(@as(u32, 0), state.nzcv);
}
