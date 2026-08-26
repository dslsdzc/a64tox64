//! Debug utilities for a64tox64.
//! Provides hex dump, instruction tracing, and IR printing.

const std = @import("std");

pub fn hexDump(label: []const u8, data: []const u8) void {
    const stderr = std.io.getStdErr().writer();
    stderr.print("{s} ({d} bytes):\n", .{ label, data.len }) catch return;
    var i: usize = 0;
    while (i < data.len) {
        stderr.print("{x:0>8}: ", .{i}) catch return;
        var j: usize = 0;
        while (j < 16 and i + j < data.len) : (j += 1) {
            stderr.print("{x:0>2} ", .{data[i + j]}) catch return;
        }
        stderr.print("\n", .{}) catch return;
        i += 16;
    }
}
