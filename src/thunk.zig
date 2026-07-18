const std = @import("std");
const Emit = @import("emit.zig");

pub const THUNK_SIZE: usize = 128;

pub fn emitHostThunk(buf: []u8, host_addr: u64, args_gt_6: bool) []u8 {
    _ = args_gt_6;
    var ctx = Emit.EmitContext.init(buf, &Emit.DefaultMapping);
    ctx.byte(0xE8);
    const delta = @as(i64, @intCast(host_addr)) - (@as(i64, @intCast(@intFromPtr(buf.ptr))) + 5);
    ctx.bytes(std.mem.asBytes(&@as(i32, @intCast(delta))));
    ctx.byte(0x48); ctx.byte(0x89); ctx.byte(0xC7); // mov rdi, rax
    ctx.byte(0xC3); // ret
    return buf[0..ctx.offset];
}

pub fn patchThunkCall(thunk: []u8, code_addr: u64, host_addr: u64) void {
    if (thunk.len < 6 or thunk[0] != 0xE8) return;
    const delta = @as(i64, @intCast(host_addr)) - (@as(i64, @intCast(code_addr)) + 5);
    std.mem.writeInt(i32, thunk[1..5], @as(i32, @intCast(delta)), .little);
}
