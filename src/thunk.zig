const std = @import("std");

pub const THUNK_SIZE: usize = 32;

pub fn emitHostThunk(buf: []u8, host_addr: u64) []u8 {
    buf[0] = 0x48; buf[1] = 0xB8;
    std.mem.writeInt(u64, buf[2..10], host_addr, .little);
    buf[10] = 0xFF; buf[11] = 0xD0;
    buf[12] = 0x48; buf[13] = 0x89; buf[14] = 0xC7;
    buf[15] = 0xC3;
    return buf[0..16];
}

pub fn patchThunkCall(_: []u8, _: u64, _: u64) void {}
