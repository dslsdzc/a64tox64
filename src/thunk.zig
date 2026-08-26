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

/// No-op: thunks emitted by emitHostThunk embed host_addr at compile time
/// (as an immediate in the MOV RAX, imm64 instruction), so no runtime
/// patching is needed. This function is called by resolvePltEntries in
/// runtime.zig for all resolved PLT entries but is intentionally empty.
pub fn patchThunkCall(buf: []u8, code_addr: u64, host_addr: u64) void {
    _ = buf;
    _ = code_addr;
    _ = host_addr;
    // Thunks emitted by emitHostThunk embed host_addr at compile time
    // (MOV RAX, imm64 at offset 0-9). No runtime patching required.
}
