//! a64tox64 — ARM64 → x86-64 JIT Dynamic Binary Translator.
//! CLI: a64tox64 <arm64-elf>

const std = @import("std");

pub const ir = @import("ir.zig");
pub const state = @import("state.zig");
pub const decode = @import("decode.zig");
pub const ir_builder = @import("ir_builder.zig");
pub const emit = @import("emit.zig");
pub const emit_direct = @import("emit_direct.zig");
pub const block = @import("block.zig");
pub const cache = @import("cache.zig");
pub const runtime = @import("runtime.zig");
pub const elf = @import("elf.zig");

pub fn main() !void {
    const allocator = std.heap.page_allocator;
    const linux = std.os.linux;

    var cmdline: [4096]u8 = undefined;
    const cfd = linux.open("/proc/self/cmdline", .{ .ACCMODE = .RDONLY }, 0);
    if (cfd > std.math.maxInt(i32)) return error.NoCmdline;
    const n = linux.read(@as(i32, @intCast(cfd)), @as([*]u8, &cmdline), cmdline.len);
    _ = linux.close(@as(i32, @intCast(cfd)));

    var pos: usize = 0;
    while (pos < n and cmdline[pos] != 0) : (pos += 1) {}
    pos += 1;
    while (pos < n and cmdline[pos] == 0) : (pos += 1) {}
    const as = pos;
    while (pos < n and cmdline[pos] != 0) : (pos += 1) {}
    if (pos == as) return error.NoArgs;

    var path: [4096]u8 = undefined;
    @memset(&path, 0);
    @memcpy(path[0 .. pos - as], cmdline[as..pos]);

    const fd = linux.open(@as([*:0]u8, @ptrCast(&path)), .{ .ACCMODE = .RDONLY }, 0);
    if (fd > std.math.maxInt(i32)) return error.NoFile;
    const fdi: i32 = @intCast(fd);
    const fsize = linux.lseek(fdi, 0, linux.SEEK.END);
    _ = linux.lseek(fdi, 0, linux.SEEK.SET);

    const mapped = std.posix.mmap(null, fsize, std.posix.PROT{ .READ = true }, std.posix.MAP{ .TYPE = .PRIVATE }, fdi, 0) catch return error.MmapFail;
    _ = linux.close(fdi);

    var jit = runtime.JitRuntime.init(allocator);
    defer jit.deinit();
    try jit.loadElf(mapped);
    std.posix.munmap(mapped);
    jit.execute(jit.state.pc);
    linux.exit(@as(i32, @intCast(jit.state.x[0] & 0xFF)));
}

test { _ = ir; _ = state; _ = decode; _ = ir_builder; _ = emit; _ = emit_direct; _ = block; _ = cache; _ = runtime; _ = elf; }
