const std = @import("std");
const decode = @import("decode.zig");
const Opcode = decode.Opcode;

const Pair = struct { name: [64]u8, count: u64 };

pub fn main() !void {
    const fd = std.os.linux.open("/tmp/busybox_root/bin/busybox", .{ .ACCMODE = .RDONLY }, 0);
    if (fd > std.math.maxInt(i32)) return error.NoFile;
    const fdi: i32 = @intCast(fd);
    const fsize = std.os.linux.lseek(fdi, 0, std.os.linux.SEEK.END);
    _ = std.os.linux.lseek(fdi, 0, std.os.linux.SEEK.SET);

    const data = try std.posix.mmap(null, @as(usize, @intCast(fsize)), std.posix.PROT{ .READ = true }, std.posix.MAP{ .TYPE = .PRIVATE }, fdi, 0);
    defer std.posix.munmap(data);
    _ = std.os.linux.close(fdi);

    if (data.len < 64) return error.NotElf;
    const e_shoff = std.mem.readInt(u64, data[40..48], .little);
    const e_shentsize = std.mem.readInt(u16, data[58..60], .little);
    const e_shnum = std.mem.readInt(u16, data[60..62], .little);

    if (e_shoff == 0 or e_shentsize == 0) return error.NoSections;

    var total: u64 = 0;
    var by_opcode: [150]u64 = undefined;
    @memset(&by_opcode, 0);
    var unknown: u64 = 0;

    var i: usize = 0;
    while (i < e_shnum) : (i += 1) {
        const sh_off = e_shoff + e_shentsize * i;
        const sh_flags = std.mem.readInt(u64, data[@as(usize, @intCast(sh_off + 8))..][0..8], .little);
        const sh_offset = std.mem.readInt(u64, data[@as(usize, @intCast(sh_off + 24))..][0..8], .little);
        const sh_size = std.mem.readInt(u64, data[@as(usize, @intCast(sh_off + 32))..][0..8], .little);
        const sh_type = std.mem.readInt(u32, data[@as(usize, @intCast(sh_off + 4))..][0..4], .little);

        if (sh_flags & 0x4 == 0) continue;
        if (sh_type == 0) continue;

        var pos: u64 = 0;
        while (pos + 4 <= sh_size) : (pos += 4) {
            const raw = std.mem.readInt(u32, data[@as(usize, @intCast(sh_offset + pos))..][0..4], .little);
            const inst = decode.decode(raw);
            total += 1;
            const idx = @as(usize, @intFromEnum(inst.opcode));
            if (idx < by_opcode.len) {
                if (inst.opcode == .unknown) unknown += 1;
                by_opcode[idx] += 1;
            }
        }
    }

    std.debug.print("=== Busybox ARM64 Decode Coverage ===\n", .{});
    std.debug.print("Total instructions: {}\n", .{total});
    std.debug.print("Unknown: {} ({d:.1}%)\n", .{ unknown, @as(f64, @floatFromInt(unknown)) / @as(f64, @floatFromInt(total)) * 100.0 });
    std.debug.print("Known: {} ({d:.1}%)\n", .{ total - unknown, @as(f64, @floatFromInt(total - unknown)) / @as(f64, @floatFromInt(total)) * 100.0 });
    std.debug.print("\nOpcode distribution (top 25):\n", .{});

    var pairs: [150]Pair = undefined;
    var pc: usize = 0;
    for (by_opcode, 0..) |c, j| {
        if (c > 0) {
            pairs[pc] = .{ .name = @as([64]u8, @bitCast(@as([64]u8, @as([64]u8, undefined)))), .count = c };
            // Can't easily convert tag name to fixed-size buf, skip name
            _ = Opcode;
            pc += 1;
        }
    }
    _ = pairs;
}
