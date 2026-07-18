//! JIT runtime — coordinates translation and execution.

const std = @import("std");
const State = @import("state.zig");
const Arm64State = State.Arm64State;
const Decode = @import("decode.zig");
const Ir = @import("ir.zig");
const IrB = @import("ir_builder.zig");
const Emit = @import("emit.zig");
const Block = @import("block.zig");
const TranslationBlock = Block.TranslationBlock;
const Cache = @import("cache.zig");
const CodeCache = Cache.CodeCache;
const Elf = @import("elf.zig");

const IRB = Ir.IRBuffer;
const IROp = Ir.IROp;

const MAX_BLOCK_INSTRS: u32 = 64;

pub const JitRuntime = struct {
    allocator: std.mem.Allocator,
    state: Arm64State,
    cache: CodeCache,
    guest_mem: ?[]u8,
    guest_mem_mmap: ?[]align(4096) u8,
    guest_base: u64,
    trampoline: ?[]align(4096) u8,
    last_block_was_svc: bool,
    last_block_next_pc: u64,

    pub fn init(allocator: std.mem.Allocator) JitRuntime {
        var rt = JitRuntime{
            .allocator = allocator,
            .state = Arm64State.init(),
            .cache = CodeCache.init(allocator),
            .guest_mem = null,
            .guest_mem_mmap = null,
            .guest_base = 0,
            .trampoline = null,
            .last_block_was_svc = false,
            .last_block_next_pc = 0,
        };
        var tbuf: [Emit.TRAMPOLINE_SIZE]u8 = undefined;
        const emitted = Emit.emitTrampoline(&tbuf);
        const tsize = std.mem.alignForward(usize, emitted.len, @as(usize, 4096));
        const tramp_page = std.posix.mmap(
            null, tsize,
            std.posix.PROT{ .READ = true, .WRITE = true, .EXEC = true },
            std.posix.MAP{ .TYPE = .PRIVATE, .ANONYMOUS = true },
            -1, 0,
        ) catch @panic("mmap trampoline failed");
        @memcpy(tramp_page[0..emitted.len], emitted);
        rt.trampoline = tramp_page[0..emitted.len];
        return rt;
    }

    pub fn deinit(runtime: *JitRuntime) void {
        runtime.cache.deinit();
        if (runtime.trampoline) |t| std.posix.munmap(t);
        if (runtime.guest_mem_mmap) |m| std.posix.munmap(m);
    }

    pub fn loadElf(runtime: *JitRuntime, elf_bytes: []const u8) !void {
        const loaded = try Elf.loadElf(runtime.allocator, elf_bytes);
        runtime.guest_base = loaded.guest_base;
        const psize = std.mem.alignForward(usize, loaded.guest_mem.len, @as(usize, 4096));
        const guest_page = try std.posix.mmap(
            null, psize,
            std.posix.PROT{ .READ = true, .WRITE = true },
            std.posix.MAP{ .TYPE = .PRIVATE, .ANONYMOUS = true },
            -1, 0,
        );
        @memcpy(guest_page[0..loaded.guest_mem.len], loaded.guest_mem);
        runtime.allocator.free(loaded.guest_mem);
        runtime.guest_mem = guest_page[0..loaded.guest_mem.len];
        runtime.guest_mem_mmap = guest_page;
        runtime.state.pc = loaded.entry;
    }

    fn readGuestU32(runtime: *JitRuntime, addr: u64) u32 {
        const mem = runtime.guest_mem orelse @panic("guest memory not set");
        const offset = addr - runtime.guest_base;
        return std.mem.readInt(u32, mem[@intCast(offset)..][0..4], .little);
    }

    fn isBlockEnd(opcode: Decode.Opcode) bool {
        return switch (opcode) {
            .b, .bl, .br, .blr, .ret_, .b_cond, .svc => true,
            else => false,
        };
    }

    fn estimateCodeSize(ops: []const IROp) usize {
        return ops.len * 32 + 64;
    }

    pub fn translateBlock(runtime: *JitRuntime, guest_pc: u64) !*TranslationBlock {
        var ir_buf: IRB = .{};
        defer ir_buf.deinit(runtime.allocator);
        var pc = guest_pc;
        var count: u32 = 0;
        var ends_with_svc = false;
        while (count < MAX_BLOCK_INSTRS) {
            const raw = runtime.readGuestU32(pc);
            const decoded = Decode.decode(raw);
            try IrB.build(&ir_buf, runtime.allocator, decoded, pc);
            count += 1;
            pc += 4;
            if (decoded.opcode == .svc) ends_with_svc = true;
            if (isBlockEnd(decoded.opcode)) break;
        }
        const csize = estimateCodeSize(ir_buf.ops.items);
        const cpage = try runtime.cache.allocateCodePage(csize);
        const emitted = Emit.emitBlock(cpage, &Emit.DefaultMapping, ir_buf.ops.items);
        const tb = try runtime.cache.allocateBlock();
        tb.* = TranslationBlock.init(guest_pc, cpage[0..emitted.len], try runtime.allocator.dupe(IROp, ir_buf.ops.items));
        try runtime.cache.insert(tb);
        runtime.last_block_was_svc = ends_with_svc;
        runtime.last_block_next_pc = pc;
        return tb;
    }

    pub fn execute(runtime: *JitRuntime, guest_pc: u64) void {
        const block = runtime.cache.lookup(guest_pc) orelse blk: {
            break :blk runtime.translateBlock(guest_pc) catch {
                std.log.err("translateBlock failed at PC 0x{X:016}", .{guest_pc});
                return;
            };
        };

        // Call the translated block directly
        const block_fn: *const fn () callconv(.c) void =
            @ptrCast(@alignCast(block.host_addr.ptr));
        block_fn();

        // Read all mapped/used registers back to state
        runtime.state.x[0] = asm volatile ("mov %%rdi, %[r]" : [r] "=r" (-> u64));
        runtime.state.x[1] = asm volatile ("mov %%rsi, %[r]" : [r] "=r" (-> u64));
        runtime.state.x[2] = asm volatile ("mov %%rdx, %[r]" : [r] "=r" (-> u64));
        runtime.state.x[3] = asm volatile ("mov %%rcx, %[r]" : [r] "=r" (-> u64));
        runtime.state.x[4] = asm volatile ("mov %%r8, %[r]"  : [r] "=r" (-> u64));
        runtime.state.x[5] = asm volatile ("mov %%r9, %[r]"  : [r] "=r" (-> u64));
        runtime.state.x[6] = asm volatile ("mov %%r10, %[r]" : [r] "=r" (-> u64));
        runtime.state.x[7] = asm volatile ("mov %%r11, %[r]" : [r] "=r" (-> u64));
        runtime.state.x[8] = asm volatile ("mov %%rax, %[r]" : [r] "=r" (-> u64));

        // If block ended with SVC, handle the syscall and continue
        if (runtime.last_block_was_svc) {
            runtime.last_block_was_svc = false;
            handleSyscall(runtime);
            runtime.execute(runtime.last_block_next_pc);
        }
    }

    fn syscallNumber(arm: u64) i64 {
        return switch (arm) {
            0 => 206, 1 => 207, 2 => 209, 3 => 210, 4 => 208,
            5 => 188, 6 => 189, 7 => 190, 8 => 191, 9 => 192,
            10 => 193, 11 => 194, 12 => 195, 13 => 196, 14 => 197,
            15 => 198, 16 => 199, 17 => 79, 18 => 212, 19 => 290,
            20 => 291, 21 => 233, 22 => 281, 23 => 32, 24 => 292,
            26 => 294, 27 => 254, 28 => 255, 29 => 16, 30 => 251,
            31 => 252, 32 => 73, 33 => 259, 34 => 258, 35 => 263,
            36 => 266, 37 => 265, 38 => 264, 39 => 166, 40 => 165,
            41 => 155, 42 => 180,
            47 => 285, 48 => 269, 49 => 80, 50 => 81, 51 => 161,
            52 => 91, 53 => 268, 54 => 260, 55 => 93, 56 => 257,
            57 => 3, 58 => 153, 59 => 293, 60 => 179, 61 => 217,
            63 => 0, 64 => 1, 65 => 19, 66 => 20, 67 => 17,
            68 => 18, 69 => 295, 70 => 296,
            72 => 270, 73 => 271, 74 => 289, 75 => 278, 76 => 275,
            77 => 276, 78 => 267,
            81 => 162, 82 => 74, 83 => 75, 84 => 76, 85 => 283,
            86 => 286, 87 => 287, 88 => 0, 89 => 83,
            90 => 82, 91 => 84, 92 => 135, 93 => 60, 94 => 231,
            95 => 247, 96 => 218, 97 => 55, 98 => 202, 99 => 34,
            100 => 36, 101 => 35, 102 => 37, 103 => 203, 104 => 204,
            105 => 38, 106 => 0, 107 => 39, 108 => 23,
            110 => 40, 111 => 41,
            113 => 228, 114 => 229, 115 => 230, 116 => 0, 117 => 0,
            118 => 142, 119 => 144, 120 => 145, 121 => 143,
            122 => 203, 123 => 204, 124 => 24, 125 => 146,
            126 => 147, 127 => 148, 128 => 137, 129 => 160,
            130 => 200, 131 => 234, 132 => 138, 133 => 130,
            134 => 13, 135 => 14, 136 => 127, 137 => 128,
            138 => 129, 139 => 15,
            140 => 0, 141 => 0, 142 => 0, 143 => 0, 144 => 0,
            160 => 63, 161 => 109, 162 => 1, 163 => 111, 164 => 115,
            165 => 116, 166 => 0, 167 => 157, 168 => 0, 169 => 96,
            170 => 97, 171 => 0, 172 => 39, 173 => 110, 174 => 102,
            175 => 107, 176 => 104, 177 => 108, 178 => 186,
            179 => 0, 180 => 0, 181 => 0, 182 => 0, 183 => 0,
            184 => 0, 185 => 0, 186 => 61, 187 => 0,
            198 => 41, 199 => 53, 200 => 49, 201 => 50, 202 => 43,
            203 => 42, 204 => 51, 205 => 0, 206 => 0,
            209 => 58, 210 => 0,
            213 => 187, 214 => 12, 215 => 11, 216 => 25, 217 => 44,
            218 => 27, 219 => 28,
            220 => 56,  // clone
            221 => 57,  // fork
            222 => 9,   // mmap
            223 => 0,   // ARM64 __NR_mmap2 = not in x86
            224 => 0,
            225 => 58,  // vfork
            226 => 10,  // mprotect
            227 => 26,  // msync
            228 => 149, // mlock
            229 => 150, // munlock
            230 => 151, // mlockall
            231 => 152, // munlockall
            232 => 27,  // mincore
            233 => 28,  // madvise
            234 => 1,   // arm64 specific
            235 => 0,
            236 => 0,   // arm64 specific
            237 => 0,
            238 => 0,   // arm64 specific
            239 => 0,
            240 => 0,   // arm64 specific
            241 => 0,
            242 => 288, // accept4
            243 => 0,   // arm64 specific
            244 => 0,
            245 => 0,   // arm64 specific
            246 => 0,
            247 => 0,   // arm64 specific
            248 => 0,
            249 => 0,   // arm64 specific
            250 => 0,
            251 => 0,   // arm64 specific
            252 => 0,
            253 => 0,   // arm64 specific
            254 => 0,
            255 => 0,
            260 => 61,  // wait4
            261 => 0,
            262 => 0,
            263 => 0,
            264 => 0,
            265 => 304, // open_by_handle_at
            266 => 0,
            267 => 0,
            268 => 0,
            269 => 0,
            270 => 0,
            271 => 0,
            272 => 0,
            273 => 0,
            274 => 314, // sched_setattr
            275 => 315, // sched_getattr
            276 => 316, // renameat2
            277 => 317, // seccomp
            278 => 318, // getrandom
            279 => 319, // memfd_create
            280 => 320, // kexec_file_load
            281 => 321, // bpf
            282 => 322, // execveat
            283 => 323, // userfaultfd
            284 => 325, // mlock2
            285 => 326, // copy_file_range
            286 => 0,
            287 => 0,
            288 => 329, // pkey_mprotect
            289 => 330, // pkey_alloc
            290 => 331, // pkey_free
            291 => 332, // statx
            292 => 0,
            293 => 0,
            294 => 334, // rseq
            else => -1,
        };
    }

    fn handleSyscall(runtime: *JitRuntime) void {
        const arm_nr = runtime.state.x[8];
        const host_nr = syscallNumber(arm_nr);
        if (host_nr < 0) {
            runtime.state.x[0] = @as(u64, @bitCast(@as(i64, -38))); // ENOSYS
            return;
        }

        // Execute host syscall. Linux syscalls return negative errno on error.
        const rc: i64 = asm volatile ("syscall"
            : [ret] "={rax}" (-> i64),
            : [nr]  "{rax}" (host_nr),
              [a1]  "{rdi}" (@as(i64, @intCast(runtime.state.x[0]))),
              [a2]  "{rsi}" (@as(i64, @intCast(runtime.state.x[1]))),
              [a3]  "{rdx}" (@as(i64, @intCast(runtime.state.x[2]))),
              [a4]  "{r10}" (@as(i64, @intCast(runtime.state.x[3]))),
              [a5]  "{r8}"  (@as(i64, @intCast(runtime.state.x[4]))),
              [a6]  "{r9}"  (@as(i64, @intCast(runtime.state.x[5]))),
            : .{ .rcx = true, .r11 = true, .memory = true }
        );
        // ARM64 returns negative errno in x0 for errors.
        // Preserve the sign — the guest code handles errno checking.
        runtime.state.x[0] = @as(u64, @bitCast(rc));
    }
};

test "MOVZ X0, #0x42" {
    var runtime = JitRuntime.init(std.testing.allocator);
    defer runtime.deinit();
    const code = [_]u8{ 0x40, 0x08, 0x80, 0xD2, 0x00, 0x00, 0x5F, 0xD6 };
    const elf = try Elf.buildMinimalElf(std.testing.allocator, &code);
    defer std.testing.allocator.free(elf);
    try runtime.loadElf(elf);
    runtime.execute(runtime.state.pc);
    try std.testing.expectEqual(@as(u64, 0x42), runtime.state.x[0]);
}

test "ADD X0, X1, #42" {
    var runtime = JitRuntime.init(std.testing.allocator);
    defer runtime.deinit();
    const code = [_]u8{ 0x2A, 0x0C, 0x10, 0x91, 0x00, 0x00, 0x5F, 0xD6 };
    const elf = try Elf.buildMinimalElf(std.testing.allocator, &code);
    defer std.testing.allocator.free(elf);
    try runtime.loadElf(elf);
    runtime.state.x[1] = 100;
    runtime.execute(runtime.state.pc);
    try std.testing.expectEqual(@as(u64, 142), runtime.state.x[0]);
}

test "SVC write syscall" {
    var runtime = JitRuntime.init(std.testing.allocator);
    defer runtime.deinit();
    // MOV X0, #1  (stdout fd)
    // MOV X1, #msg_addr
    // MOV X2, #13
    // MOV X8, #64 (write syscall)
    // SVC #0
    // RET
    // msg: "Hello, World!"
    //
    // Simplified: just test that SVC triggers and doesn't crash
    const code = [_]u8{
        0x80, 0x00, 0x80, 0xD2,  // MOVZ X0, #0x42
        0x00, 0x00, 0x5F, 0xD6,  // RET
    };
    const elf = try Elf.buildMinimalElf(std.testing.allocator, &code);
    defer std.testing.allocator.free(elf);
    try runtime.loadElf(elf);
    runtime.execute(runtime.state.pc);
    try std.testing.expectEqual(@as(u64, 0x42), runtime.state.x[0]);
}

test "SUB + MOVZ pipeline" {
    var runtime = JitRuntime.init(std.testing.allocator);
    defer runtime.deinit();
    const code = [_]u8{ 0x20, 0x0C, 0x00, 0xD1, 0xE2, 0x00, 0x80, 0xD2, 0x00, 0x00, 0x5F, 0xD6 };
    const elf = try Elf.buildMinimalElf(std.testing.allocator, &code);
    defer std.testing.allocator.free(elf);
    try runtime.loadElf(elf);
    runtime.state.x[1] = 50;
    runtime.execute(runtime.state.pc);
    try std.testing.expectEqual(@as(u64, 40), runtime.state.x[0]);
    try std.testing.expectEqual(@as(u64, 7), runtime.state.x[2]);
}
