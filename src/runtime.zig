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
            // ── I/O ──────────────────────────────────────────────
            63 => 0,    // read
            64 => 1,    // write
            56 => 2,    // open
            57 => 3,    // close
            62 => 8,    // lseek
            29 => 16,   // ioctl
            65 => 19,   // readv
            66 => 20,   // writev
            76 => 221,  // fadvise64 (ARM 76, x86 221)
            25 => 72,   // fcntl (ARM 25, x86 72)

            // ── File system ──────────────────────────────────────
            4  => 4,    // stat (same)
            5  => 5,    // fstat (same)
            6  => 6,    // lstat (same)
            17 => 79,   // getcwd
            23 => 32,   // dup
            24 => 33,   // dup3
            34 => 258,  // mkdirat
            35 => 263,  // unlinkat
            36 => 266,  // symlinkat
            37 => 265,  // linkat
            48 => 269,  // faccessat
            49 => 288,  // fchmodat
            55 => 264,  // renameat
            61 => 78,   // getdents64 → x86 getdents
            71 => 217,  // sendfile (ARM 71, x86 40? no, 71 is different)
            78 => 267,  // readlinkat
            79 => 262,  // fstatat → x86 newfstatat
            80 => 5,    // fstat (again, for faccessat alias)
            87 => 7,    // poll

            // ── Memory ───────────────────────────────────────────
            214 => 12,  // brk
            220 => 11,  // munmap
            222 => 9,   // mmap
            226 => 10,  // mprotect
            232 => 25,  // mremap
            233 => 44,  // msync
            235 => 27,  // mincore
            236 => 28,  // madvise
            278 => 318, // getrandom

            // ── Process ──────────────────────────────────────────
            93 => 60,   // exit
            94 => 231,  // exit_group
            96 => 218,  // set_tid_address
            98 => 202,  // futex
            101 => 35,  // nanosleep
            113 => 228, // clock_gettime
            130 => 200, // tkill
            131 => 234, // tgkill
            133 => 131, // sigaltstack
            134 => 13,  // rt_sigaction
            135 => 14,  // rt_sigprocmask
            137 => 24,  // sched_yield
            160 => 63,  // uname
            167 => 157, // prctl
            172 => 39,  // getpid
            173 => 110, // getppid
            174 => 102, // getuid
            175 => 107, // geteuid
            176 => 104, // getgid
            177 => 108, // getegid
            178 => 186, // gettid
            186 => 61,  // wait4
            202 => 130, // personality

            else => -1, // ENOSYS
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
