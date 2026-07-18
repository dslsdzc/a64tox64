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

    pub fn init(allocator: std.mem.Allocator) JitRuntime {
        var rt = JitRuntime{
            .allocator = allocator,
            .state = Arm64State.init(),
            .cache = CodeCache.init(allocator),
            .guest_mem = null,
            .guest_mem_mmap = null,
            .guest_base = 0,
            .trampoline = null,
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
        while (count < MAX_BLOCK_INSTRS) {
            const raw = runtime.readGuestU32(pc);
            const decoded = Decode.decode(raw);
            try IrB.build(&ir_buf, runtime.allocator, decoded, pc);
            count += 1;
            pc += 4;
            if (isBlockEnd(decoded.opcode)) break;
        }
        const csize = estimateCodeSize(ir_buf.ops.items);
        const cpage = try runtime.cache.allocateCodePage(csize);
        const emitted = Emit.emitBlock(cpage, &Emit.DefaultMapping, ir_buf.ops.items);
        const tb = try runtime.cache.allocateBlock();
        tb.* = TranslationBlock.init(guest_pc, cpage[0..emitted.len], try runtime.allocator.dupe(IROp, ir_buf.ops.items));
        try runtime.cache.insert(tb);
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

        // Save result register (RDI = ARM64 x0) back to state
        const result = asm volatile (
            \\ mov %%rdi, %[r]
            : [r] "=r" (-> u64),
        );
        runtime.state.x[0] = result;
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

test "MOVZ with page_allocator (CLI match)" {
    var runtime = JitRuntime.init(std.heap.page_allocator);
    defer runtime.deinit();
    const code = [_]u8{ 0x40, 0x08, 0x80, 0xD2, 0x00, 0x00, 0x5F, 0xD6 };
    const elf = try Elf.buildMinimalElf(std.heap.page_allocator, &code);
    defer std.heap.page_allocator.free(elf);
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
