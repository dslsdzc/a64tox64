//! Translation code cache.

const std = @import("std");
const Block = @import("block.zig");
const TranslationBlock = Block.TranslationBlock;

pub const CodeCache = struct {
    allocator: std.mem.Allocator,
    map: std.AutoHashMap(u64, *TranslationBlock),
    blocks: std.ArrayListUnmanaged(*TranslationBlock) = .{ .items = &.{}, .capacity = 0 },
    pages: std.ArrayListUnmanaged([]u8) = .{ .items = &.{}, .capacity = 0 },
    generation: u32,

    pub fn init(allocator: std.mem.Allocator) CodeCache {
        return .{
            .allocator = allocator,
            .map = std.AutoHashMap(u64, *TranslationBlock).init(allocator),
            .blocks = .{ .items = &.{}, .capacity = 0 },
            .pages = .{ .items = &.{}, .capacity = 0 },
            .generation = 0,
        };
    }

    pub fn deinit(cache: *CodeCache) void {
        for (cache.blocks.items) |b| cache.allocator.destroy(b);
        cache.blocks.deinit(cache.allocator);
        for (cache.pages.items) |page| cache.allocator.free(page);
        cache.pages.deinit(cache.allocator);
        cache.map.deinit();
    }

    pub fn lookup(cache: *CodeCache, guest_pc: u64) ?*TranslationBlock {
        return cache.map.get(guest_pc);
    }

    pub fn insert(cache: *CodeCache, block: *TranslationBlock) !void {
        try cache.map.put(block.guest_pc, block);
    }

    pub fn allocateBlock(cache: *CodeCache) !*TranslationBlock {
        const block = try cache.allocator.create(TranslationBlock);
        block.* = TranslationBlock.init(0, &.{});
        try cache.blocks.append(cache.allocator, block);
        return block;
    }

    pub fn allocateCodePage(cache: *CodeCache, size: usize) ![]u8 {
        const page = try std.posix.mmap(
            null,
            size,
            std.posix.PROT{ .READ = true, .WRITE = true, .EXEC = true },
            std.posix.MAP{ .TYPE = .PRIVATE, .ANONYMOUS = true },
            -1,
            0,
        );
        try cache.pages.append(cache.allocator, page);
        return page;
    }

    pub fn invalidatePage(cache: *CodeCache, guest_page_start: u64) void {
        // Note: GDB JIT entries for invalidated blocks are not cleaned up here.
        // In the future, GdbJit.removeBatch() could be called to remove stale
        // GDB JIT entries when a batch's blocks are invalidated, but tracking
        // which batch a block belongs to adds complexity. For now, GDB JIT only
        // adds entries — stale entries remain visible in GDB (harmless).
        const aligned = guest_page_start & ~@as(u64, 0xFFF);
        var it = cache.map.iterator();
        var keys: [64]u64 = undefined;
        var count: usize = 0;
        while (it.next()) |entry| {
            const pc = entry.key_ptr.*;
            if (pc & ~@as(u64, 0xFFF) == aligned) {
                if (count < keys.len) keys[count] = pc;
                count += 1;
            }
        }
        for (keys[0..count]) |key| {
            _ = cache.map.remove(key);
        }
    }
};

test "CodeCache lookup after insert" {
    var cache = CodeCache.init(std.testing.allocator);
    defer cache.deinit();

    const block = try cache.allocateBlock();
    block.* = TranslationBlock.init(0x1000, &.{});
    try cache.insert(block);

    try std.testing.expect(cache.lookup(0x1000) != null);
    try std.testing.expect(cache.lookup(0x2000) == null);
}

test "CodeCache invalidate" {
    var cache = CodeCache.init(std.testing.allocator);
    defer cache.deinit();

    const b1 = try cache.allocateBlock();
    b1.* = TranslationBlock.init(0x1050, &.{});
    try cache.insert(b1);

    cache.invalidatePage(0x1000);
    try std.testing.expect(cache.lookup(0x1050) == null);
}
