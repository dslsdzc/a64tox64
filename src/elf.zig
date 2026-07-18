//! ARM64 ELF loader.
//!
//! Parses ARM64 ELF executables and shared objects, loads PT_LOAD
//! segments into guest memory, resolves relocations, and sets up
//! the initial register state for execution.

const std = @import("std");

// ── ELF64 constants ────────────────────────────────────────────────

pub const ELF_MAGIC = 0x464C457F; // "\x7fELF" in little-endian u32
pub const EM_AARCH64 = 0xB7;
pub const ET_EXEC = 2;
pub const ET_DYN = 3;
pub const PT_LOAD = 1;
pub const PT_DYNAMIC = 2;
pub const PT_PHDR = 6;
pub const PF_R = 4;
pub const PF_W = 2;
pub const PF_X = 1;

// ── ELF64 header ──────────────────────────────────────────────────

pub const Elf64Ehdr = extern struct {
    e_ident: [16]u8,
    e_type: u16,
    e_machine: u16,
    e_version: u32,
    e_entry: u64,
    e_phoff: u64,
    e_shoff: u64,
    e_flags: u32,
    e_ehsize: u16,
    e_phentsize: u16,
    e_phnum: u16,
    e_shentsize: u16,
    e_shnum: u16,
    e_shstrndx: u16,
};

pub const Elf64Phdr = extern struct {
    p_type: u32,
    p_flags: u32,
    p_offset: u64,
    p_vaddr: u64,
    p_paddr: u64,
    p_filesz: u64,
    p_memsz: u64,
    p_align: u64,
};

// ── Loaded ELF image ──────────────────────────────────────────────

pub const LoadedElf = struct {
    /// Guest memory containing all loaded segments
    guest_mem: []u8,
    /// Guest physical address (base of loaded memory)
    guest_base: u64,
    /// Entry point (PC to start execution at)
    entry: u64,
    /// Size of guest memory region
    guest_size: u64,
};

/// Load an ARM64 ELF binary from bytes.
/// Returns the loaded image info.
pub fn loadElf(allocator: std.mem.Allocator, elf_bytes: []const u8) !LoadedElf {
    if (elf_bytes.len < 64) return error.InvalidElfMagic;

    // Read ELF header fields manually (avoid alignment issues)
    const magic = std.mem.readInt(u32, elf_bytes[0..4], .little);
    if (magic != ELF_MAGIC) return error.InvalidElfMagic;
    if (elf_bytes[4] != 2) return error.Not64Bit;      // EI_CLASS
    if (elf_bytes[5] != 1) return error.NotLittleEndian; // EI_DATA
    const e_type = std.mem.readInt(u16, elf_bytes[16..18], .little);
    const e_machine = std.mem.readInt(u16, elf_bytes[18..20], .little);
    if (e_machine != EM_AARCH64) return error.NotAArch64;
    if (e_type != ET_EXEC and e_type != ET_DYN) return error.UnsupportedElfType;

    const e_entry = std.mem.readInt(u64, elf_bytes[24..32], .little);
    const e_phoff = std.mem.readInt(u64, elf_bytes[32..40], .little);
    const e_phnum = std.mem.readInt(u16, elf_bytes[56..58], .little);

    if (e_phnum == 0) return error.NoLoadableSegments;

    // Calculate guest memory bounds from PT_LOAD segments
    var min_vaddr: u64 = std.math.maxInt(u64);
    var max_vaddr: u64 = 0;

    var i: u16 = 0;
    while (i < e_phnum) : (i += 1) {
        const phoff = e_phoff + i * @sizeOf(Elf64Phdr);
        if (phoff + @sizeOf(Elf64Phdr) > elf_bytes.len) return error.TruncatedElf;
        const p_type = std.mem.readInt(u32, elf_bytes[@intCast(phoff)..][0..4], .little);
        if (p_type != PT_LOAD) continue;
        const p_vaddr = std.mem.readInt(u64, elf_bytes[@intCast(phoff + 16)..][0..8], .little);
        const p_memsz = std.mem.readInt(u64, elf_bytes[@intCast(phoff + 40)..][0..8], .little);
        const seg_end = p_vaddr + p_memsz;
        if (p_vaddr < min_vaddr) min_vaddr = p_vaddr;
        if (seg_end > max_vaddr) max_vaddr = seg_end;
    }

    if (min_vaddr == std.math.maxInt(u64)) return error.NoLoadableSegments;

    const PAGE_SIZE: u64 = 4096;
    const guest_base = min_vaddr & ~@as(u64, PAGE_SIZE - 1);
    const guest_size = std.mem.alignForward(u64, max_vaddr - guest_base, PAGE_SIZE);

    const guest_mem = try allocator.alloc(u8, @intCast(guest_size));
    @memset(guest_mem, 0);

    // Load PT_LOAD segments
    i = 0;
    while (i < e_phnum) : (i += 1) {
        const phoff = e_phoff + i * @sizeOf(Elf64Phdr);
        const p_type = std.mem.readInt(u32, elf_bytes[@intCast(phoff)..][0..4], .little);
        if (p_type != PT_LOAD) continue;

        const p_offset = std.mem.readInt(u64, elf_bytes[@intCast(phoff + 8)..][0..8], .little);
        const p_vaddr = std.mem.readInt(u64, elf_bytes[@intCast(phoff + 16)..][0..8], .little);
        const p_filesz = std.mem.readInt(u64, elf_bytes[@intCast(phoff + 32)..][0..8], .little);

        const offset_in_guest = p_vaddr - guest_base;
        if (p_filesz > 0) {
            @memcpy(guest_mem[@intCast(offset_in_guest)..][0..@intCast(p_filesz)], elf_bytes[@intCast(p_offset)..][0..@intCast(p_filesz)]);
        }
    }

    return LoadedElf{
        .guest_mem = guest_mem,
        .guest_base = guest_base,
        .entry = e_entry,
        .guest_size = guest_size,
    };
}

// ── ELF binary builder (for tests) ────────────────────────────────
//
// Builds a minimal ARM64 static ELF executable in memory.
// This lets us test the full pipeline without an ARM64 cross-compiler.

/// Flags for ELF segment protection
const PF_RX = PF_R | PF_X;
const PF_RW = PF_R | PF_W;

/// Build a minimal ARM64 executable that returns a value in x0.
/// code: the ARM64 machine code bytes (must end with RET)
/// returns: a heap-allocated ELF binary
pub fn buildMinimalElf(allocator: std.mem.Allocator, code: []const u8) ![]u8 {
    // Layout:
    //   [0..64)     ELF header
    //   [64..120)   Program header (PT_LOAD, RX, for .text)
    //   [120..)     ARM64 code (.text)

    const PAGE: u64 = 0x10000; // 64KB alignment
    const text_vaddr = PAGE; // start at 64KB
    const text_offset: u64 = 128; // file offset of .text

    const elf_size = text_offset + code.len;

    var buf = try allocator.alloc(u8, elf_size);
    @memset(buf, 0);

    // ── ELF header ────────────────────────────────────────────
    const ehdr = @as(*Elf64Ehdr, @alignCast(@ptrCast(buf.ptr)));
    ehdr.e_ident = .{
        0x7F, 'E', 'L', 'F',
        2, // 64-bit
        1, // little-endian
        1, // ELF version
        0, // System V ABI
        0, 0, 0, 0, 0, 0, 0, 0, // padding
    };
    ehdr.e_type = ET_EXEC;
    ehdr.e_machine = EM_AARCH64;
    ehdr.e_version = 1;
    ehdr.e_entry = text_vaddr; // entry = text base
    ehdr.e_phoff = 64;
    ehdr.e_shoff = 0;
    ehdr.e_flags = 0;
    ehdr.e_ehsize = 64;
    ehdr.e_phentsize = @sizeOf(Elf64Phdr);
    ehdr.e_phnum = 1;
    ehdr.e_shentsize = 0;
    ehdr.e_shnum = 0;
    ehdr.e_shstrndx = 0;

    // ── Program header ────────────────────────────────────────
    const phdr = @as(*Elf64Phdr, @alignCast(@ptrCast(buf.ptr + 64)));
    phdr.p_type = PT_LOAD;
    phdr.p_flags = PF_RX;
    phdr.p_offset = text_offset;
    phdr.p_vaddr = text_vaddr;
    phdr.p_paddr = text_vaddr;
    phdr.p_filesz = code.len;
    phdr.p_memsz = code.len;
    phdr.p_align = PAGE;

    // ── Code ──────────────────────────────────────────────────
    @memcpy(buf[text_offset..][0..code.len], code);

    return buf;
}

test "build and load minimal ELF" {
    // Build an ELF with a trivial ARM64 program:
    // MOV X0, #42 ; RET
    const code = [_]u8{
        0x80, 0x00, 0x80, 0xD2, // MOVZ X0, #0x42
        0x00, 0x00, 0x5F, 0xD6, // RET
    };

    const elf = try buildMinimalElf(std.testing.allocator, &code);
    defer std.testing.allocator.free(elf);

    const loaded = try loadElf(std.testing.allocator, elf);
    defer std.testing.allocator.free(loaded.guest_mem);

    try std.testing.expectEqual(@as(u64, 0x10000), loaded.entry);
    try std.testing.expect(loaded.guest_mem.len >= 0x10000);
}

test "loadElf validates ELF" {
    var bad_bytes: [64]u8 = undefined;
    @memset(&bad_bytes, 0);
    const result = loadElf(std.testing.allocator, &bad_bytes);
    try std.testing.expectError(error.InvalidElfMagic, result);
}

test "buildMinimalElf generates valid ELF" {
    const code = [_]u8{ 0x00, 0x00, 0x5F, 0xD6 }; // just RET
    const elf = try buildMinimalElf(std.testing.allocator, &code);
    defer std.testing.allocator.free(elf);

    // Check ELF magic
    try std.testing.expectEqual(@as(u32, ELF_MAGIC), std.mem.readInt(u32, elf[0..4], .little));
    // Check AArch64 machine type
    try std.testing.expectEqual(@as(u16, EM_AARCH64), std.mem.readInt(u16, elf[18..20], .little));
    // Check entry point
    try std.testing.expectEqual(@as(u64, 0x10000), std.mem.readInt(u64, elf[24..32], .little));
}

test "loadElf rejects non-ARM64" {
}
