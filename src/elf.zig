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
pub const PT_TLS = 7;
pub const PT_GNU_EH_FRAME = 0x6474e550;
pub const PT_GNU_STACK = 0x6474e551;
pub const PT_GNU_RELRO = 0x6474e552;
pub const PF_R = 4;
pub const PF_W = 2;
pub const PF_X = 1;

// ── Dynamic section tags ─────────────────────────────────────────
pub const DT_NULL = 0;
pub const DT_NEEDED = 1;
pub const DT_STRTAB = 5;
pub const DT_SYMTAB = 6;
pub const DT_STRSZ = 10;
pub const DT_GNU_HASH = 0x6ffffef5;
pub const DT_INIT = 12;
pub const DT_FINI = 13;
pub const DT_INIT_ARRAY = 25;
pub const DT_FINI_ARRAY = 26;
pub const DT_INIT_ARRAYSZ = 27;
pub const DT_FINI_ARRAYSZ = 28;
pub const DT_PLTGOT = 3;
pub const DT_PLTRELSZ = 2;
pub const DT_PLTREL = 20;
pub const DT_JMPREL = 23;
pub const DT_RELA = 7;
pub const DT_RELASZ = 8;
pub const DT_RELAENT = 9;
pub const DT_VERNEED = 0x6ffffffe;
pub const DT_VERNEEDNUM = 0x6fffffff;
pub const DT_VERSYM = 0x6ffffff0;

// ── Relocation types (R_AARCH64) ─────────────────────────────────
pub const R_AARCH64_NONE = 0;
pub const R_AARCH64_ABS64 = 257;
pub const R_AARCH64_GLOB_DAT = 1025;
pub const R_AARCH64_JUMP_SLOT = 1026;
pub const R_AARCH64_RELATIVE = 1027;

// ── Symbol bindings ──────────────────────────────────────────────
pub const STB_GLOBAL = 1;
pub const STB_WEAK = 2;

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

// ── Dynamic library loading ─────────────────────────────────────

pub const DynLib = struct {
    name: []const u8,
    guest_mem: []u8,
    guest_base: u64,
    guest_size: u64,
    entry: u64,
    symtab: u64,
    strtab: u64,
    strsz: u64,
    needed: std.ArrayListUnmanaged([]const u8) = .{ .items = &.{}, .capacity = 0 },
    init: u64,
    init_array: u64,
    init_arraysz: u64,
    fini: u64,
    fini_array: u64,
    fini_arraysz: u64,
    // Relocation info (guest vaddr)
    rela: u64,
    relasz: u64,
    jmprel: u64,
    pltrelsz: u64,
};

/// Load a dynamic library into guest memory and parse its metadata.
/// The library is loaded at a computed base address, avoiding overlap
/// with existing guest regions.
pub fn loadDynLib(allocator: std.mem.Allocator, elf_bytes: []const u8, name: []const u8, next_base: u64) !DynLib {
    const raw = try loadElf(allocator, elf_bytes);
    defer allocator.free(raw.guest_mem);

    // The .so file has segments at vaddr = raw.guest_base + offset.
    // At load_base, the same segments should be at load_base + offset.
    // We allocate guest_mem to cover load_base..load_base+raw.guest_base+raw.guest_size
    // and place raw data at offset raw.guest_base from the start.
    const load_base = next_base;
    const guest_pad = raw.guest_base; // bytes before first segment
    const guest_size = guest_pad + raw.guest_size;
    const guest_mem = try allocator.alloc(u8, @intCast(guest_size));
    @memset(guest_mem, 0);
    @memcpy(guest_mem[@intCast(guest_pad)..][0..raw.guest_mem.len], raw.guest_mem);

    // Apply R_AARCH64_RELATIVE: add load_base to original base = 0
    var fix_count: usize = 0;
    {
        const dyn = parseDynamic(elf_bytes, getPhdrInfo(elf_bytes).?.phoff, getPhdrInfo(elf_bytes).?.phnum);
        if (dyn) |d| {
            if (d.rela != 0 and d.relasz > 0) {
                const num_rela = @as(usize, @intCast(d.relasz / @sizeOf(Elf64Rela)));
                var ri: usize = 0;
                while (ri < num_rela) : (ri += 1) {
                    const rela_fileoff = loadVaddrToFileOffset(elf_bytes, getPhdrInfo(elf_bytes).?.phoff, getPhdrInfo(elf_bytes).?.phnum, d.rela + ri * @sizeOf(Elf64Rela)) orelse continue;
                    if (rela_fileoff + @sizeOf(Elf64Rela) > elf_bytes.len) continue;
                    const r_offset = std.mem.readInt(u64, elf_bytes[@intCast(rela_fileoff)..][0..8], .little);
                    const r_info = std.mem.readInt(u64, elf_bytes[@intCast(rela_fileoff + 8)..][0..8], .little);
                    const r_addend = std.mem.readInt(i64, elf_bytes[@intCast(rela_fileoff + 16)..][0..8], .little);
                    if (r_type(r_info) == R_AARCH64_RELATIVE) {
                        // r_offset is a vaddr relative to the original base (0 for .so).
                        // In guest_mem, data is at offset raw.guest_base from start.
                        // So the GOT entry is at r_offset in guest_mem.
                        if (r_offset + 8 <= guest_mem.len) {
                            const value = load_base + @as(u64, @bitCast(r_addend));
                            std.mem.writeInt(u64, guest_mem[@intCast(r_offset)..][0..8], value, .little);
                            fix_count += 1;
                        }
                    }
                }
            }
        }
    }

    // Parse dynamic section for symbol/string tables
    const e_phoff = getPhdrInfo(elf_bytes).?.phoff;
    const e_phnum = getPhdrInfo(elf_bytes).?.phnum;
    const dyn2 = parseDynamic(elf_bytes, e_phoff, e_phnum);

    var result = DynLib{
        .name = name,
        .guest_mem = guest_mem,
        .guest_base = load_base,
        .guest_size = guest_size,
        .entry = raw.entry + load_base, // .so entry is relative to base
        .symtab = 0,
        .strtab = 0,
        .strsz = 0,
        .needed = .{ .items = &.{}, .capacity = 0 },
        .init = 0,
        .init_array = 0,
        .init_arraysz = 0,
        .fini = 0,
        .fini_array = 0,
        .fini_arraysz = 0,
        .rela = 0,
        .relasz = 0,
        .jmprel = 0,
        .pltrelsz = 0,
    };

    if (dyn2) |d| {
        result.rela = if (d.rela != 0) load_base + d.rela else 0;
        result.relasz = d.relasz;
        result.jmprel = if (d.jmprel != 0) load_base + d.jmprel else 0;
        result.pltrelsz = d.pltrelsz;
        result.symtab = if (d.symtab != 0) load_base + d.symtab else 0;
        result.strtab = if (d.strtab != 0) load_base + d.strtab else 0;
        result.strsz = d.strsz;
        result.init = if (d.init != 0) load_base + d.init else 0;
        result.init_array = if (d.init_array != 0) load_base + d.init_array else 0;
        result.init_arraysz = d.init_arraysz;
        result.fini = if (d.fini != 0) load_base + d.fini else 0;
        result.fini_array = if (d.fini_array != 0) load_base + d.fini_array else 0;
        result.fini_arraysz = d.fini_arraysz;
    }

    // Extract DT_NEEDED library names
    const needed_names = try getNeededLibs(elf_bytes, e_phoff, e_phnum, allocator);
    result.needed = needed_names;

    return result;
}

const PhdrInfo = struct { phoff: u64, phnum: u16 };

fn getPhdrInfo(elf_bytes: []const u8) ?PhdrInfo {
    if (elf_bytes.len < 64) return null;
    const e_type = std.mem.readInt(u16, elf_bytes[16..18], .little);
    _ = e_type;
    return PhdrInfo{
        .phoff = std.mem.readInt(u64, elf_bytes[32..40], .little),
        .phnum = std.mem.readInt(u16, elf_bytes[56..58], .little),
    };
}

/// Find a symbol by name across all loaded libraries.
pub fn findGlobalSymbol(libs: []const DynLib, name: []const u8) ?u64 {
    for (libs) |lib| {
        if (lib.strtab == 0 or lib.symtab == 0) continue;

        // Walk the symbol table looking for a name match
        // (Heuristic: search first N entries)
        const max_sym: usize = 4096;
        var i: u64 = 1; // skip STN_UNDEF
        while (i < max_sym) : (i += 1) {
            const sym_guest = lib.symtab + i * @sizeOf(Elf64Sym);
            const mem_off = sym_guest - lib.guest_base;
            if (mem_off + 8 > lib.guest_mem.len) break;

            const st_name = std.mem.readInt(u32, lib.guest_mem[@intCast(mem_off)..][0..4], .little);
            const st_info = lib.guest_mem[@intCast(mem_off + 4)];
            _ = st_info;
            const st_value = std.mem.readInt(u64, lib.guest_mem[@intCast(mem_off + 8)..][0..8], .little);
            const st_shndx = std.mem.readInt(u16, lib.guest_mem[@intCast(mem_off + 6)..][0..2], .little);

            if (st_value == 0 or st_shndx == 0) continue;

            // Read the symbol name from string table
            const str_off = (lib.strtab + st_name) - lib.guest_base;
            if (str_off > lib.guest_mem.len) continue;
            const max_len = @min(@as(usize, 256), lib.guest_mem.len - @as(usize, @intCast(str_off)));
            const sym_name = lib.guest_mem[@intCast(str_off)..][0..max_len];
            const name_end = std.mem.indexOfScalar(u8, sym_name, 0) orelse max_len;
            const actual_name = sym_name[0..name_end];

            if (std.mem.eql(u8, actual_name, name)) {
                return lib.guest_base + st_value;
            }
        }
    }
    return null;
}

/// Apply cross-library relocations to a loaded library.
/// Reads RELA entries from guest memory, resolves symbols across all_libs,
/// and writes the resolved addresses to GOT entries.
pub fn resolveLibrary(lib: *DynLib, all_libs: []const DynLib, elf_bytes: []const u8) void {
    const p_off2 = getPhdrInfo(elf_bytes) orelse return;
    const dyn = parseDynamic(elf_bytes, p_off2.phoff, p_off2.phnum) orelse return;
    const guest = lib.guest_mem;
    const base = lib.guest_base;

    // Helper: read relocation entries from guest memory
    const readRela = struct {
        fn read(mem: []const u8, b: u64, vaddr: u64, idx: usize) ?struct { u64, u64, i64 } {
            const addr = vaddr + idx * @sizeOf(Elf64Rela);
            if (addr < b) return null;
            const off = addr - b;
            if (off + @sizeOf(Elf64Rela) > mem.len) return null;
            const r_off = std.mem.readInt(u64, mem[@intCast(off)..][0..8], .little);
            const r_info = std.mem.readInt(u64, mem[@intCast(off + 8)..][0..8], .little);
            const r_add = std.mem.readInt(i64, mem[@intCast(off + 16)..][0..8], .little);
            return .{ r_off, r_info, r_add };
        }
    }.read;

    // Process RELA (non-PLT relocations)
    if (dyn.rela != 0 and dyn.relasz > 0) {
        const rela_guest = base + dyn.rela;
        const num = @as(usize, @intCast(dyn.relasz / @sizeOf(Elf64Rela)));
        var i: usize = 0;
        while (i < num) : (i += 1) {
            if (readRela(guest, base, rela_guest, i)) |rela| {
                const r_off, const r_info, const r_add = rela;
                const rt = r_type(r_info);
                if (rt == R_AARCH64_GLOB_DAT) {
                    const sym_idx = r_sym(r_info);
                    const sym_name = getSymbolName(guest, base, lib.symtab, lib.strtab, sym_idx) orelse continue;
                    const sym_val = findGlobalSymbol(all_libs, sym_name) orelse continue;
                    const target_off = r_off - base;
                    if (target_off + 8 <= guest.len) {
                        std.mem.writeInt(u64, guest[@intCast(target_off)..][0..8], sym_val + @as(u64, @bitCast(r_add)), .little);
                    }
                }
            }
        }
    }

    // Process JMPREL (PLT relocations)
    if (dyn.jmprel != 0 and dyn.pltrelsz > 0) {
        const jmprel_guest = base + dyn.jmprel;
        const num = @as(usize, @intCast(dyn.pltrelsz / @sizeOf(Elf64Rela)));
        var i: usize = 0;
        while (i < num) : (i += 1) {
            if (readRela(guest, base, jmprel_guest, i)) |rela| {
                const r_off, const r_info, const r_add = rela;
                const rt = r_type(r_info);
                if (rt == R_AARCH64_JUMP_SLOT) {
                    const sym_idx = r_sym(r_info);
                    const sym_name = getSymbolName(guest, base, lib.symtab, lib.strtab, sym_idx) orelse continue;
                    const sym_val = findGlobalSymbol(all_libs, sym_name) orelse continue;
                    const target_off = r_off - base;
                    if (target_off + 8 <= guest.len) {
                        std.mem.writeInt(u64, guest[@intCast(target_off)..][0..8], sym_val + @as(u64, @bitCast(r_add)), .little);
                    }
                }
            }
        }
    }
}

/// Extract DT_NEEDED library names from the .dynamic section.
/// Returns a list of library names (just the basenames, like "libc.so").
pub fn getNeededLibs(elf_bytes: []const u8, e_phoff: u64, e_phnum: u16, allocator: std.mem.Allocator) !std.ArrayListUnmanaged([]const u8) {
    var result: std.ArrayListUnmanaged([]const u8) = .{ .items = &.{}, .capacity = 0 };
    const dyn = parseDynamic(elf_bytes, e_phoff, e_phnum) orelse return result;

    // Find the PT_LOAD containing .dynamic to get strtab and needed entries
    // .dynamic entries are read from the file. DT_NEEDED values are indices into DT_STRTAB.
    // We need DT_STRTAB address to resolve names.
    var strtab_vaddr: u64 = 0;
    var strtab_fileoff: u64 = 0;
    var strtab_load_vaddr: u64 = 0;
    var strtab_load_fileoff: u64 = 0;
    var i: u16 = 0;
    while (i < e_phnum) : (i += 1) {
        const phoff = e_phoff + i * @sizeOf(Elf64Phdr);
        const p_type = std.mem.readInt(u32, elf_bytes[@intCast(phoff)..][0..4], .little);
        if (p_type != PT_LOAD) continue;
        const p_vaddr = std.mem.readInt(u64, elf_bytes[@intCast(phoff + 16)..][0..8], .little);
        const p_off2 = std.mem.readInt(u64, elf_bytes[@intCast(phoff + 8)..][0..8], .little);
        const p_memsz = std.mem.readInt(u64, elf_bytes[@intCast(phoff + 40)..][0..8], .little);
        if (dyn.strtab != 0 and dyn.strtab >= p_vaddr and dyn.strtab < p_vaddr + p_memsz) {
            strtab_vaddr = dyn.strtab;
            strtab_fileoff = p_off2 + (dyn.strtab - p_vaddr);
            strtab_load_vaddr = p_vaddr;
            strtab_load_fileoff = p_off2;
        }
    }
    if (strtab_vaddr == 0) return result;

    // Find the PT_LOAD containing .dynamic
    var dyn_fileoff: u64 = 0;
    var dyn_load_vaddr: u64 = 0;
    i = 0;
    while (i < e_phnum) : (i += 1) {
        const phoff = e_phoff + i * @sizeOf(Elf64Phdr);
        const p_type = std.mem.readInt(u32, elf_bytes[@intCast(phoff)..][0..4], .little);
        if (p_type != PT_LOAD) continue;
        const p_vaddr = std.mem.readInt(u64, elf_bytes[@intCast(phoff + 16)..][0..8], .little);
        const p_off2 = std.mem.readInt(u64, elf_bytes[@intCast(phoff + 8)..][0..8], .little);
        const p_memsz = std.mem.readInt(u64, elf_bytes[@intCast(phoff + 40)..][0..8], .little);
        if (dyn.rela != 0 and dyn.rela >= p_vaddr and dyn.rela < p_vaddr + p_memsz) {
            dyn_fileoff = p_off2 + (dyn.rela - p_vaddr);
            dyn_load_vaddr = p_vaddr;
            break;
        }
    }
    if (dyn_fileoff == 0) return result;

    var dyn_offset: usize = @intCast(dyn_fileoff);
    while (dyn_offset + @sizeOf(Elf64Dyn) <= elf_bytes.len) {
        const d_tag = std.mem.readInt(i64, elf_bytes[dyn_offset..][0..8], .little);
        const d_val = std.mem.readInt(u64, elf_bytes[dyn_offset + 8..][0..8], .little);
        if (d_tag == DT_NULL) break;
        if (d_tag == DT_NEEDED) {
            // Read the library name from the string table
            const str_off = strtab_fileoff + d_val;
            if (str_off < elf_bytes.len) {
                const remaining = elf_bytes.len - @as(usize, @intCast(str_off));
                const max_n = @min(remaining, @as(usize, 256));
                const name_slice = elf_bytes[@intCast(str_off)..][0..max_n];
                const name_end = std.mem.indexOfScalar(u8, name_slice, 0) orelse max_n;
                const name = try allocator.dupe(u8, name_slice[0..name_end]);
                try result.append(allocator, name);
            }
        }
        dyn_offset += @sizeOf(Elf64Dyn);
    }

    return result;
}

/// Get the name of a symbol by its index.
pub fn getSymbolName(guest: []const u8, base: u64, symtab: u64, strtab: u64, sym_idx: u64) ?[]const u8 {
    if (sym_idx == 0 or symtab == 0 or strtab == 0) return null;
    const sym_guest = symtab + sym_idx * @sizeOf(Elf64Sym);
    if (sym_guest < base) return null;
    const off = sym_guest - base;
    if (off + 8 > guest.len) return null;
    const st_name = std.mem.readInt(u32, guest[@intCast(off)..][0..4], .little);
    const str_guest = strtab + st_name;
    if (str_guest < base) return null;
    const soff = str_guest - base;
    if (soff >= guest.len) return null;
    const max_len = @min(@as(usize, 256), guest.len - @as(usize, @intCast(soff)));
    const slice = guest[@intCast(soff)..][0..max_len];
    const end = std.mem.indexOfScalar(u8, slice, 0) orelse max_len;
    return slice[0..end];
}

// ── Dynamic ELF structures ───────────────────────────────────────

pub const Elf64Dyn = extern struct {
    d_tag: i64,
    d_val: u64,
};

pub const Elf64Sym = extern struct {
    st_name: u32,
    st_info: u8,
    st_other: u8,
    st_shndx: u16,
    st_value: u64,
    st_size: u64,
};

pub const Elf64Rela = extern struct {
    r_offset: u64,
    r_info: u64,
    r_addend: i64,
};

pub fn r_type(r_info: u64) u64 {
    return r_info & 0xFFFFFFFF;
}

pub fn r_sym(r_info: u64) u64 {
    return r_info >> 32;
}

// ── Dynamic linker ───────────────────────────────────────────────

pub const DynResult = struct {
    // Relocation info
    rela: u64,        // DT_RELA address
    relasz: u64,      // DT_RELASZ
    jmprel: u64,      // DT_JMPREL
    pltrelsz: u64,    // DT_PLTRELSZ
    pltrel: u64,      // DT_PLTREL (0=RELA, 1=REL)
    // Symbol tables
    symtab: u64,      // DT_SYMTAB address
    strtab: u64,      // DT_STRTAB address
    strsz: u64,       // DT_STRSZ
    // Init/fini
    init: u64,
    init_array: u64,
    init_arraysz: u64,
    fini: u64,
    fini_array: u64,
    fini_arraysz: u64,
    // Loaded dependencies
    needed: std.ArrayListUnmanaged([]u8) = .{ .items = &.{}, .capacity = 0 },
};

/// Parse the .dynamic section and extract all relevant entries.
pub fn parseDynamic(elf_bytes: []const u8, e_phoff: u64, e_phnum: u16) ?DynResult {
    var result = DynResult{
        .rela = 0, .relasz = 0, .jmprel = 0, .pltrelsz = 0, .pltrel = 0,
        .symtab = 0, .strtab = 0, .strsz = 0,
        .init = 0, .init_array = 0, .init_arraysz = 0,
        .fini = 0, .fini_array = 0, .fini_arraysz = 0,
        .needed = .{ .items = &.{}, .capacity = 0 },
    };

    // Find the PT_DYNAMIC segment
    var dyn_vaddr: u64 = 0;
    var dyn_size: u64 = 0;
    var i: u16 = 0;
    while (i < e_phnum) : (i += 1) {
        const phoff = e_phoff + i * @sizeOf(Elf64Phdr);
        const p_type = std.mem.readInt(u32, elf_bytes[@intCast(phoff)..][0..4], .little);
        if (p_type == PT_DYNAMIC) {
            dyn_vaddr = std.mem.readInt(u64, elf_bytes[@intCast(phoff + 16)..][0..8], .little);
            dyn_size = std.mem.readInt(u64, elf_bytes[@intCast(phoff + 40)..][0..8], .little);
            break;
        }
    }
    if (dyn_vaddr == 0) return null;

    // Walk the .dynamic entries
    // For the MVP, we read .dynamic from the ELF file directly.
    // In production, .dynamic is in loaded guest memory.
    // Since guest_base ≈ file offset, read from file at matching offset.
    // This works for simple cases where .dynamic is in a PT_LOAD segment.

    // Find the PT_LOAD that contains .dynamic to get file offset
    var load_fileoff: u64 = 0;
    var load_vaddr: u64 = 0;
    i = 0;
    while (i < e_phnum) : (i += 1) {
        const phoff = e_phoff + i * @sizeOf(Elf64Phdr);
        const p_type = std.mem.readInt(u32, elf_bytes[@intCast(phoff)..][0..4], .little);
        if (p_type == PT_LOAD) {
            const p_vaddr = std.mem.readInt(u64, elf_bytes[@intCast(phoff + 16)..][0..8], .little);
            const p_offset2 = std.mem.readInt(u64, elf_bytes[@intCast(phoff + 8)..][0..8], .little);
            const p_memsz = std.mem.readInt(u64, elf_bytes[@intCast(phoff + 40)..][0..8], .little);
            if (dyn_vaddr >= p_vaddr and dyn_vaddr < p_vaddr + p_memsz) {
                load_fileoff = p_offset2;
                load_vaddr = p_vaddr;
                break;
            }
        }
    }
    if (load_fileoff == 0) return null;

    const dyn_fileoff = load_fileoff + (dyn_vaddr - load_vaddr);
    const max_entries = @as(usize, @intCast(dyn_size / @sizeOf(Elf64Dyn)));
    if (max_entries == 0) return null;

    var ents: [128]Elf64Dyn = undefined;
    const actual_entries = @min(max_entries, ents.len);
    @memcpy(std.mem.sliceAsBytes(ents[0..actual_entries]), elf_bytes[@intCast(dyn_fileoff)..][0..actual_entries * @sizeOf(Elf64Dyn)]);

    for (ents[0..actual_entries]) |entry| {
        switch (entry.d_tag) {
            DT_NULL => break,
            DT_NEEDED => {},
            DT_RELA => result.rela = entry.d_val,
            DT_RELASZ => result.relasz = entry.d_val,
            DT_JMPREL => result.jmprel = entry.d_val,
            DT_PLTRELSZ => result.pltrelsz = entry.d_val,
            DT_PLTREL => result.pltrel = @intCast(entry.d_val),
            DT_SYMTAB => result.symtab = entry.d_val,
            DT_STRTAB => result.strtab = entry.d_val,
            DT_STRSZ => result.strsz = entry.d_val,
            DT_INIT => result.init = entry.d_val,
            DT_FINI => result.fini = entry.d_val,
            DT_INIT_ARRAY => result.init_array = entry.d_val,
            DT_INIT_ARRAYSZ => result.init_arraysz = entry.d_val,
            DT_FINI_ARRAY => result.fini_array = entry.d_val,
            DT_FINI_ARRAYSZ => result.fini_arraysz = entry.d_val,
            else => {},
        }
    }
    return result;
}

/// Apply relocations in guest memory.
/// guest: the loaded guest memory
/// guest_base: base address of the loaded ELF
/// elf_bytes: the original ELF file bytes
/// returns: number of relocations applied
pub fn applyRelocations(guest: []u8, guest_base: u64, elf_bytes: []const u8, e_phoff: u64, e_phnum: u16) !usize {
    const dyn = parseDynamic(elf_bytes, e_phoff, e_phnum) orelse return 0;
    var count: usize = 0;

    // Apply RELA relocations (non-PLT)
    if (dyn.rela != 0 and dyn.relasz > 0) {
        const num_rela = @as(usize, @intCast(dyn.relasz / @sizeOf(Elf64Rela)));
        var i: usize = 0;
        while (i < num_rela) : (i += 1) {
            const rela_off = loadVaddrToFileOffset(elf_bytes, e_phoff, e_phnum, dyn.rela + i * @sizeOf(Elf64Rela)) orelse continue;
            if (rela_off + @sizeOf(Elf64Rela) > elf_bytes.len) continue;

            const r_offset = std.mem.readInt(u64, elf_bytes[@intCast(rela_off)..][0..8], .little);
            const r_info = std.mem.readInt(u64, elf_bytes[@intCast(rela_off + 8)..][0..8], .little);
            const r_addend = std.mem.readInt(i64, elf_bytes[@intCast(rela_off + 16)..][0..8], .little);
            const r_type2 = r_type(r_info);

            if (r_type2 == R_AARCH64_RELATIVE) {
                // R_AARCH64_RELATIVE: *r_offset = guest_base + addend
                const target_off = r_offset - guest_base;
                if (target_off + 8 <= guest.len) {
                    const value = guest_base + @as(u64, @bitCast(r_addend));
                    std.mem.writeInt(u64, guest[@intCast(target_off)..][0..8], value, .little);
                    count += 1;
                }
            } else if (r_type2 == R_AARCH64_GLOB_DAT) {
                // R_AARCH64_GLOB_DAT: *r_offset = symbol value
                // For now, resolve as relative (workaround for simple cases)
                const sym_idx = r_sym(r_info);
                const sym_val = findSymbolValue(elf_bytes, e_phoff, e_phnum, dyn.symtab, dyn.strtab, dyn.strsz, sym_idx) orelse 0;
                const target_off = r_offset - guest_base;
                if (target_off + 8 <= guest.len) {
                    const value = if (sym_val != 0) guest_base + sym_val else guest_base;
                    std.mem.writeInt(u64, guest[@intCast(target_off)..][0..8], value, .little);
                    count += 1;
                }
            } else if (r_type2 == R_AARCH64_JUMP_SLOT) {
                // R_AARCH64_JUMP_SLOT: PLT entry – set to zero (will be lazily resolved)
                const target_off = r_offset - guest_base;
                if (target_off + 8 <= guest.len) {
                    const sym_idx = r_sym(r_info);
                    const sym_val = findSymbolValue(elf_bytes, e_phoff, e_phnum, dyn.symtab, dyn.strtab, dyn.strsz, sym_idx) orelse 0;
                    const value = if (sym_val != 0) guest_base + sym_val else 0;
                    std.mem.writeInt(u64, guest[@intCast(target_off)..][0..8], value, .little);
                    count += 1;
                }
            }
        }
    }
    return count;
}

/// Find the file offset that corresponds to a given virtual address.
fn loadVaddrToFileOffset(elf_bytes: []const u8, e_phoff: u64, e_phnum: u16, vaddr: u64) ?u64 {
    var i: u16 = 0;
    while (i < e_phnum) : (i += 1) {
        const phoff = e_phoff + i * @sizeOf(Elf64Phdr);
        const p_type = std.mem.readInt(u32, elf_bytes[@intCast(phoff)..][0..4], .little);
        if (p_type != PT_LOAD) continue;
        const p_offset = std.mem.readInt(u64, elf_bytes[@intCast(phoff + 8)..][0..8], .little);
        const p_vaddr = std.mem.readInt(u64, elf_bytes[@intCast(phoff + 16)..][0..8], .little);
        const p_memsz = std.mem.readInt(u64, elf_bytes[@intCast(phoff + 40)..][0..8], .little);
        if (vaddr >= p_vaddr and vaddr < p_vaddr + p_memsz) {
            return p_offset + (vaddr - p_vaddr);
        }
    }
    return null;
}

/// Find the value (address) of a symbol by index.
fn findSymbolValue(elf_bytes: []const u8, e_phoff: u64, e_phnum: u16, symtab_vaddr: u64, strtab_vaddr: u64, strsz: u64, sym_idx: u64) ?u64 {
    if (sym_idx == 0) return null; // STN_UNDEF
    const sym_off = loadVaddrToFileOffset(elf_bytes, e_phoff, e_phnum, symtab_vaddr + sym_idx * @sizeOf(Elf64Sym)) orelse return null;
    if (sym_off + @sizeOf(Elf64Sym) > elf_bytes.len) return null;

    const st_name = std.mem.readInt(u32, elf_bytes[@intCast(sym_off)..][0..4], .little);
    const st_value = std.mem.readInt(u64, elf_bytes[@intCast(sym_off + 8)..][0..8], .little); // skip st_info + st_other + st_shndx
    const st_shndx = std.mem.readInt(u16, elf_bytes[@intCast(sym_off + 6)..][0..2], .little);
    _ = st_shndx;

    // For defined symbols (st_value != 0), return the value
    if (st_value != 0) return st_value;
    _ = strtab_vaddr;
    _ = strsz;
    _ = st_name;
    return null;
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

test "parseDynamic returns null for static ELF" {
    const code = [_]u8{ 0x00, 0x00, 0x5F, 0xD6 };
    const elf_bytes = try buildMinimalElf(std.testing.allocator, &code);
    defer std.testing.allocator.free(elf_bytes);
    const e_phoff = std.mem.readInt(u64, elf_bytes[32..40], .little);
    const e_phnum = std.mem.readInt(u16, elf_bytes[56..58], .little);
    const dyn = parseDynamic(elf_bytes, e_phoff, e_phnum);
    try std.testing.expect(dyn == null);
}

test "loadElf rejects non-ARM64" {
}
