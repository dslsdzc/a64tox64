//! GDB JIT debug interface.
//!
//! Synthesizes ELF + DWARF debug info for translated blocks and registers
//! them with GDB via the JIT interface. Each block appears as a function
//! "guest_0x{PC}" with guest PC as DWARF line number.
//!
//! GDB capabilities:
//!   info functions          → shows all translated blocks
//!   break *0x{guest_pc}     → break at guest address
//!   disas guest_0x{guest_pc} → disassemble translated code
//!   stepi                   → shows guest PC as source line
//!
//! Architecture: batches of 16 blocks are synthesized into one ELF .o
//! with DWARF debug info and registered via __jit_debug_register_code().

const std = @import("std");

pub const BATCH_SIZE = 16;

var g_sym_name_buf: [64]u8 = undefined;
var g_producer_buf: [32]u8 = undefined;
var g_filename_buf: [32]u8 = undefined;

// ── GDB JIT interface types ──────────────────────────────────

const JitDescriptor = extern struct {
    version: u32,
    action: u32,
    relevant_entry: ?*JitCodeEntry,
    first_entry: ?*JitCodeEntry,
};

const JitCodeEntry = extern struct {
    next: ?*JitCodeEntry,
    prev: ?*JitCodeEntry,
    symbol_name: ?[*:0]const u8,
    filename: ?[*:0]const u8,
    code_addr: u64,
    code_size: u64,
};

var g_descriptor: JitDescriptor = .{
    .version = 1,
    .action = 0,
    .relevant_entry = null,
    .first_entry = null,
};

export fn __jit_debug_register_code() void {}

/// A block entry in a batch.
const BlockEntry = struct {
    host_addr: u64,
    code_size: usize,
    guest_pc: u64,
};

/// A batch of blocks sharing one synthesized ELF.
const Batch = struct {
    entries: [BATCH_SIZE]BlockEntry = undefined,
    count: usize = 0,
    elf_data: ?[]u8 = null,
    jit_entry: ?*JitCodeEntry = null,
};

var batches: [64]Batch = undefined;
var batch_count: usize = 0;

pub fn init() void {
    g_descriptor = .{
        .version = 1,
        .action = 0,
        .relevant_entry = null,
        .first_entry = null,
    };
    batch_count = 0;
}

pub fn addBlock(host_addr: u64, code_size: usize, guest_pc: u64) void {
    var bi: usize = 0;
    while (bi < batch_count) {
        if (batches[bi].count < BATCH_SIZE) break;
        bi += 1;
    }
    if (bi >= batch_count) {
        if (batch_count >= batches.len) return;
        batches[bi] = .{};
        batch_count += 1;
    }
    const batch = &batches[bi];
    const idx = batch.count;
    batch.count += 1;
    batch.entries[idx] = .{
        .host_addr = host_addr,
        .code_size = code_size,
        .guest_pc = guest_pc,
    };
    if (batch.count >= BATCH_SIZE) commitBatch(batch);
}

fn commitBatch(batch: *Batch) void {
    const data = synthesizeElf(batch.entries[0..batch.count]) orelse return;
    batch.elf_data = data;

    const entry = std.heap.page_allocator.create(JitCodeEntry) catch return;
    entry.* = .{
        .next = g_descriptor.first_entry,
        .prev = null,
        .symbol_name = "a64tox64_jit",
        .filename = "a64tox64",
        .code_addr = @intFromPtr(data.ptr),
        .code_size = data.len,
    };
    if (g_descriptor.first_entry) |f| f.prev = entry;
    g_descriptor.first_entry = entry;
    g_descriptor.relevant_entry = entry;
    g_descriptor.action = 0;
    __jit_debug_register_code();
    batch.jit_entry = entry;
}

pub fn removeBatch(batch: *Batch) void {
    const e = batch.jit_entry orelse return;
    if (e.prev) |p| p.next = e.next;
    if (e.next) |n| n.prev = e.prev;
    if (g_descriptor.first_entry == e) g_descriptor.first_entry = e.next;
    g_descriptor.relevant_entry = e;
    g_descriptor.action = 1;
    __jit_debug_register_code();
    batch.jit_entry = null;
    if (batch.elf_data) |d| std.heap.page_allocator.free(d);
    batch.elf_data = null;
}

// ── ELF + DWARF synthesis ────────────────────────────────────

const Elf64Ehdr = extern struct {
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

const Elf64Shdr = extern struct {
    sh_name: u32,
    sh_type: u32,
    sh_flags: u64,
    sh_addr: u64,
    sh_offset: u64,
    sh_size: u64,
    sh_link: u32,
    sh_info: u32,
    sh_addralign: u64,
    sh_entsize: u64,
};

fn uleb128(buf: []u8, val: u64) usize {
    var v = val;
    var i: usize = 0;
    while (v >= 0x80) {
        buf[i] = @as(u8, @truncate(v & 0x7F)) | 0x80;
        v >>= 7;
        i += 1;
    }
    buf[i] = @as(u8, @truncate(v));
    return i + 1;
}

fn synthStr(buf: []u8, s: []const u8) usize {
    @memcpy(buf[0..s.len], s);
    return s.len;
}

fn pad4(offset: usize) usize {
    return (offset + 3) & ~@as(usize, 3);
}

fn synthesizeElf(blocks: []const BlockEntry) ?[]u8 {
    const n = blocks.len;
    // Section sizes (estimated)
    const sz_abbrev: usize = 64;
    const sz_info: usize = 20 + n * 28;
    const sz_line: usize = 50 + n * 20;
    const sz_strtab: usize = 80;
    const num_shdr: u16 = 5;
    const shdr_size = num_shdr * @sizeOf(Elf64Shdr);
    const total = 64 + sz_abbrev + sz_info + sz_line + sz_strtab + shdr_size;

    const buf = std.heap.page_allocator.alloc(u8, total) catch return null;
    @memset(buf, 0);

    // ── Compute section offsets ─────────────────────────────
    var off: usize = 64;
    const s_abbrev_off = off;
    const s_abbrev_sz = sz_abbrev;
    off += sz_abbrev;
    const s_info_off = off;
    const s_info_sz = sz_info;
    off += sz_info;
    const s_line_off = off;
    const s_line_sz = sz_line;
    off += sz_line;
    const s_str_off = off;
    const s_str_sz = sz_strtab;
    off += sz_strtab;
    const s_shdr_off = off;

    // ── ELF header ──────────────────────────────────────────
    const ehdr = @as(*Elf64Ehdr, @ptrCast(@alignCast(buf.ptr)));
    ehdr.e_ident = .{ 0x7F, 'E', 'L', 'F', 2, 1, 1, 0, 0,0,0,0,0,0,0,0 };
    ehdr.e_type = 0;
    ehdr.e_machine = 0x3E;
    ehdr.e_version = 1;
    ehdr.e_ehsize = 64;
    ehdr.e_shentsize = @sizeOf(Elf64Shdr);
    ehdr.e_shnum = num_shdr;
    ehdr.e_shstrndx = 4;
    ehdr.e_shoff = s_shdr_off;

    // ── Section 1: .debug_abbrev ────────────────────────────
    var p = s_abbrev_off;
    // Abbrev 1: DW_TAG_compile_unit, DW_CHILDREN_yes
    buf[p+0] = 1; p+=1; // number
    buf[p+0] = 0x11; p+=1; // DW_TAG_compile_unit
    buf[p+0] = 1; p+=1; // DW_CHILDREN_yes
    buf[p+0] = 0x25; p+=1; // DW_AT_producer
    buf[p+0] = 0x08; p+=1; // DW_FORM_string
    buf[p+0] = 0x13; p+=1; // DW_AT_language
    buf[p+0] = 0x0B; p+=1; // DW_FORM_data1
    buf[p+0] = 0x03; p+=1; // DW_AT_name
    buf[p+0] = 0x08; p+=1; // DW_FORM_string
    buf[p+0] = 0; p+=1; buf[p+0] = 0; p+=1; // terminator

    // Abbrev 2: DW_TAG_subprogram, DW_CHILDREN_no
    buf[p+0] = 2; p+=1;
    buf[p+0] = 0x2E; p+=1; // DW_TAG_subprogram
    buf[p+0] = 0; p+=1; // DW_CHILDREN_no
    buf[p+0] = 0x03; p+=1; // DW_AT_name
    buf[p+0] = 0x08; p+=1; // DW_FORM_string
    buf[p+0] = 0x11; p+=1; // DW_AT_low_pc
    buf[p+0] = 0x01; p+=1; // DW_FORM_addr
    buf[p+0] = 0x12; p+=1; // DW_AT_high_pc
    buf[p+0] = 0x01; p+=1; // DW_FORM_data8 (as offset)
    buf[p+0] = 0; p+=1; buf[p+0] = 0; p+=1; // terminator
    buf[p+0] = 0; p+=1; buf[p+0] = 0; p+=1; // end of table

    // ── Section 2: .debug_info ──────────────────────────────
    p = s_info_off;
    // CU header
    const unit_len_off = p;
    p += 4; // unit_length (patch later)
    std.mem.writeInt(u16, buf[p..][0..2], 5, .little); p += 2; // DWARF version 5... use 4
    // Actually let me use DWARF 2/3/4 which is more compatible
    p = s_info_off;
    p += 4; // unit_length placeholder
    std.mem.writeInt(u16, buf[p..][0..2], 4, .little); p += 2; // DWARF version 4
    std.mem.writeInt(u32, buf[p..][0..4], 0, .little); p += 4; // debug_abbrev offset
    buf[p+0] = 8; p+=1; // address_size

    // CU: abbrev 1
    buf[p+0] = 1; p+=1; // abbrev number
    // producer: "a64tox64"
    @memcpy(buf[p..][0..8], "a64tox64");
    p += 9; // includes null
    // language: DW_LANG_C89 = 0x2
    buf[p+0] = 2; p+=1;
    // name: "guest_code"
    @memcpy(buf[p..][0..10], "guest_code");
    p += 12; // includes null

    // Subprogram entries (abbrev 2)
    var i: usize = 0;
    while (i < n) : (i += 1) {
        const b = &blocks[i];
        buf[p+0] = 2; p+=1; // abbrev number
        // name
        const sname = std.fmt.bufPrint(&g_sym_name_buf, "guest_0x{X}", .{b.guest_pc}) catch "err";
        @memcpy(buf[p..][0..sname.len], sname);
        p += sname.len + 1; // includes null
        // low_pc
        std.mem.writeInt(u64, buf[p..][0..8], b.host_addr, .little); p += 8;
        // high_pc (as offset from low_pc, not absolute)
        std.mem.writeInt(u64, buf[p..][0..8], b.code_size, .little); p += 8;
    }
    // Patch unit_length
    const unit_len: u32 = @intCast(p - s_info_off - 4);
    std.mem.writeInt(u32, buf[unit_len_off..][0..4], unit_len, .little);

    // ── Section 3: .debug_line ──────────────────────────────
    p = s_line_off;
    const line_start = p;
    p += 4; // unit_length placeholder
    std.mem.writeInt(u16, buf[p..][0..2], 4, .little); p += 2; // DWARF version 4
    const prologue_len_off = p;
    p += 4; // prologue_length placeholder
    const prologue_start = p;
    buf[p+0] = 1; p+=1; // minimum_instruction_length
    buf[p+0] = 1; p+=1; // default_is_stmt
    buf[p+0] = 0; p+=1; // line_base (signed)
    buf[p+0] = 5; p+=1; // line_range
    buf[p+0] = 2 + 1 + 1 + 0; p+=1; // opcode_base (DW_LNS_copy=1, DW_LNS_advance_pc=2, DW_LNS_advance_line=3, etc.)
    // Standard opcode lengths for opcodes 1..opcode_base-1
    buf[p+0] = 0; p+=1; // DW_LNS_copy: 0 args
    buf[p+0] = 1; p+=1; // DW_LNS_advance_pc: 1 arg
    buf[p+0] = 1; p+=1; // DW_LNS_advance_line: 1 arg
    buf[p+0] = 1; p+=1; // DW_LNS_set_file: 1 arg
    buf[p+0] = 1; p+=1; // DW_LNS_set_column: 1 arg
    // Include directories
    buf[p+0] = 0; p+=1; // end of directories
    // File names
    _ = synthStr(buf[p..], "guest"); p += 5; // includes null
    // dir_index
    buf[p+0] = 0; p+=1;
    // time, size
    std.mem.writeInt(u32, buf[p..][0..4], 0, .little); p += 4;
    std.mem.writeInt(u32, buf[p..][0..4], 0, .little); p += 4;
    buf[p+0] = 0; p+=1; // end of file names

    // Patch prologue_length
    const pl_len: u32 = @intCast(p - prologue_start);
    std.mem.writeInt(u32, buf[prologue_len_off..][0..4], pl_len, .little);

    // Line number program
    i = 0;
    while (i < n) : (i += 1) {
        const b = &blocks[i];
        // DW_LNE_set_address
        buf[p+0] = 0; p+=1; // extended opcode
        buf[p+0] = 9; p+=1; // length (1 + 8)
        buf[p+0] = 2; p+=1; // DW_LNE_set_address
        std.mem.writeInt(u64, buf[p..][0..8], b.host_addr, .little); p += 8;
        // DW_LNS_set_line
        buf[p+0] = 3; p+=1; // DW_LNS_set_line
        var lbuf: [10]u8 = undefined;
        const lsz = uleb128(&lbuf, b.guest_pc);
        @memcpy(buf[p..][0..lsz], lbuf[0..lsz]); p += lsz;
        // DW_LNS_copy
        buf[p+0] = 1; p+=1; // DW_LNS_copy
    }
    // End sequence
    buf[p+0] = 0; p+=1;
    buf[p+0] = 1; p+=1; // DW_LNE_end_sequence, length=1
    buf[p+0] = 1; p+=1; // opcode

    // Patch unit_length for line
    const line_len: u32 = @intCast(p - line_start - 4);
    std.mem.writeInt(u32, buf[line_start..][0..4], line_len, .little);

    // ── Section 4: .shstrtab ────────────────────────────────
    p = s_str_off;
    buf[p+0] = 0; p+=1;
    const str_debug_abbrev = p;
    @memcpy(buf[p..][0..13], ".debug_abbrev"); p += 14; buf[p+0] = 0; p+=1;
    const str_debug_info = p;
    @memcpy(buf[p..][0..11], ".debug_info"); p += 11; buf[p+0] = 0; p+=1;
    const str_debug_line = p;
    @memcpy(buf[p..][0..11], ".debug_line"); p += 11; buf[p+0] = 0; p+=1;
    const str_shstrtab = p;
    @memcpy(buf[p..][0..9], ".shstrtab"); p += 9; buf[p+0] = 0; p+=1;

    // ── Section headers ─────────────────────────────────────
    p = s_shdr_off;
    // SHT_NULL
    var sh = @as(*Elf64Shdr, @ptrCast(@alignCast(buf[p..]))); p += @sizeOf(Elf64Shdr);
    sh.* = .{ .sh_name = 0, .sh_type = 0, .sh_flags = 0, .sh_addr = 0, .sh_offset = 0, .sh_size = 0, .sh_link = 0, .sh_info = 0, .sh_addralign = 0, .sh_entsize = 0 };

    // .debug_abbrev
    sh = @as(*Elf64Shdr, @ptrCast(@alignCast(buf[p..]))); p += @sizeOf(Elf64Shdr);
    sh.* = .{ .sh_name = @intCast(str_debug_abbrev - s_str_off), .sh_type = 0x7A000003, .sh_flags = 0, .sh_addr = 0, .sh_offset = s_abbrev_off, .sh_size = s_abbrev_sz, .sh_link = 0, .sh_info = 0, .sh_addralign = 1, .sh_entsize = 0 };

    // .debug_info
    sh = @as(*Elf64Shdr, @ptrCast(@alignCast(buf[p..]))); p += @sizeOf(Elf64Shdr);
    sh.* = .{ .sh_name = @intCast(str_debug_info - s_str_off), .sh_type = 0x7A000001, .sh_flags = 0, .sh_addr = 0, .sh_offset = s_info_off, .sh_size = s_info_sz, .sh_link = 0, .sh_info = 0, .sh_addralign = 1, .sh_entsize = 0 };

    // .debug_line
    sh = @as(*Elf64Shdr, @ptrCast(@alignCast(buf[p..]))); p += @sizeOf(Elf64Shdr);
    sh.* = .{ .sh_name = @intCast(str_debug_line - s_str_off), .sh_type = 0x7A000002, .sh_flags = 0, .sh_addr = 0, .sh_offset = s_line_off, .sh_size = s_line_sz, .sh_link = 0, .sh_info = 0, .sh_addralign = 1, .sh_entsize = 0 };

    // .shstrtab
    sh = @as(*Elf64Shdr, @ptrCast(@alignCast(buf[p..]))); p += @sizeOf(Elf64Shdr);
    sh.* = .{ .sh_name = @intCast(str_shstrtab - s_str_off), .sh_type = 3, .sh_flags = 0, .sh_addr = 0, .sh_offset = s_str_off, .sh_size = s_str_sz, .sh_link = 0, .sh_info = 0, .sh_addralign = 1, .sh_entsize = 0 };

    // Trim to actual size
    const actual = p;
    if (actual < total) {
        return buf[0..actual];
    }
    return buf;
}
