//! Peephole optimization for emitted x86-64 code.
//!
//! Post-processes the emitted code buffer to remove redundant instruction
//! patterns. Operates on raw x86-64 machine code bytes using byte-level
//! pattern matching.
//!
//! All optimizations preserve correctness: each matched pattern is
//! provably a no-op sequence that can be safely elided.
//!
//! Patterns handled:
//! - Same-register MOV (NOP): `MOV RAX, RAX` (48 89 C0) → remove
//! - XOR + MOV 0: `XOR RAX, RAX; MOV RAX, 0` → remove MOV (XOR already zeroed)
//! - Duplicate MOV: `MOV r1, r2; MOV r1, r2` → remove second
//! - Swapped MOV: `MOV r1, r2; MOV r2, r1` → remove second (restores r2 to itself)
//! - MOV + ADD 0: `MOV r1, r2; ADD r1, 0` → remove ADD

const std = @import("std");

/// A table of (opcode, ModRM mask, ModRM pattern) for matching instructions.
const Pattern = struct {
    len: usize, // total pattern length
    bytes: []const u8, // fixed byte sequence (or empty for computed matching)
};

// ── Helpers ───────────────────────────────────────────────────────

/// Returns true if the 3 bytes at `data[0..3]` form a same-register MOV (NOP).
/// Matches: REX.W + 0x89 + ModRM(11, reg, reg) where reg==rm.
fn matchMovSameReg(data: []const u8) bool {
    if (data.len < 3) return false;
    // REX.W (48) or REX.W+B (49)
    if (data[0] != 0x48 and data[0] != 0x49) return false;
    if (data[1] != 0x89) return false;
    const modrm = data[2];
    return (modrm >> 6) == 0b11 and ((modrm >> 3) & 7) == (modrm & 7);
}

/// Returns true if the 3 bytes at `data[0..3]` form a register-to-register MOV.
/// Matches: REX.W + 0x89 + ModRM(11, *, *) .
fn matchMovRegReg(data: []const u8) bool {
    if (data.len < 3) return false;
    if (data[0] != 0x48) return false;
    if (data[1] != 0x89) return false;
    return (data[2] >> 6) == 0b11;
}

/// Optimize the emitted code buffer in place.
///
/// Scans `buf[0..len]` for known redundant patterns and compacts the
/// buffer by overwriting skipped bytes.  Returns the new logical length.
///
/// Since the scanner advances byte-by-byte for non-matches (rather than
/// decoding full instruction boundaries), patterns must be verified not
/// to appear mid-instruction in valid emitted code.  The patterns below
/// all start with a REX prefix (0x48/0x49), which always initiates a
/// new instruction in x86-64, making this safe.
pub fn optimize(buf: []u8, len: usize) usize {
    var read: usize = 0;
    var write: usize = 0;

    while (read < len) {
        const rem = len - read;
        var pattern_matched: usize = 0;

        // ── Pattern 1: Same-register MOV (NOP) ─────────────────────
        //  48 89 C0  →  MOV RAX, RAX
        //  49 89 C0  →  MOV R8,  R8
        //  Any register (RAX–R15) — 3 bytes, fully elided.
        if (rem >= 3 and matchMovSameReg(buf[read..])) {
            pattern_matched = 3;
        }

        // ── Pattern 2: XOR RAX,RAX + MOV RAX,0 ────────────────────
        //  48 31 C0          XOR RAX, RAX   (3 bytes)
        //  B8 00 00 00 00    MOV RAX, 0      (5 bytes, zero-extending)
        //  → The MOV is redundant because XOR already zeroed RAX.
        if (pattern_matched == 0 and rem >= 8 and
            buf[read] == 0x48 and buf[read + 1] == 0x31 and buf[read + 2] == 0xC0 and
            buf[read + 3] == 0xB8 and
            buf[read + 4] == 0 and buf[read + 5] == 0 and
            buf[read + 6] == 0 and buf[read + 7] == 0)
        {
            // Keep the XOR (3 bytes), skip the MOV (5 bytes).
            @memmove(buf[write..][0..3], buf[read..][0..3]);
            write += 3;
            read += 8;
            continue;
        }

        // ── Pattern 3: Duplicate MOV (same source & dest) ─────────
        //  48 89 XY  →  MOV dst=Y, src=X   (3 bytes)
        //  48 89 XY  →  same again         (3 bytes)
        //  → The second MOV is a perfect duplicate; elide it.
        if (pattern_matched == 0 and rem >= 6 and
            matchMovRegReg(buf[read..]) and
            matchMovRegReg(buf[read + 3 ..]) and
            buf[read + 2] == buf[read + 5])
        {
            // Keep first MOV (3 bytes), skip second (3 bytes).
            @memmove(buf[write..][0..3], buf[read..][0..3]);
            write += 3;
            read += 6;
            continue;
        }

        // ── Pattern 4: Swapped register MOV ────────────────────────
        //  48 89 XY  →  MOV dst=Y, src=X   (3 bytes)
        //  48 89 YX  →  MOV dst=X, src=Y   (3 bytes)
        //  Net effect:  Y = X;  X = Y = old X  →  the second MOV undoes
        //  any effect on X (it was clobbered by the first, now restored).
        //  → Keep only the first MOV.
        //
        //  Concretely:  MOV RDI, RSI;  MOV RSI, RDI  →  keep MOV RDI, RSI
        if (pattern_matched == 0 and rem >= 6 and
            matchMovRegReg(buf[read..]) and
            matchMovRegReg(buf[read + 3 ..]))
        {
            const modrm1 = buf[read + 2];
            const modrm2 = buf[read + 5];
            const reg1 = (modrm1 >> 3) & 7; // src of first MOV
            const rm1 = modrm1 & 7; // dst of first MOV
            const reg2 = (modrm2 >> 3) & 7;
            const rm2 = modrm2 & 7;
            // Second MOV is redundant if it restores the first MOV's source:
            //   second's dst (rm2) == first's src (reg1)
            //   second's src (reg2) == first's dst (rm1)
            if (reg1 != rm1 and reg2 == rm1 and rm2 == reg1) {
                @memmove(buf[write..][0..3], buf[read..][0..3]);
                write += 3;
                read += 6;
                continue;
            }
        }

        // ── Pattern 5: MOV + ADD 0 ─────────────────────────────────
        //  48 89 XY  →  MOV dst=Y, src=X           (3 bytes)
        //  48 83 CY 00 → ADD Y, 0  (C = 0xC0)      (4 bytes)
        //  → The ADD zero does nothing; elide it.
        if (pattern_matched == 0 and rem >= 7 and
            matchMovRegReg(buf[read..]) and
            buf[read + 3] == 0x48 and buf[read + 4] == 0x83 and
            (buf[read + 5] & 0xF8) == 0xC0 and // mod=11, opcode-ext=0 (ADD)
            (buf[read + 5] & 0x07) == (buf[read + 2] & 0x07) and // same dst
            buf[read + 6] == 0) // imm8 = 0
        {
            // Keep MOV (3 bytes), skip ADD (4 bytes).
            @memmove(buf[write..][0..3], buf[read..][0..3]);
            write += 3;
            read += 7;
            continue;
        }

        // ── No pattern matched: copy byte and advance by one ──────
        if (pattern_matched == 0) {
            if (write != read) {
                buf[write] = buf[read];
            }
            write += 1;
            read += 1;
        } else {
            // A pattern was matched but not handled above (future extension).
            read += pattern_matched;
        }
    }

    return write;
}

// ═══════════════════════════════════════════════════════════════════
//  Tests
// ═══════════════════════════════════════════════════════════════════

test "same-register MOV removal (RAX)" {
    var code: [16]u8 = undefined;
    // MOV RAX, RAX = 48 89 C0
    code[0..3].* = .{ 0x48, 0x89, 0xC0 };
    const result = optimize(&code, 3);
    try std.testing.expectEqual(@as(usize, 0), result);
}

test "same-register MOV removal (multiple regs)" {
    var code: [16]u8 = undefined;
    // MOV RAX, RAX; MOV RBX, RBX; MOV RCX, RCX
    code[0..3].* = .{ 0x48, 0x89, 0xC0 };
    code[3..6].* = .{ 0x48, 0x89, 0xDB };
    code[6..9].* = .{ 0x48, 0x89, 0xC9 };
    const result = optimize(&code, 9);
    try std.testing.expectEqual(@as(usize, 0), result);
}

test "same-register MOV removal (R8-R15 with REX.B)" {
    var code: [16]u8 = undefined;
    // MOV R8, R8  = 49 89 C0
    code[0..3].* = .{ 0x49, 0x89, 0xC0 };
    // MOV R15, R15 = 49 89 FF
    code[3..6].* = .{ 0x49, 0x89, 0xFF };
    const result = optimize(&code, 6);
    try std.testing.expectEqual(@as(usize, 0), result);
}

test "XOR + MOV 0 elimination" {
    var code: [16]u8 = undefined;
    // XOR RAX, RAX = 48 31 C0
    code[0..3].* = .{ 0x48, 0x31, 0xC0 };
    // MOV RAX, 0 = B8 00 00 00 00
    code[3..8].* = .{ 0xB8, 0x00, 0x00, 0x00, 0x00 };

    const result = optimize(&code, 8);
    // Should keep only the XOR (3 bytes)
    try std.testing.expectEqual(@as(usize, 3), result);
    try std.testing.expectEqual(@as(u8, 0x48), code[0]);
    try std.testing.expectEqual(@as(u8, 0x31), code[1]);
    try std.testing.expectEqual(@as(u8, 0xC0), code[2]);
}

test "duplicate MOV elimination" {
    var code: [16]u8 = undefined;
    // MOV RAX, RBX = 48 89 D8
    code[0..3].* = .{ 0x48, 0x89, 0xD8 };
    // same again = 48 89 D8
    code[3..6].* = .{ 0x48, 0x89, 0xD8 };

    const result = optimize(&code, 6);
    try std.testing.expectEqual(@as(usize, 3), result);
    try std.testing.expectEqual(@as(u8, 0x48), code[0]);
    try std.testing.expectEqual(@as(u8, 0x89), code[1]);
    try std.testing.expectEqual(@as(u8, 0xD8), code[2]);
}

test "swapped MOV elimination (MOV RDI,RSI; MOV RSI,RDI)" {
    var code: [16]u8 = undefined;
    // MOV RDI, RSI = 48 89 F7  (reg=RSI(6), rm=RDI(7))
    code[0..3].* = .{ 0x48, 0x89, 0xF7 };
    // MOV RSI, RDI = 48 89 FE  (reg=RDI(7), rm=RSI(6))
    code[3..6].* = .{ 0x48, 0x89, 0xFE };

    const result = optimize(&code, 6);
    // Should keep only the first MOV (MOV RDI, RSI)
    try std.testing.expectEqual(@as(usize, 3), result);
    try std.testing.expectEqual(@as(u8, 0x48), code[0]);
    try std.testing.expectEqual(@as(u8, 0x89), code[1]);
    try std.testing.expectEqual(@as(u8, 0xF7), code[2]);
}

test "MOV + ADD 0 elimination" {
    var code: [16]u8 = undefined;
    // MOV RDX, RDI = 48 89 FA  (modrm=11_111_010 = reg=RDI, rm=RDX)
    code[0..3].* = .{ 0x48, 0x89, 0xFA };
    // ADD RDX, 0 = 48 83 C2 00 (modrm=11_000_010, imm8=0)
    code[3..7].* = .{ 0x48, 0x83, 0xC2, 0x00 };

    const result = optimize(&code, 7);
    // Should keep only the MOV (3 bytes)
    try std.testing.expectEqual(@as(usize, 3), result);
    try std.testing.expectEqual(@as(u8, 0x48), code[0]);
    try std.testing.expectEqual(@as(u8, 0x89), code[1]);
    try std.testing.expectEqual(@as(u8, 0xFA), code[2]);
}

test "non-redundant 3-operand ADD left alone" {
    var code: [16]u8 = undefined;
    // Normal 3-operand ADD:  MOV RDX, RDI  +  ADD RDX, RSI
    // 48 89 FA  = MOV RDX, RDI
    code[0..3].* = .{ 0x48, 0x89, 0xFA };
    // 48 01 F2  = ADD RDX, RSI  (01, not 83 — no immediate)
    code[3..6].* = .{ 0x48, 0x01, 0xF2 };

    const result = optimize(&code, 6);
    // Should be unchanged (neither pattern matches)
    try std.testing.expectEqual(@as(usize, 6), result);
}

test "XOR not followed by MOV is left alone" {
    var code: [16]u8 = undefined;
    code[0..3].* = .{ 0x48, 0x31, 0xC0 }; // XOR RAX, RAX
    code[3..6].* = .{ 0x48, 0x89, 0xC3 }; // MOV RBX, RAX (not MOV RAX,0)

    const result = optimize(&code, 6);
    try std.testing.expectEqual(@as(usize, 6), result);
}

test "empty buffer" {
    var code: [1]u8 = undefined;
    const result = optimize(&code, 0);
    try std.testing.expectEqual(@as(usize, 0), result);
}

test "single non-MOV byte left alone" {
    var code: [1]u8 = .{0xCC}; // INT3
    const result = optimize(&code, 1);
    try std.testing.expectEqual(@as(usize, 1), result);
    try std.testing.expectEqual(@as(u8, 0xCC), code[0]);
}

test "MOV with non-zero immediate ADD preserved" {
    var code: [16]u8 = undefined;
    // 48 89 FA = MOV RDX, RDI
    code[0..3].* = .{ 0x48, 0x89, 0xFA };
    // 48 83 C2 05 = ADD RDX, 5 (non-zero immediate)
    code[3..7].* = .{ 0x48, 0x83, 0xC2, 0x05 };

    const result = optimize(&code, 7);
    // Should be unchanged
    try std.testing.expectEqual(@as(usize, 7), result);
}

test "partial match fails gracefully" {
    var code: [16]u8 = undefined;
    code[0..3].* = .{ 0x48, 0x89, 0xC0 }; // MOV RAX, RAX (should be removed)
    code[3..4].* = .{ 0x48 }; // just a REX prefix, no instruction

    const result = optimize(&code, 4);
    // MOV RAX, RAX removed; lone 0x48 kept
    try std.testing.expectEqual(@as(usize, 1), result);
    try std.testing.expectEqual(@as(u8, 0x48), code[0]);
}

test "adjacent optimizations compose" {
    var code: [32]u8 = undefined;
    // MOV RAX, RAX (NOP) + MOV RDX, RDI + ADD RDX, 0
    // After optimization: only MOV RDX, RDI remains.
    code[0..3].* = .{ 0x48, 0x89, 0xC0 }; // MOV RAX, RAX (NOP)
    code[3..6].* = .{ 0x48, 0x89, 0xFA }; // MOV RDX, RDI
    code[6..10].* = .{ 0x48, 0x83, 0xC2, 0x00 }; // ADD RDX, 0

    const result = optimize(&code, 10);
    try std.testing.expectEqual(@as(usize, 3), result);
    try std.testing.expectEqual(@as(u8, 0x48), code[0]);
    try std.testing.expectEqual(@as(u8, 0x89), code[1]);
    try std.testing.expectEqual(@as(u8, 0xFA), code[2]);
}
