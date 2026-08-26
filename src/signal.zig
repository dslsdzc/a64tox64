//! Signal handlers for JIT crash diagnostics and recovery.
//!
//! Features:
//! - SIGSEGV/SIGILL/SIGFPE/SIGBUS crash diagnostics (fault addr, RIP, block, regs)
//! - SIGSEGV in JIT writable code pages → self-modifying code (SMC) detection:
//!   invalidates affected blocks and advances past the fault for re-translation
//! - Recovery: signals NOT in JIT code re-raise with default handler (core dump)
//! - Crash only: signals in JIT code print diagnostics and return (process survives)
//! - O(log n) block lookup via sorted block range array

const std = @import("std");
const linux = std.os.linux;
const Cache = @import("cache.zig");
const CodeCache = Cache.CodeCache;

var saved_cache: ?*CodeCache = null;

/// Pending signal to forward to guest (set by handler, consumed by runtime).
pub var pending_signal: i32 = -1;
pub var pending_fault_addr: u64 = 0;


const MAX_BLOCK_ENTRIES = 4096;
var block_ranges: [MAX_BLOCK_ENTRIES]BlockRange = undefined;
var block_count: usize = 0;

const MAX_GUEST_PAGES = 1024;
var guest_pages: [MAX_GUEST_PAGES]u64 = undefined;
var guest_page_count: usize = 0;

/// Base address of guest memory (set by setup()). Used to distinguish
/// guest memory writes from other SEGV faults.
var guest_mem_base: u64 = 0;
var guest_mem_size: usize = 0;

const BlockRange = struct {
    start: u64,
    end: u64,
    guest_pc: u64,
};

// x86-64 mcontext_t: gregs[0..22]
const MContext = extern struct { gregs: [23]u64 };
const UContext = extern struct {
    flags: u64, link: u64, stack: [3]u64,
    mcontext: MContext,
    sigmask: linux.sigset_t,
};

const SA_SIGINFO: c_ulong = 4;
const SA_ONSTACK: c_ulong = 0x08000000;
const SA_RESTART: c_ulong = 0x10000000;

/// Register a block for fast crash-time lookup. Call after each translation.
/// Also tracks guest pages for SMC detection.
pub fn registerBlock(host_addr: []u8, guest_pc: u64) void {
    if (block_count >= MAX_BLOCK_ENTRIES) return;
    const start = @intFromPtr(host_addr.ptr);
    block_ranges[block_count] = .{
        .start = start,
        .end = start + host_addr.len,
        .guest_pc = guest_pc,
    };
    block_count += 1;

    // Track guest page for SMC detection
    const guest_page = guest_pc & ~@as(u64, 0xFFF);
    for (guest_pages[0..guest_page_count]) |p| {
        if (p == guest_page) return; // already tracked
    }
    if (guest_page_count < MAX_GUEST_PAGES) {
        guest_pages[guest_page_count] = guest_page;
        guest_page_count += 1;
    }
}

const MAX_GUEST_SIGNALS = 64;
var guest_handler_flags: [MAX_GUEST_SIGNALS]u8 = blk: {
    var arr: [MAX_GUEST_SIGNALS]u8 = undefined;
    @memset(&arr, 0);
    break :blk arr;
};

/// Mark whether a guest signal handler is registered for `sig`.
/// Called by the runtime when guest rt_sigaction is intercepted.
pub fn setGuestHandler(sig: usize, registered: bool) void {
    if (sig < MAX_GUEST_SIGNALS) guest_handler_flags[sig] = @intFromBool(registered);
}

fn guestHandlerRegistered(sig: usize) bool {
    return sig < MAX_GUEST_SIGNALS and guest_handler_flags[sig] != 0;
}

/// Mark a guest page as having modified code — invalidate all blocks.
/// Called from the SEGV handler when guest code writes to a page with
/// cached translations. Returns the number of blocks invalidated.
fn invalidateGuestPage(guest_page: u64) usize {
    var count: usize = 0;
    var i: usize = 0;
    while (i < block_count) {
        const bp = block_ranges[i].guest_pc & ~@as(u64, 0xFFF);
        if (bp == guest_page) {
            // Remove by shifting
            var j = i;
            while (j + 1 < block_count) {
                block_ranges[j] = block_ranges[j + 1];
                j += 1;
            }
            block_count -= 1;
            count += 1;
        } else {
            i += 1;
        }
    }
    // Also invalidate in the actual code cache
    if (saved_cache) |cache| {
        cache.invalidatePage(guest_page);
    }
    return count;
}

/// Block lookup for crash-time diagnostics.
/// Linear scan: block_ranges is NOT sorted by host address (blocks come from
/// multiple code-cache mmaps and can be registered out of order), so binary
/// search would miss entries. Max 4096 entries, crash handler is a rare path.
fn findBlock(rip: u64) ?u64 {
    var i: usize = 0;
    while (i < block_count) {
        const entry = block_ranges[i];
        if (rip >= entry.start and rip < entry.end) return entry.guest_pc;
        i += 1;
    }
    return null;
}

fn putHex(v: u64) void {
    const hex = "0123456789abcdef";
    var buf: [18]u8 = undefined;
    buf[0] = '0'; buf[1] = 'x';
    var i: usize = 17;
    var val = v;
    while (i >= 2) {
        buf[i] = hex[val & 0xF];
        val >>= 4;
        i -= 1;
    }
    _ = linux.write(2, &buf, buf.len);
}

fn putStr(s: []const u8) void { _ = linux.write(2, s.ptr, s.len); }

/// Check if a guest memory page has cached translations (SMC candidate).
/// Returns the guest page number or 0 if no translations exist.
fn guestPageWithTranslations(fault_addr: u64) u64 {
    // Must be within guest memory range
    if (guest_mem_size == 0) return 0;
    if (fault_addr < guest_mem_base or fault_addr >= guest_mem_base + guest_mem_size) return 0;
    const guest_page = fault_addr & ~@as(u64, 0xFFF);
    // Check if we have translations for this guest page
    for (guest_pages[0..guest_page_count]) |p| {
        if (p == guest_page) return guest_page;
    }
    return 0;
}

fn handler(sig: linux.SIG, info: *const linux.siginfo_t, ctx_ptr: ?*anyopaque) callconv(.c) void {
    const sig_int = @intFromEnum(sig);
    const ctx = @as(*const UContext, @ptrCast(@alignCast(ctx_ptr orelse return)));
    const gregs = @constCast(&ctx.mcontext.gregs);
    const rip = gregs[16];
    const info_addr = @intFromPtr(info);
    const fault_addr = @as(*usize, @alignCast(@ptrCast(@as(*anyopaque, @ptrFromInt(info_addr + 16))))).*;

    // 1. SMC detection first: SIGSEGV writing to a guest page that has
    //    cached translations → allow the write, invalidate, continue.
    if (sig == linux.SIG.SEGV) {
        const smc_page = guestPageWithTranslations(fault_addr);
        if (smc_page != 0) {
            putStr("\n=== SMC ===\n");
            putStr("Guest page "); putHex(smc_page);
            putStr(" modified — invalidating translations\n");
            const n = invalidateGuestPage(smc_page);
            putHex(n); putStr(" blocks invalidated\n");
            var pi: usize = 0;
            while (pi < guest_page_count) {
                if (guest_pages[pi] == smc_page) {
                    var pj = pi;
                    while (pj + 1 < guest_page_count) {
                        guest_pages[pj] = guest_pages[pj + 1];
                        pj += 1;
                    }
                    guest_page_count -= 1;
                } else {
                    pi += 1;
                }
            }
            return;
        }
    }

    // 2. Crash diagnostics for faults inside JIT blocks.
    const in_jit = findBlock(rip) != null;
    if (in_jit) {
        const sig_name = switch (sig) {
            linux.SIG.SEGV => "SIGSEGV",
            linux.SIG.ILL => "SIGILL",
            linux.SIG.FPE => "SIGFPE",
            linux.SIG.BUS => "SIGBUS",
            else => "SIGNAL",
        };
        putStr("\n=== "); putStr(sig_name); putStr(" ===\n");
        putStr("Fault addr: "); putHex(fault_addr); putStr("\n");
        putStr("Host RIP:   "); putHex(rip); putStr("\n");
        // findBlock(rip) is non-null here (in_jit); print the guest PC
        if (findBlock(rip)) |pc| {
            putStr("Guest PC:   "); putHex(pc); putStr("\n");
        }
        putStr("\nRAX="); putHex(gregs[13]); putStr(" RBX="); putHex(gregs[11]);
        putStr(" RCX="); putHex(gregs[14]); putStr(" RDX="); putHex(gregs[12]);
        putStr("\nRDI="); putHex(gregs[8]);  putStr(" RSI="); putHex(gregs[9]);
        putStr(" RBP="); putHex(gregs[10]); putStr(" RSP="); putHex(gregs[15]);
        putStr("\nR8 ="); putHex(gregs[0]);  putStr(" R9 ="); putHex(gregs[1]);
        putStr(" R10="); putHex(gregs[2]);  putStr(" R11="); putHex(gregs[3]);
        putStr("\nR12="); putHex(gregs[4]);  putStr(" R13="); putHex(gregs[5]);
        putStr(" R14="); putHex(gregs[6]);  putStr(" R15="); putHex(gregs[7]);
        putStr("\n\n");
    }

    // 3. Forward to guest handler — only when a guest handler is registered.
    //    (guest_sigactions is populated by rt_sigaction interception, which
    //    is not implemented yet, so this branch is currently never taken.)
    if (sig_int >= 0 and @as(usize, @intCast(sig_int)) < MAX_GUEST_SIGNALS) {
        if (guestHandlerRegistered(@as(usize, @intCast(sig_int)))) {
            pending_signal = @as(i32, @intCast(sig_int));
            pending_fault_addr = fault_addr;
            putStr("→ Signal pending — will dispatch to guest handler\n");
            return;
        }
    }

    // 4. Otherwise: restore default handler and re-raise (core dump).
    var dfl: linux.Sigaction = .{
        .handler = .{ .handler = null },
        .mask = @as(linux.sigset_t, undefined),
        .flags = 0,
    };
    _ = linux.sigaction(sig, &dfl, null);
    _ = linux.syscall2(.kill, @as(u64, @bitCast(@as(i64, linux.getpid()))), @as(usize, sig_int));
    return; // unreachable
}

/// Set guest memory bounds for SMC detection.
pub fn setupGuestMem(base: u64, size: usize) void {
    guest_mem_base = base;
    guest_mem_size = size;
}

pub fn setup(cache: *CodeCache) !void {
    saved_cache = cache;

    const stack = try std.posix.mmap(null, 65536,
        .{ .READ = true, .WRITE = true },
        .{ .TYPE = .PRIVATE, .ANONYMOUS = true }, -1, 0);
    const AltStack = extern struct { ss_sp: usize, ss_flags: i32, ss_size: usize };
    var alt: AltStack = .{ .ss_sp = @intFromPtr(stack.ptr), .ss_size = stack.len, .ss_flags = 0 };
    _ = linux.syscall3(.sigaltstack, @intFromPtr(&alt), 0, 0);

    const flags: c_ulong = SA_SIGINFO | SA_ONSTACK | SA_RESTART;
    var sa: linux.Sigaction = .{
        .handler = .{ .sigaction = handler },
        .mask = @as(linux.sigset_t, undefined),
        .flags = flags,
    };
    _ = linux.sigaction(linux.SIG.SEGV, &sa, null);
    _ = linux.sigaction(linux.SIG.ILL, &sa, null);
    _ = linux.sigaction(linux.SIG.FPE, &sa, null);
    _ = linux.sigaction(linux.SIG.BUS, &sa, null);
}
