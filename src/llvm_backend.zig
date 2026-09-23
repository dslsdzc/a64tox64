//! LLVM backend for a64tox64.
//!
//! Translates IROp[] → LLVM IR → ORC JIT → x86-64 machine code.
//! Used as a selective backend when the hand-written emitter can't
//! handle a block (e.g. x14+ registers, SIMD) or when LLVM's
//! register allocator would produce better code.
//!
//! Depends on system libLLVM-22.so; linked via build.zig.

const std = @import("std");
const Ir = @import("ir.zig");
const IROp = Ir.IROp;
const Tag = Ir.Tag;

// ── LLVM opaque type aliases ─────────────────────────────────

const ContextRef = *opaque {};
const BuilderRef = *opaque {};
const ModuleRef = *opaque {};
const ValueRef = *opaque {};
const TypeRef = *opaque {};
const OrcJITRef = *opaque {};
const OrcJITDylibRef = *opaque {};
const ErrorRef = ?*anyopaque;
const OrcThreadSafeModuleRef = *opaque {};
const OrcThreadSafeContextRef = *opaque {};
const OrcJITBuilderRef = *opaque {};
const OrcJITTargetMachineBuilderRef = *opaque {};
const LLVMTargetRef = *opaque {};
const LLVMTargetMachineOptionsRef = *opaque {};
const LLVMTargetMachineRef = *opaque {};

const OrcJITTargetAddress = u64;

// ── Helpers ──────────────────────────────────────────────────

/// Create a v4f32 constant with all lanes = val.
fn v4f32Const(b: BuilderRef, f32_ty: TypeRef, v4f32_ty: TypeRef, val: f64) ValueRef {
    const zero = LLVMConstNull(v4f32_ty);
    const scalar = LLVMConstReal(f32_ty, val);
    const idx0 = LLVMConstInt(LLVMInt32Type(), 0, 0);
    const v1 = LLVMBuildInsertElement(b, zero, scalar, idx0, "vc_0");
    // Shuffle to broadcast lane 0 to all lanes — but we don't have BuildShuffleVector handy
    // Instead, insert into all 4 lanes
    const idx1 = LLVMConstInt(LLVMInt32Type(), 1, 0);
    const v2 = LLVMBuildInsertElement(b, v1, scalar, idx1, "vc_1");
    const idx2 = LLVMConstInt(LLVMInt32Type(), 2, 0);
    const v3 = LLVMBuildInsertElement(b, v2, scalar, idx2, "vc_2");
    const idx3 = LLVMConstInt(LLVMInt32Type(), 3, 0);
    return LLVMBuildInsertElement(b, v3, scalar, idx3, "vc_3");
}

// ── saSLP SIMD combining (Liu et al. 2019) ────────────────────
// Detect consecutive 128-bit NEON ops on adjacent vector register pairs
// and combine them into a single 256-bit AVX operation.
//
// ARM64 NEON processes 128-bit vectors as 4x32-bit elements (stored as v4f32
// in the LLVM backend). x86-64 AVX processes 256-bit vectors as 8x32-bit
// elements (v8f32). Two consecutive independent 128-bit NEON ops can be
// combined into one 256-bit AVX op by:
//   1. Shuffle-concatenating two v4f32 source values into v8f32
//   2. Applying the operation on v8f32
//   3. Shuffle-extracting results back to two v4f32 stores

/// Returns true if the op tag can be combined via saSLP.
fn isCombinableSimdOp(tag: Tag) bool {
    return switch (tag) {
        .vfadd, .vfsub, .vfmul,
        .vadd, .vsub, .vmul,
        .vand, .vorr, .veor,
        .vfmin, .vfmax,
        => true,
        else => false,
    };
}

/// Check if two consecutive SIMD ops can be combined into one wider op.
/// Both ops must have the same tag, operate on adjacent vreg pairs,
/// and have no data dependency between them.
fn tryCombineSimdPair(op0: IROp, op1: IROp) bool {
    if (op0.tag != op1.tag) return false;
    if (!isCombinableSimdOp(op0.tag)) return false;

    // Both must target vector registers (v0-v31, stored as guest VReg 31+)
    if (op0.dest < 31 or op1.dest < 31) return false;
    if (op0.src0 < 31 or op1.src0 < 31) return false;

    // Dest vregs must be adjacent even-odd pair (vN, vN+1)
    const d0 = op0.dest - 31;
    const d1 = op1.dest - 31;
    if (d0 % 2 != 0 or d1 != d0 + 1) return false;

    // src0 vregs must be adjacent even-odd pair
    const s00 = op0.src0 - 31;
    const s01 = op1.src0 - 31;
    if (s00 % 2 != 0 or s01 != s00 + 1) return false;

    // No data dependency: op0's dest not used as source by op1
    if (op0.dest == op1.src0 or op0.dest == op1.src1) return false;

    // Immediates must match (both 0 for element-wise ops)
    if (op0.imm != op1.imm) return false;

    // src1 handling — both must have or both lack src1
    const op0_src1 = op0.src1 >= 31 and op0.src1 < 128;
    const op1_src1 = op1.src1 >= 31 and op1.src1 < 128;
    if (op0_src1 != op1_src1) return false;

    if (op0_src1) {
        const s10 = op0.src1 - 31;
        const s11 = op1.src1 - 31;
        if (s10 % 2 != 0 or s11 != s10 + 1) return false;
        if (op0.dest == op1.src1) return false;
    }

    return true;
}

/// Format into a fixed-size buffer and return a null-terminated pointer.
fn fmtZ(buf: []u8, comptime fmt: []const u8, args: anytype) [*:0]const u8 {
    @memset(buf, 0);
    _ = std.fmt.bufPrint(buf, fmt, args) catch {};
    return @as([*:0]const u8, @ptrCast(buf.ptr));
}

// ── External declarations ────────────────────────────────────

// Target initialization (per-arch in LLVM 22+; no more LLVMInitializeNativeTarget)
extern fn LLVMInitializeX86TargetInfo() void;
extern fn LLVMInitializeX86Target() void;
extern fn LLVMInitializeX86TargetMC() void;
extern fn LLVMInitializeX86AsmPrinter() void;

// Core
extern fn LLVMContextCreate() ContextRef;
extern fn LLVMContextDispose(ContextRef) void;
extern fn LLVMModuleCreateWithName([*:0]const u8) ModuleRef;
extern fn LLVMDisposeModule(ModuleRef) void;
extern fn LLVMCreateBuilder() BuilderRef;
extern fn LLVMDisposeBuilder(BuilderRef) void;
extern fn LLVMPositionBuilderAtEnd(BuilderRef, ValueRef) void;
extern fn LLVMAppendBasicBlock(ValueRef, [*:0]const u8) ValueRef;
extern fn LLVMInt64Type() TypeRef;
extern fn LLVMInt32Type() TypeRef;
extern fn LLVMInt8Type() TypeRef;
extern fn LLVMInt16Type() TypeRef;
extern fn LLVMInt1Type() TypeRef;
extern fn LLVMVoidType() TypeRef;
extern fn LLVMPointerType(TypeRef, c_uint) TypeRef;
extern fn LLVMFunctionType(TypeRef, [*]const TypeRef, c_uint, c_uint) TypeRef;
extern fn LLVMAddFunction(ModuleRef, [*:0]const u8, TypeRef) ValueRef;
extern fn LLVMAddGlobal(ModuleRef, TypeRef, [*:0]const u8) ValueRef;
extern fn LLVMSetInitializer(ValueRef, ValueRef) void;
extern fn LLVMGetParam(ValueRef, c_uint) ValueRef;
extern fn LLVMSetValueName(ValueRef, [*:0]const u8) void;
extern fn LLVMSetVolatile(ValueRef, c_uint) void;
extern fn LLVMConstInt(TypeRef, c_ulonglong, c_uint) ValueRef;
extern fn LLVMBuildAlloca(BuilderRef, TypeRef, [*:0]const u8) ValueRef;
extern fn LLVMBuildLoad2(BuilderRef, TypeRef, ValueRef, [*:0]const u8) ValueRef;
extern fn LLVMBuildStore(BuilderRef, ValueRef, ValueRef) ValueRef;
extern fn LLVMBuildAdd(BuilderRef, ValueRef, ValueRef, [*:0]const u8) ValueRef;
extern fn LLVMBuildSub(BuilderRef, ValueRef, ValueRef, [*:0]const u8) ValueRef;
extern fn LLVMBuildMul(BuilderRef, ValueRef, ValueRef, [*:0]const u8) ValueRef;
extern fn LLVMBuildAnd(BuilderRef, ValueRef, ValueRef, [*:0]const u8) ValueRef;
extern fn LLVMBuildOr(BuilderRef, ValueRef, ValueRef, [*:0]const u8) ValueRef;
extern fn LLVMBuildXor(BuilderRef, ValueRef, ValueRef, [*:0]const u8) ValueRef;
extern fn LLVMBuildUDiv(BuilderRef, ValueRef, ValueRef, [*:0]const u8) ValueRef;
extern fn LLVMBuildSDiv(BuilderRef, ValueRef, ValueRef, [*:0]const u8) ValueRef;
extern fn LLVMBuildNot(BuilderRef, ValueRef, [*:0]const u8) ValueRef;
extern fn LLVMBuildNeg(BuilderRef, ValueRef, [*:0]const u8) ValueRef;
extern fn LLVMBuildShl(BuilderRef, ValueRef, ValueRef, [*:0]const u8) ValueRef;
extern fn LLVMBuildLShr(BuilderRef, ValueRef, ValueRef, [*:0]const u8) ValueRef;
extern fn LLVMBuildAShr(BuilderRef, ValueRef, ValueRef, [*:0]const u8) ValueRef;
extern fn LLVMBuildGEP2(BuilderRef, TypeRef, ValueRef, [*]const ValueRef, c_uint, [*:0]const u8) ValueRef;
extern fn LLVMBuildRetVoid(BuilderRef) ValueRef;
extern fn LLVMBuildRet(BuilderRef, ValueRef) ValueRef;
extern fn LLVMBuildPtrToInt(BuilderRef, ValueRef, TypeRef, [*:0]const u8) ValueRef;
extern fn LLVMBuildIntToPtr(BuilderRef, ValueRef, TypeRef, [*:0]const u8) ValueRef;
extern fn LLVMBuildBr(BuilderRef, ValueRef) ValueRef;
extern fn LLVMVerifyFunction(ValueRef, c_uint) void;
extern fn LLVMPrintModuleToFile(ModuleRef, [*:0]const u8, [*c]?[*:0]u8) c_int;
extern fn LLVMGetErrorMessage(ErrorRef) [*:0]const u8;
extern fn LLVMDisposeErrorMessage([*:0]const u8) void;

// SIMD / vector
extern fn LLVMVectorType(ElementType: TypeRef, ElementCount: c_uint) TypeRef;
extern fn LLVMBuildFAdd(BuilderRef, ValueRef, ValueRef, [*:0]const u8) ValueRef;
extern fn LLVMBuildFSub(BuilderRef, ValueRef, ValueRef, [*:0]const u8) ValueRef;
extern fn LLVMBuildFMul(BuilderRef, ValueRef, ValueRef, [*:0]const u8) ValueRef;
extern fn LLVMBuildFDiv(BuilderRef, ValueRef, ValueRef, [*:0]const u8) ValueRef;
extern fn LLVMBuildBitCast(BuilderRef, ValueRef, TypeRef, [*:0]const u8) ValueRef;
extern fn LLVMBuildExtractElement(BuilderRef, ValueRef, ValueRef, [*:0]const u8) ValueRef;
extern fn LLVMBuildInsertElement(BuilderRef, ValueRef, ValueRef, ValueRef, [*:0]const u8) ValueRef;
extern fn LLVMBuildShuffleVector(BuilderRef, ValueRef, ValueRef, ValueRef, [*:0]const u8) ValueRef;
extern fn LLVMBuildFPTrunc(BuilderRef, ValueRef, TypeRef, [*:0]const u8) ValueRef;
extern fn LLVMBuildFPExt(BuilderRef, ValueRef, TypeRef, [*:0]const u8) ValueRef;
extern fn LLVMBuildFPCast(BuilderRef, ValueRef, TypeRef, [*:0]const u8) ValueRef;
extern fn LLVMBuildTrunc(BuilderRef, ValueRef, TypeRef, [*:0]const u8) ValueRef;
extern fn LLVMBuildSExt(BuilderRef, ValueRef, TypeRef, [*:0]const u8) ValueRef;
extern fn LLVMBuildZExt(BuilderRef, ValueRef, TypeRef, [*:0]const u8) ValueRef;
extern fn LLVMBuildSIToFP(BuilderRef, ValueRef, TypeRef, [*:0]const u8) ValueRef;
extern fn LLVMBuildUIToFP(BuilderRef, ValueRef, TypeRef, [*:0]const u8) ValueRef;
extern fn LLVMBuildFPToSI(BuilderRef, ValueRef, TypeRef, [*:0]const u8) ValueRef;
extern fn LLVMBuildFPToUI(BuilderRef, ValueRef, TypeRef, [*:0]const u8) ValueRef;
extern fn LLVMBuildCall2(BuilderRef, TypeRef, ValueRef, [*]const ValueRef, c_uint, [*:0]const u8) ValueRef;
extern fn LLVMGetNamedFunction(ModuleRef, [*:0]const u8) ?ValueRef;
extern fn LLVMConstNull(TypeRef) ValueRef;
extern fn LLVMConstVector([*]const ValueRef, c_uint) ValueRef;
extern fn LLVMConstReal(TypeRef, f64) ValueRef;
extern fn LLVMDoubleType() TypeRef;
extern fn LLVMFloatType() TypeRef;
extern fn LLVMBuildICmp(BuilderRef, c_uint, ValueRef, ValueRef, [*:0]const u8) ValueRef;
extern fn LLVMBuildFCmp(BuilderRef, c_uint, ValueRef, ValueRef, [*:0]const u8) ValueRef;
extern fn LLVMBuildSelect(BuilderRef, ValueRef, ValueRef, ValueRef, [*:0]const u8) ValueRef;

// ICmp predicates
const LLVMIntEQ: c_uint = 32;
const LLVMIntNE: c_uint = 33;
const LLVMIntSGT: c_uint = 38;
const LLVMIntSGE: c_uint = 39;
const LLVMIntSLT: c_uint = 40;
const LLVMIntSLE: c_uint = 41;
const LLVMIntUGT: c_uint = 34;
const LLVMIntUGE: c_uint = 35;
const LLVMIntULT: c_uint = 36;
const LLVMIntULE: c_uint = 37;

// FCmp predicates
const LLVMRealOEQ: c_uint = 33;
const LLVMRealONE: c_uint = 34;
const LLVMRealOGT: c_uint = 35;
const LLVMRealOGE: c_uint = 36;
const LLVMRealOLT: c_uint = 37;
const LLVMRealOLE: c_uint = 38;

// ORC JIT
extern fn LLVMOrcCreateLLJITBuilder() OrcJITBuilderRef;
extern fn LLVMOrcCreateLLJIT(*OrcJITRef, OrcJITBuilderRef) ErrorRef;
extern fn LLVMOrcDisposeLLJIT(OrcJITRef) ErrorRef;
extern fn LLVMOrcLLJITGetMainJITDylib(OrcJITRef) OrcJITDylibRef;
extern fn LLVMOrcCreateNewThreadSafeContext() OrcThreadSafeContextRef;
extern fn LLVMOrcCreateNewThreadSafeContextFromLLVMContext(ContextRef) OrcThreadSafeContextRef;
extern fn LLVMOrcCreateNewThreadSafeModule(ModuleRef, OrcThreadSafeContextRef) OrcThreadSafeModuleRef;
extern fn LLVMOrcLLJITAddLLVMIRModule(OrcJITRef, OrcJITDylibRef, OrcThreadSafeModuleRef) ErrorRef;
extern fn LLVMOrcLLJITLookup(OrcJITRef, *OrcJITTargetAddress, [*:0]const u8) ErrorRef;
extern fn LLVMOrcLLJITBuilderSetJITTargetMachineBuilder(OrcJITBuilderRef, OrcJITTargetMachineBuilderRef) void;
extern fn LLVMOrcJITTargetMachineBuilderCreateFromTargetMachine(LLVMTargetMachineRef) OrcJITTargetMachineBuilderRef;
extern fn LLVMGetDefaultTargetTriple() [*:0]const u8;
extern fn LLVMSetTarget(ModuleRef, [*:0]const u8) void;
extern fn LLVMGetTargetFromTriple([*:0]const u8, *LLVMTargetRef, *?[*:0]const u8) c_int;
extern fn LLVMCreateTargetMachineOptions() LLVMTargetMachineOptionsRef;
extern fn LLVMTargetMachineOptionsSetCodeGenOptLevel(LLVMTargetMachineOptionsRef, c_uint) void;
extern fn LLVMCreateTargetMachineWithOptions(LLVMTargetRef, [*:0]const u8, LLVMTargetMachineOptionsRef) LLVMTargetMachineRef;
extern fn LLVMDisposeTargetMachine(LLVMTargetMachineRef) void;

const LLVMReturnStatusAction: c_uint = 1;
const LLVMAbortProcessAction: c_uint = 2;

/// Byte offset of v[0] within Arm64State struct.
/// Calculated: x[31]@0 + sp@248 + pc@256 + nzcv@264 + fpcr@268 + fpsr@272 + pad@276-287 = 288
const V_REG_OFFSET: u64 = 288;

// ── JIT state ───────────────────────────────────────────────

var jit: ?JitState = null;

extern fn getenv([*:0]const u8) ?[*:0]const u8;

/// Check if LLVM backend is enabled. Default: disabled (ORC JIT can hang on
/// some systems). Set A64TOX64_LLVM=1 to enable.
fn enableLlvm() bool {
    const env = getenv("A64TOX64_LLVM") orelse return false;
    return env[0] == '1';
}

const JitState = struct {
    jit_ref: OrcJITRef,
    allocator: std.mem.Allocator,
    module_count: u32 = 0,
};

// ── Public API ──────────────────────────────────────────────

pub fn init(allocator: std.mem.Allocator) !void {
    if (jit != null) return;
    LLVMInitializeX86TargetInfo();
    LLVMInitializeX86Target();
    LLVMInitializeX86TargetMC();
    LLVMInitializeX86AsmPrinter();

    // Create a TargetMachine with CodeGenOpt::None to prevent ORC JIT
    // from applying aggressive optimizations that break sp_get/sp_put
    // memory semantics (CodeGenOpt::Default generates incorrect code
    // for loads from state struct).
    const triple = LLVMGetDefaultTargetTriple();
    var target: LLVMTargetRef = undefined;
    var err_msg: ?[*:0]const u8 = null;
    if (LLVMGetTargetFromTriple(triple, &target, &err_msg) != 0) {
        std.log.err("LLVM target lookup: {s}", .{err_msg orelse "unknown"});
        return error.LlvmInitFailed;
    }
    const tm_options = LLVMCreateTargetMachineOptions();
    LLVMTargetMachineOptionsSetCodeGenOptLevel(tm_options, @as(c_uint, 0)); // LLVMCodeGenLevelNone
    const tm = LLVMCreateTargetMachineWithOptions(target, triple, tm_options);

    var jit_ref: OrcJITRef = undefined;
    const lljit_builder = LLVMOrcCreateLLJITBuilder();
    const jtmb = LLVMOrcJITTargetMachineBuilderCreateFromTargetMachine(tm);
    LLVMOrcLLJITBuilderSetJITTargetMachineBuilder(lljit_builder, jtmb);
    const err = LLVMOrcCreateLLJIT(&jit_ref, lljit_builder);
    if (err != null) {
        const msg = LLVMGetErrorMessage(err);
        defer LLVMDisposeErrorMessage(msg);
        std.log.err("LLVM init: {s}", .{msg});
        LLVMDisposeTargetMachine(tm);
        return error.LlvmInitFailed;
    }
    jit = JitState{ .jit_ref = jit_ref, .allocator = allocator };
}

pub fn deinit() void {
    const s = jit orelse return;
    _ = LLVMOrcDisposeLLJIT(s.jit_ref);
    // Context is owned by ThreadSafeModules added to the JIT.
    // LLVMOrcDisposeLLJIT handles cleanup of all contexts.
    jit = null;
}

/// Return true if the block should use the LLVM backend.
pub fn shouldUseLlvm(ops: []const IROp) bool {
    if (jit == null) return false;
    // Track register pressure: hand-emitter has 12 host regs
    // (R14 dedicated scratch, R15 for SP, RSP excluded).
    var unique_regs: u32 = 0;
    var seen: u32 = 0; // bitmask for regs 0-30
    for (ops) |op| {
        if (op.dest < 31) {
            const bit = @as(u32, 1) << @as(u5, @truncate(op.dest));
            if (seen & bit == 0) { seen |= bit; unique_regs += 1; }
        }
        if (op.src0 < 31) {
            const bit = @as(u32, 1) << @as(u5, @truncate(op.src0));
            if (seen & bit == 0) { seen |= bit; unique_regs += 1; }
        }
        if (op.src1 < 31 and op.src1 != 0x1F) {
            const bit = @as(u32, 1) << @as(u5, @truncate(op.src1));
            if (seen & bit == 0) { seen |= bit; unique_regs += 1; }
        }
        if (isSimdTag(op.tag)) return true;
    }
    // Hand emitter handles ≤12 unique regs. >12 regs need LLVM.
    // Use LLVM for high-register blocks (MFHBT hybrid approach).
    if (unique_regs > 12) return true;
    if (!enableLlvm()) return false;
    for (ops) |op| {
        if (isSimdTag(op.tag)) return true;
    }
    return false;
}

/// Returns true if the tag is a SIMD/vector operation that requires LLVM.
fn isSimdTag(tag: Tag) bool {
    return switch (tag) {
        .load_v128, .store_v128, .load_v64, .store_v64,
        .vadd, .vsub, .vmul, .vmla, .vmls,
        .vabs, .vneg, .vmin, .vmax, .vabd,
        .vpadd, .vpmin, .vpmax,
        .vqadd, .vqsub, .vaddlp, .vaddlv, .vabdl,
        .vfadd, .vfsub, .vfmul, .vfdiv, .vfma, .vfms,
        .vfmin, .vfmax, .vfabs, .vfneg,
        .vfrecpe, .vfrecps, .vfsqrt, .vfmulx,
        .vceq, .vcgt, .vcge, .vfcmp,
        .vshl, .vshr, .vsra, .vsri, .vsli,
        .vqshl, .vqshlu, .vrshr, .vshrn, .vrshrn,
        .fcvt, .fcvtzs, .fcvtzu, .scvtf, .ucvtf,
        .fcvtl, .fcvtn, .xtn, .uxtl, .sxtl,
        .vext, .vtrn, .vuzp, .vzip,
        .vrev16, .vrev32, .vrev64, .vdup,
        .vtbl, .vtbx,
        .vand, .vorr, .veor, .vbic, .vorn, .vbsl,
        => true,
        else => false,
    };
}

/// Compile a block of IR ops into x86-64 machine code via LLVM ORC JIT.
/// Returns the host address of the compiled code.
pub fn compileBlock(ops: []const IROp, guest_pc: u64) ?[]u8 {
    const s = jit orelse {
        std.log.err("LLVM not initialized", .{});
        return null;
    };

    // ── 1. Create LLVM module ─────────────────────────────────
    var mod_buf: [64]u8 = undefined;
    const mod_name = fmtZ(&mod_buf, "block_0x{X}", .{guest_pc});
    const module = LLVMModuleCreateWithName(mod_name);
    LLVMSetTarget(module, LLVMGetDefaultTargetTriple());
    const builder = LLVMCreateBuilder();

    const i64_ty = LLVMInt64Type();
    const i32_ty = LLVMInt32Type();
    const i8_ty = LLVMInt8Type();
    const f32_ty = LLVMFloatType();
    const v4f32_ty = LLVMVectorType(f32_ty, 4);  // 128-bit NEON vector
    const v4i32_ty = LLVMVectorType(i32_ty, 4);  // integer vector type for saturating ops

    // ── saSLP wider vector types (256-bit AVX) ────────────────────
    const v8f32_ty = LLVMVectorType(f32_ty, 8);  // 256-bit AVX vector

    // Shuffle masks for v4f32 ↔ v8f32 concatenation/splitting.
    // concat: shuffle two v4f32 into v8f32 (indices 0-3=low, 4-7=high)
    // split_low: extract v4f32 elements [0..3] from v8f32
    // split_high: extract v4f32 elements [4..7] from v8f32
    const concat8_mask = blk: {
        var vals: [8]ValueRef = undefined;
        for (&vals, 0..) |*v, j| v.* = LLVMConstInt(i32_ty, @as(c_ulonglong, @intCast(j)), 0);
        break :blk LLVMConstVector(&vals, 8);
    };
    const split4_low_mask = blk: {
        var vals: [4]ValueRef = undefined;
        for (&vals, 0..) |*v, j| v.* = LLVMConstInt(i32_ty, @as(c_ulonglong, @intCast(j)), 0);
        break :blk LLVMConstVector(&vals, 4);
    };
    const split4_high_mask = blk: {
        var vals: [4]ValueRef = undefined;
        for (&vals, 0..) |*v, j| v.* = LLVMConstInt(i32_ty, @as(c_ulonglong, @intCast(j + 4)), 0);
        break :blk LLVMConstVector(&vals, 4);
    };

    // fn(i64* %state, i64 %sp) -> i64   (returns taken branch target, 0 = not taken)
    var params = [_]TypeRef{ LLVMPointerType(i64_ty, 0), i64_ty };
    const fn_ty = LLVMFunctionType(i64_ty, &params, 2, 0);
    const fn_ = LLVMAddFunction(module, mod_name, fn_ty);
    const entry_bb = LLVMAppendBasicBlock(fn_, "entry");
    LLVMPositionBuilderAtEnd(builder, entry_bb);
    const state_ptr = LLVMGetParam(fn_, 0);
    _ = LLVMSetValueName(state_ptr, "state");
    const sp_init = LLVMGetParam(fn_, 1);
    _ = LLVMSetValueName(sp_init, "sp_init");
    // Global variable for SP — sp_get and sp_put both access this global.
    // LLVM knows a single global is a single memory location, so volatile
    // ordering is guaranteed (no aliasing ambiguity like inttoptr chains).
    const sp_global = LLVMAddGlobal(module, i64_ty, "__sp_jit");
    LLVMSetInitializer(sp_global, LLVMConstInt(i64_ty, 0, 0));
    // Initialize global SP from function argument
    const init_st = LLVMBuildStore(builder, sp_init, sp_global);
    LLVMSetVolatile(init_st, 1);

    // ── 2. Collect used regs ──────────────────────────────────
    var used: [31]bool = undefined;
    @memset(&used, false);
    var vec_used: [32]bool = undefined;
    @memset(&vec_used, false);
    var has_vec = false;
    for (ops) |op| {
        if (op.dest < 31) used[@as(usize, @intCast(op.dest))] = true;
        if (op.src0 < 31) used[@as(usize, @intCast(op.src0))] = true;
        if (op.src1 < 31) used[@as(usize, @intCast(op.src1))] = true;
        // Track v-reg usage for SIMD ops
        switch (op.tag) {
            .load_v128, .store_v128, .vadd, .vsub, .vmul, .vfadd, .vfmul,
            .vshl, .vshr, .fcvt => {
                has_vec = true;
                if (op.dest < 128 and op.dest >= 31) vec_used[@as(usize, @intCast(op.dest - 31))] = true;
                if (op.src0 < 128 and op.src0 >= 31) vec_used[@as(usize, @intCast(op.src0 - 31))] = true;
                if (op.src1 < 128 and op.src1 >= 31) vec_used[@as(usize, @intCast(op.src1 - 31))] = true;
            },
            else => {},
        }
    }

    // ── 3. Create allocas + load from state ───────────────────
    var reg_alloca: [31]?ValueRef = undefined;
    @memset(&reg_alloca, null);
    var vec_alloca: [32]?ValueRef = undefined;
    @memset(&vec_alloca, null);

    // Scalar regs (x0-x30) — existing logic
    for (&used, 0..) |u, i| {
        if (!u) continue;
        // Mark volatile for regs written by sp_get (x16) — prevents LLVM
        // from reordering the non-volatile init store over the volatile
        // sp_get store, which would zero out the SP value.
        const is_sp_get_dest = blk: {
            var found = false;
            for (ops) |op| { if (op.tag == .sp_get and op.dest == i) { found = true; break; } }
            break :blk found;
        };
        const arm_u: u16 = @intCast(i);
        var nbuf: [16]u8 = undefined;
        const aname = fmtZ(&nbuf, "x{}_ptr", .{arm_u});
        reg_alloca[i] = LLVMBuildAlloca(builder, i64_ty, aname);
        // Use sub(neg) instead of add to prevent LLVM from encoding
        // offsets 128-255 as signed byte (e.g. [rax+0x80] = [rax-128]).
        const state_i = LLVMBuildPtrToInt(builder, state_ptr, i64_ty, "si");
        const neg_off = @as(u64, @bitCast(@as(i64, -@as(i64, @intCast(i * 8)))));
        const addr_i = LLVMBuildSub(builder, state_i, LLVMConstInt(i64_ty, neg_off, 0), "ai");
        const gep = LLVMBuildIntToPtr(builder, addr_i, LLVMPointerType(i64_ty, 0), aname);
        const init_val = LLVMBuildLoad2(builder, i64_ty, gep, aname);
        const init_store = LLVMBuildStore(builder, init_val, reg_alloca[i].?);
        if (is_sp_get_dest) LLVMSetVolatile(init_store, 1);
    }

    // Vector regs (v0-v31) — at Arm64State offset ~288 (31*u64 + u64*3 + u32*3 + pad)
    if (has_vec) {
        // Compute v_base = state_ptr + 288 (byte offset of v[0] in Arm64State)
        const v_byte_off = LLVMConstInt(i64_ty, V_REG_OFFSET, 0);
        const state_i8 = LLVMBuildBitCast(builder, state_ptr, LLVMPointerType(i8_ty, 0), "state_i8");
        var vbo_indices = [_]ValueRef{v_byte_off};
        const v_base_i8 = LLVMBuildGEP2(builder, i8_ty, state_i8, &vbo_indices, 1, "v_base_i8");
        const v_base = LLVMBuildBitCast(builder, v_base_i8, LLVMPointerType(v4f32_ty, 0), "v_base");

        for (&vec_used, 0..) |u, i| {
            if (!u) continue;
            const vec_u: u16 = @intCast(i);
            var nbuf: [16]u8 = undefined;
            const vname = fmtZ(&nbuf, "v{}_ptr", .{vec_u});
            vec_alloca[i] = LLVMBuildAlloca(builder, v4f32_ty, vname);
            // Load from state.v[i]
            const v_idx = LLVMConstInt(i64_ty, i, 0);
            var vindices = [_]ValueRef{v_idx};
            const vgep = LLVMBuildGEP2(builder, v4f32_ty, v_base, &vindices, 1, vname);
            const init_v = LLVMBuildLoad2(builder, v4f32_ty, vgep, vname);
            _ = LLVMBuildStore(builder, init_v, vec_alloca[i].?);
        }
    }

    // ── 4. Translate IR ops (with saSLP SIMD combining) ─────
    var flag_state = FlagState{};
    var combine_skip: usize = std.math.maxInt(usize);
    for (ops, 0..) |op, i| {
        if (i == combine_skip) continue;

        // Try saSLP combining: if this op and the next form an independent
        // adjacent-vreg-pair with the same operation, emit a single v8f32
        // (256-bit) operation instead of two v4f32 (128-bit) operations.
        if (i + 1 < ops.len and tryCombineSimdPair(op, ops[i + 1])) {
            emitCombinedSimdOp(builder, &vec_alloca, op, ops[i + 1], v4f32_ty, v8f32_ty, concat8_mask, split4_low_mask, split4_high_mask);
            combine_skip = i + 1;
            continue;
        }

        emitOp(builder, module, sp_global, &reg_alloca, &vec_alloca, op, i64_ty, i32_ty, v4f32_ty, v4i32_ty, f32_ty, &flag_state) catch continue;
    }

    // ── 4b. Terminal conditional branch ──────────────────────
    // A region ending in br_cond returns the taken target (nonzero) or 0
    // (not taken) so the runtime can dispatch the .cond successor.
    var ret_val = LLVMConstInt(i64_ty, 0, 0);
    if (ops.len > 0 and ops[ops.len - 1].tag == .br_cond) {
        const br = ops[ops.len - 1];
        const n = flag_state.n orelse LLVMConstInt(LLVMInt1Type(), 0, 0);
        const z = flag_state.z orelse LLVMConstInt(LLVMInt1Type(), 0, 0);
        const c = flag_state.c orelse LLVMConstInt(LLVMInt1Type(), 0, 0);
        const v = flag_state.v orelse LLVMConstInt(LLVMInt1Type(), 0, 0);
        const true1 = LLVMConstInt(LLVMInt1Type(), 1, 0);
        const cond_i1: ValueRef = switch (br.flags & 0xF) {
            0b0000 => z, // EQ
            0b0001 => LLVMBuildNot(builder, z, "cne"), // NE
            0b0010 => c, // CS/HS
            0b0011 => LLVMBuildNot(builder, c, "ccc"), // CC/LO
            0b0100 => n, // MI
            0b0101 => LLVMBuildNot(builder, n, "cpl"), // PL
            0b0110 => v, // VS
            0b0111 => LLVMBuildNot(builder, v, "cvc"), // VC
            0b1000 => LLVMBuildAnd(builder, c, LLVMBuildNot(builder, z, "cnz"), "chi"), // HI
            0b1001 => LLVMBuildOr(builder, LLVMBuildNot(builder, c, "cnc"), z, "cls"), // LS
            0b1010 => LLVMBuildICmp(builder, LLVMIntEQ, n, v, "cge"), // GE
            0b1011 => LLVMBuildICmp(builder, LLVMIntNE, n, v, "clt"), // LT
            0b1100 => LLVMBuildAnd(builder, LLVMBuildNot(builder, z, "cnz2"), LLVMBuildICmp(builder, LLVMIntEQ, n, v, "cge2"), "cgt"), // GT
            0b1101 => LLVMBuildOr(builder, z, LLVMBuildICmp(builder, LLVMIntNE, n, v, "clt2"), "cle"), // LE
            0b1110 => true1, // AL
            else => null,
        } orelse LLVMConstInt(LLVMInt1Type(), 0, 0);
        const target = LLVMConstInt(i64_ty, br.imm, 0);
        ret_val = LLVMBuildSelect(builder, cond_i1, target, ret_val, "nextpc");
    }

    // ── 5. Store back to state ───────────────────────────────
    // Write SP global back to state.sp
    {
        const sp_val = LLVMBuildLoad2(builder, i64_ty, sp_global, "sp_exit");
        const si = LLVMBuildPtrToInt(builder, state_ptr, i64_ty, "se_si");
        const neg248 = LLVMConstInt(i64_ty, @as(u64, @bitCast(@as(i64, -248))), 0);
        const addr = LLVMBuildSub(builder, si, neg248, "se_ai");
        const ptr = LLVMBuildIntToPtr(builder, addr, LLVMPointerType(i64_ty, 0), "se_p");
        _ = LLVMBuildStore(builder, sp_val, ptr);
    }
    // Scalar regs (use ptrtoint+add+inttoptr, not GEP, to prevent
    // LLVM ORC JIT signed-byte displacement issue for offsets ≥128)
    for (&used, 0..) |u, i| {
        if (!u) continue;
        const alloca = reg_alloca[i] orelse continue;
        var nbuf: [16]u8 = undefined;
        const sname = fmtZ(&nbuf, "sx{}", .{@as(u16, @intCast(i))});
        const val = LLVMBuildLoad2(builder, i64_ty, alloca, sname);
        const state_si = LLVMBuildPtrToInt(builder, state_ptr, i64_ty, "ssi");
        const store_neg = @as(u64, @bitCast(@as(i64, -@as(i64, @intCast(i * 8)))));
        const store_addr = LLVMBuildSub(builder, state_si, LLVMConstInt(i64_ty, store_neg, 0), "sa");
        const gep = LLVMBuildIntToPtr(builder, store_addr, LLVMPointerType(i64_ty, 0), sname);
        _ = LLVMBuildStore(builder, val, gep);
    }

    // Vector regs
    if (has_vec) {
        const v_byte_off2 = LLVMConstInt(i64_ty, V_REG_OFFSET, 0);
        const state_i8_2 = LLVMBuildBitCast(builder, state_ptr, LLVMPointerType(i8_ty, 0), "state_i8_s");
        var vbo_indices2 = [_]ValueRef{v_byte_off2};
        const v_base_i8 = LLVMBuildGEP2(builder, i8_ty, state_i8_2, &vbo_indices2, 1, "v_base_i8_s");
        const v_base = LLVMBuildBitCast(builder, v_base_i8, LLVMPointerType(v4f32_ty, 0), "v_base_s");
        for (&vec_used, 0..) |u, i| {
            if (!u) continue;
            const alloca = vec_alloca[i] orelse continue;
            var nbuf: [16]u8 = undefined;
            const vsname = fmtZ(&nbuf, "sv{}", .{@as(u16, @intCast(i))});
            const val = LLVMBuildLoad2(builder, v4f32_ty, alloca, vsname);
            const v_idx = LLVMConstInt(i64_ty, i, 0);
            var vindices = [_]ValueRef{v_idx};
            const vgep = LLVMBuildGEP2(builder, v4f32_ty, v_base, &vindices, 1, vsname);
            _ = LLVMBuildStore(builder, val, vgep);
        }
    }
    _ = LLVMBuildRet(builder, ret_val);
    if (getenv("A64TOX64_DUMPLLVM") != null) {
        var err: ?[*:0]u8 = null;
        _ = LLVMPrintModuleToFile(module, "/tmp/llvm_mod.ll", &err);
    }
    _ = LLVMVerifyFunction(fn_, LLVMReturnStatusAction);
    LLVMDisposeBuilder(builder);

    // ── 6. Add to ORC JIT ────────────────────────────────────
    const main_jd = LLVMOrcLLJITGetMainJITDylib(s.jit_ref);
    // Must use ThreadSafeContext for LLVM 22+ ORC JIT.
    const local_ctx = LLVMContextCreate();
    const tsc = LLVMOrcCreateNewThreadSafeContextFromLLVMContext(local_ctx);
    const tsm = LLVMOrcCreateNewThreadSafeModule(module, tsc);
    if (LLVMOrcLLJITAddLLVMIRModule(s.jit_ref, main_jd, tsm)) |e| {
        const msg = LLVMGetErrorMessage(e);
        defer LLVMDisposeErrorMessage(msg);
        std.log.err("LLVM add module: {s}", .{msg});
        return null;
    }

    // Module is consumed by ORC JIT once added via ThreadSafeModule.
    // The ThreadSafeModule (tsm) takes ownership of both module and context.
    {
        const js = &jit.?;
        js.module_count += 1;
        if (js.module_count > 10) {
            std.log.warn("LLVM: {} modules compiled — JIT memory may grow unbounded", .{js.module_count});
        }
    }

    // ── 7. Look up compiled address ──────────────────────────
    var fn_addr: OrcJITTargetAddress = 0;
    if (LLVMOrcLLJITLookup(s.jit_ref, &fn_addr, mod_name)) |e| {
        const msg = LLVMGetErrorMessage(e);
        defer LLVMDisposeErrorMessage(msg);
        std.log.err("LLVM lookup: {s}", .{msg});
        return null;
    }
    if (fn_addr == 0) return null;
    // Estimate code size from IR ops (LLVM ~12 bytes/op avg)
    const code_sz = @min(@as(usize, @intCast(ops.len * 16 + 128)), 4096);
    if (getenv("A64TOX64_DUMPLLVM") != null) {
        const n2 = @as(usize, 4096);
        std.debug.print("LLVM block 0x{X:016} host=0x{X:016}:", .{ guest_pc, fn_addr });
        for (@as([*]u8, @ptrFromInt(fn_addr))[0..n2]) |bb| std.debug.print(" {X:0>2}", .{bb});
        std.debug.print("\n", .{});
    }
        return @as([*]u8, @ptrFromInt(fn_addr))[0..code_sz];
}

// ── IR op emission ──────────────────────────────────────────

/// Get or create an LLVM intrinsic function in the module.
fn getIntrinsic(mod: ModuleRef, name: [*:0]const u8, ret_ty: TypeRef, param_tys: []const TypeRef) ValueRef {
    if (LLVMGetNamedFunction(mod, name)) |f| return f;
    const fn_ty = LLVMFunctionType(ret_ty, param_tys.ptr, @as(c_uint, @intCast(param_tys.len)), 0);
    const f = LLVMAddFunction(mod, name, fn_ty);
    _ = LLVMSetValueName(f, name);
    return f;
}

/// Tracked state for ARM64 NZCV flag computation (needed to evaluate
/// conditional branches, which the hand emitter does via x86 flags and the
/// LLVM backend must model explicitly). `result`/`lhs`/`rhs` are the LLVM
/// values of the most recent ALU op; `kind` says whether it was a subtract
/// (C = no borrow), add (C = carry out) or logical (C/V unchanged → 0).
/// `nzcv` holds the computed flag values once the nzcv_update op runs.
const FlagState = struct {
    lhs: ?ValueRef = null,
    rhs: ?ValueRef = null,
    result: ?ValueRef = null,
    kind: enum { none, add, sub, logical } = .none,
    n: ?ValueRef = null,
    z: ?ValueRef = null,
    c: ?ValueRef = null,
    v: ?ValueRef = null,
};

fn emitOp(
    b: BuilderRef,
    mod: ModuleRef,
    sp_global: ValueRef,
    regs: *[31]?ValueRef,
    vecs: *[32]?ValueRef,
    op: IROp,
    i64_ty: TypeRef,
    i32_ty: TypeRef,
    v4f32_ty: TypeRef,
    v4i32_ty: TypeRef,
    f32_ty: TypeRef,
    flags: *FlagState,
) !void {
    const ty: TypeRef = switch (op.tag) {
        .add_i32, .sub_i32, .mul_i32 => i32_ty,
        else => i64_ty,
    };

    switch (op.tag) {
        .add_i64, .add_i32 => emitBinop(b, regs, op, ty, LLVMBuildAdd, flags),
        .sub_i64, .sub_i32 => emitBinop(b, regs, op, ty, LLVMBuildSub, flags),
        .mul_i64, .mul_i32 => emitBinop(b, regs, op, ty, LLVMBuildMul, flags),
        .and_ => emitBinop(b, regs, op, ty, LLVMBuildAnd, flags),
        .or_ => emitBinop(b, regs, op, ty, LLVMBuildOr, flags),
        .xor_ => emitBinop(b, regs, op, ty, LLVMBuildXor, flags),
        .div_u64 => emitBinop(b, regs, op, ty, LLVMBuildUDiv, flags),
        .div_s64 => emitBinop(b, regs, op, ty, LLVMBuildSDiv, flags),
        .not_ => emitUnop(b, regs, op, ty, LLVMBuildNot),
        .neg_i64 => emitUnop(b, regs, op, ty, LLVMBuildNeg),
        .lshl_i64, .lshl_i64_imm => emitBinop(b, regs, op, ty, LLVMBuildShl, flags),
        .lshr_i64, .lshr_i64_imm => emitBinop(b, regs, op, ty, LLVMBuildLShr, flags),
        .ashr_i64, .ashr_i64_imm => emitBinop(b, regs, op, ty, LLVMBuildAShr, flags),

        .nzcv_update => {
            // Compute ARM64 N/Z/C/V from the most recent ALU op so a later
            // br_cond can be evaluated. imm=1 (CMC) means the last op was a
            // subtract (C = !borrow); imm=0 add/logical (C = carry out / 0).
            const res = flags.result orelse return;
            const lhs = flags.lhs orelse return;
            const rhs = flags.rhs orelse return;
            const zero64 = LLVMConstInt(i64_ty, 0, 0);
            flags.n = LLVMBuildICmp(b, LLVMIntSLT, res, zero64, "n");
            flags.z = LLVMBuildICmp(b, LLVMIntEQ, res, zero64, "z");
            if (flags.kind == .sub) {
                // C = no borrow = lhs u>= result
                flags.c = LLVMBuildICmp(b, LLVMIntUGE, lhs, res, "c");
                // V = overflow = (lhs^rhs) & (result^lhs) sign bit
                const x1 = LLVMBuildXor(b, lhs, rhs, "vx1");
                const x2 = LLVMBuildXor(b, res, lhs, "vx2");
                const a = LLVMBuildAnd(b, x1, x2, "va");
                flags.v = LLVMBuildICmp(b, LLVMIntSLT, a, zero64, "v");
            } else if (flags.kind == .add) {
                // C = carry out = result u< lhs
                flags.c = LLVMBuildICmp(b, LLVMIntULT, res, lhs, "c");
                // V = overflow = (lhs^result) & (rhs^result) sign bit
                const x1 = LLVMBuildXor(b, lhs, res, "vx1");
                const x2 = LLVMBuildXor(b, rhs, res, "vx2");
                const a = LLVMBuildAnd(b, x1, x2, "va");
                flags.v = LLVMBuildICmp(b, LLVMIntSLT, a, zero64, "v");
            } else {
                // logical: C and V unchanged per ARM (approximated as 0)
                flags.c = LLVMConstInt(LLVMInt1Type(), 0, 0);
                flags.v = LLVMConstInt(LLVMInt1Type(), 0, 0);
            }
        },

        .mov_i64 => {
            if (op.src0 >= 31) { // XZR → zero immediate
                const dst = regs[@as(usize, @intCast(op.dest))] orelse return;
                _ = LLVMBuildStore(b, LLVMConstInt(ty, 0, 0), dst);
            } else {
                const src = regs[@as(usize, @intCast(op.src0))] orelse return;
                const dst = regs[@as(usize, @intCast(op.dest))] orelse return;
                const val = LLVMBuildLoad2(b, ty, src, "mv");
                _ = LLVMBuildStore(b, val, dst);
            }
        },

        .load_u64 => {
            if (op.src0 >= 31) return; // XZR base → no-op
            const base = regs[@as(usize, @intCast(op.src0))] orelse return;
            const dst = regs[@as(usize, @intCast(op.dest))] orelse return;
            const base_val = LLVMBuildLoad2(b, i64_ty, base, "base");
            const off = LLVMConstInt(i64_ty, @as(u64, @bitCast(@as(i64, @intCast(@as(i32, @bitCast(op.imm)))))), 0);
            const addr = LLVMBuildAdd(b, base_val, off, "addr");
            const ptr = LLVMBuildIntToPtr(b, addr, LLVMPointerType(i64_ty, 0), "ptr");
            const val = LLVMBuildLoad2(b, i64_ty, ptr, "ld");
            _ = LLVMBuildStore(b, val, dst);
        },

        .load_u32 => {
            if (op.src0 >= 31) return;
            const base = regs[@as(usize, @intCast(op.src0))] orelse return;
            const dst = regs[@as(usize, @intCast(op.dest))] orelse return;
            const base_val = LLVMBuildLoad2(b, i64_ty, base, "base");
            const off = LLVMConstInt(i64_ty, @as(u64, @bitCast(@as(i64, @intCast(@as(i32, @bitCast(op.imm)))))), 0);
            const addr = LLVMBuildAdd(b, base_val, off, "addr");
            const ptr = LLVMBuildIntToPtr(b, addr, LLVMPointerType(i32_ty, 0), "ptr32");
            const val = LLVMBuildLoad2(b, i32_ty, ptr, "ld32");
            _ = LLVMBuildStore(b, LLVMBuildZExt(b, val, i64_ty, "zext"), dst);
        },

        .load_u16 => {
            if (op.src0 >= 31) return;
            const base = regs[@as(usize, @intCast(op.src0))] orelse return;
            const dst = regs[@as(usize, @intCast(op.dest))] orelse return;
            const base_val = LLVMBuildLoad2(b, i64_ty, base, "base");
            const off = LLVMConstInt(i64_ty, @as(u64, @bitCast(@as(i64, @intCast(@as(i32, @bitCast(op.imm)))))), 0);
            const addr = LLVMBuildAdd(b, base_val, off, "addr");
            const ptr = LLVMBuildIntToPtr(b, addr, LLVMPointerType(LLVMInt16Type(), 0), "ptr16");
            const val = LLVMBuildLoad2(b, LLVMInt16Type(), ptr, "ld16");
            _ = LLVMBuildStore(b, LLVMBuildZExt(b, val, i64_ty, "zext16"), dst);
        },

        .load_u8 => {
            if (op.src0 >= 31) return;
            const base = regs[@as(usize, @intCast(op.src0))] orelse return;
            const dst = regs[@as(usize, @intCast(op.dest))] orelse return;
            const base_val = LLVMBuildLoad2(b, i64_ty, base, "base");
            const off = LLVMConstInt(i64_ty, @as(u64, @bitCast(@as(i64, @intCast(@as(i32, @bitCast(op.imm)))))), 0);
            const addr = LLVMBuildAdd(b, base_val, off, "addr");
            const ptr = LLVMBuildIntToPtr(b, addr, LLVMPointerType(LLVMInt8Type(), 0), "ptr8");
            const val = LLVMBuildLoad2(b, LLVMInt8Type(), ptr, "ld8");
            _ = LLVMBuildStore(b, LLVMBuildZExt(b, val, i64_ty, "zext8"), dst);
        },

        .store_u64 => {
            if (op.src0 >= 31) return; // XZR base → no-op
            const base = regs[@as(usize, @intCast(op.src0))] orelse return;
            const src = regs[@as(usize, @intCast(op.src1))] orelse return;
            const base_val = LLVMBuildLoad2(b, i64_ty, base, "base");
            const src_val = LLVMBuildLoad2(b, i64_ty, src, "src");
            const off = LLVMConstInt(i64_ty, @as(u64, @bitCast(@as(i64, @intCast(@as(i32, @bitCast(op.imm)))))), 0);
            const addr = LLVMBuildAdd(b, base_val, off, "addr");
            const ptr = LLVMBuildIntToPtr(b, addr, LLVMPointerType(i64_ty, 0), "ptr");
            _ = LLVMBuildStore(b, src_val, ptr);
        },

        .store_u32 => {
            if (op.src0 >= 31) return;
            const base = regs[@as(usize, @intCast(op.src0))] orelse return;
            const src = regs[@as(usize, @intCast(op.src1))] orelse return;
            const base_val = LLVMBuildLoad2(b, i64_ty, base, "base");
            const src_val = LLVMBuildLoad2(b, i64_ty, src, "src");
            const off = LLVMConstInt(i64_ty, @as(u64, @bitCast(@as(i64, @intCast(@as(i32, @bitCast(op.imm)))))), 0);
            const addr = LLVMBuildAdd(b, base_val, off, "addr");
            const ptr = LLVMBuildIntToPtr(b, addr, LLVMPointerType(i32_ty, 0), "ptr32");
            _ = LLVMBuildStore(b, LLVMBuildTrunc(b, src_val, i32_ty, "tr32"), ptr);
        },

        .store_u16 => {
            if (op.src0 >= 31) return;
            const base = regs[@as(usize, @intCast(op.src0))] orelse return;
            const src = regs[@as(usize, @intCast(op.src1))] orelse return;
            const base_val = LLVMBuildLoad2(b, i64_ty, base, "base");
            const src_val = LLVMBuildLoad2(b, i64_ty, src, "src");
            const off = LLVMConstInt(i64_ty, @as(u64, @bitCast(@as(i64, @intCast(@as(i32, @bitCast(op.imm)))))), 0);
            const addr = LLVMBuildAdd(b, base_val, off, "addr");
            const ptr = LLVMBuildIntToPtr(b, addr, LLVMPointerType(LLVMInt16Type(), 0), "ptr16");
            _ = LLVMBuildStore(b, LLVMBuildTrunc(b, src_val, LLVMInt16Type(), "tr16"), ptr);
        },

        .store_u8 => {
            if (op.src0 >= 31) return;
            const base = regs[@as(usize, @intCast(op.src0))] orelse return;
            const src = regs[@as(usize, @intCast(op.src1))] orelse return;
            const base_val = LLVMBuildLoad2(b, i64_ty, base, "base");
            const src_val = LLVMBuildLoad2(b, i64_ty, src, "src");
            const off = LLVMConstInt(i64_ty, @as(u64, @bitCast(@as(i64, @intCast(@as(i32, @bitCast(op.imm)))))), 0);
            const addr = LLVMBuildAdd(b, base_val, off, "addr");
            const ptr = LLVMBuildIntToPtr(b, addr, LLVMPointerType(LLVMInt8Type(), 0), "ptr8");
            _ = LLVMBuildStore(b, LLVMBuildTrunc(b, src_val, LLVMInt8Type(), "tr8"), ptr);
        },

        .br, .ret_ => {},

        .sp_get => {
            const dst = regs[@as(usize, @intCast(op.dest))] orelse return;
            const ld = LLVMBuildLoad2(b, i64_ty, sp_global, "spv");
            LLVMSetVolatile(ld, 1);
            _ = LLVMBuildStore(b, ld, dst);
        },
        .sp_put => {
            const src = regs[@as(usize, @intCast(op.src0))] orelse return;
            const sp_new = LLVMBuildLoad2(b, i64_ty, src, "spn");
            const st = LLVMBuildStore(b, sp_new, sp_global);
            LLVMSetVolatile(st, 1);
        },

        // ── SIMD ops ──────────────────────────────────────────────
        .load_v128 => {
            const base = regs[@as(usize, @intCast(op.src0))] orelse return;
            const vdst = vecs[@as(usize, @intCast(op.dest - 31))] orelse return;
            const base_val = LLVMBuildLoad2(b, i64_ty, base, "base");
            const off = LLVMConstInt(i64_ty, @as(u64, @bitCast(@as(i64, @intCast(@as(i32, @bitCast(op.imm)))))), 0);
            const addr = LLVMBuildAdd(b, base_val, off, "addr");
            const ptr = LLVMBuildIntToPtr(b, addr, LLVMPointerType(v4f32_ty, 0), "vptr");
            const loaded = LLVMBuildLoad2(b, v4f32_ty, ptr, "vld");
            _ = LLVMBuildStore(b, loaded, vdst);
        },

        .store_v128 => {
            const base = regs[@as(usize, @intCast(op.src0))] orelse return;
            const vsrc = vecs[@as(usize, @intCast(op.src1 - 31))] orelse return;
            const base_val = LLVMBuildLoad2(b, i64_ty, base, "base");
            const src_val = LLVMBuildLoad2(b, v4f32_ty, vsrc, "vsrc");
            const off = LLVMConstInt(i64_ty, @as(u64, @bitCast(@as(i64, @intCast(@as(i32, @bitCast(op.imm)))))), 0);
            const addr = LLVMBuildAdd(b, base_val, off, "addr");
            const ptr = LLVMBuildIntToPtr(b, addr, LLVMPointerType(v4f32_ty, 0), "vptr");
            _ = LLVMBuildStore(b, src_val, ptr);
        },

        .vadd, .vsub, .vmul, .vfadd, .vfmul => {
            const vdst = vecs[@as(usize, @intCast(op.dest - 31))] orelse return;
            const vs0 = vecs[@as(usize, @intCast(op.src0 - 31))] orelse return;
            const is_float = op.tag == .vfadd or op.tag == .vfmul;
            const lhs = LLVMBuildLoad2(b, v4f32_ty, vs0, "vlhs");
            if (op.src1 >= 31 and op.src1 < 128) {
                const vs1 = vecs[@as(usize, @intCast(op.src1 - 31))] orelse return;
                const rhs = LLVMBuildLoad2(b, v4f32_ty, vs1, "vrhs");
                const result = if (is_float) switch (op.tag) {
                    .vfadd => LLVMBuildFAdd(b, lhs, rhs, "vres"),
                    .vfmul => LLVMBuildFMul(b, lhs, rhs, "vres"),
                    else => unreachable,
                } else switch (op.tag) {
                    .vadd => LLVMBuildAdd(b, lhs, rhs, "vres"),
                    .vsub => LLVMBuildSub(b, lhs, rhs, "vres"),
                    .vmul => LLVMBuildMul(b, lhs, rhs, "vres"),
                    else => unreachable,
                };
                _ = LLVMBuildStore(b, result, vdst);
            } else {
                _ = LLVMBuildStore(b, lhs, vdst);
            }
        },

        .fcvt => {
            const vdst = vecs[@as(usize, @intCast(op.dest - 31))] orelse return;
            const vs0 = vecs[@as(usize, @intCast(op.src0 - 31))] orelse return;
            const val = LLVMBuildLoad2(b, v4f32_ty, vs0, "vfcvt");
            const result = LLVMBuildFPCast(b, val, v4f32_ty, "vfcvt_res");
            _ = LLVMBuildStore(b, result, vdst);
        },

        // ── SIMD logical (bitwise ops on vectors) ─────────────────
        .vand => try emitVecBinop(b, vecs, op, v4f32_ty, LLVMBuildAnd),
        .vorr => try emitVecBinop(b, vecs, op, v4f32_ty, LLVMBuildOr),
        .veor => try emitVecBinop(b, vecs, op, v4f32_ty, LLVMBuildXor),
        .vbic => {
            // VBIC = AND(a, NOT(b))
            const vs0 = vecs[@as(usize, @intCast(op.src0 - 31))] orelse return;
            const vs1 = vecs[@as(usize, @intCast(op.src1 - 31))] orelse return;
            const vdst = vecs[@as(usize, @intCast(op.dest - 31))] orelse return;
            const lhs = LLVMBuildLoad2(b, v4f32_ty, vs0, "vlhs");
            const rhs = LLVMBuildLoad2(b, v4f32_ty, vs1, "vrhs");
            const not_rhs = LLVMBuildNot(b, rhs, "vnot");
            const result = LLVMBuildAnd(b, lhs, not_rhs, "vbic");
            _ = LLVMBuildStore(b, result, vdst);
        },
        .vneg => {
            const vs0 = vecs[@as(usize, @intCast(op.src0 - 31))] orelse return;
            const vdst = vecs[@as(usize, @intCast(op.dest - 31))] orelse return;
            const val = LLVMBuildLoad2(b, v4f32_ty, vs0, "vv");
            const result = LLVMBuildNeg(b, val, "vneg");
            _ = LLVMBuildStore(b, result, vdst);
        },
        .vabs => {
            const vs0 = vecs[@as(usize, @intCast(op.src0 - 31))] orelse return;
            const vdst = vecs[@as(usize, @intCast(op.dest - 31))] orelse return;
            const val = LLVMBuildLoad2(b, v4f32_ty, vs0, "vv");
            const zero = LLVMConstNull(v4f32_ty);
            const cmp = LLVMBuildFCmp(b, LLVMRealOLT, val, zero, "vcmp"); // <4 x i1>
            const neg = LLVMBuildNeg(b, val, "vneg");
            const result = LLVMBuildSelect(b, cmp, neg, val, "vabs");
            _ = LLVMBuildStore(b, result, vdst);
        },
        .vceq => {
            try emitVecCmp(b, vecs, op, v4f32_ty, LLVMIntEQ);
        },
        .vcgt => {
            try emitVecCmp(b, vecs, op, v4f32_ty, LLVMIntSGT);
        },
        .vcge => {
            try emitVecCmp(b, vecs, op, v4f32_ty, LLVMIntSGE);
        },
        .vmin => {
            try emitVecMinMax(b, vecs, op, v4f32_ty, LLVMRealOLT); // min: select(a<b, a, b)
        },
        .vmax => {
            try emitVecMinMax(b, vecs, op, v4f32_ty, LLVMRealOGT); // max: select(a>b, a, b)
        },
        .vbsl => {
            // Bitwise select: result = (src1 & src0) | (src0_other & ~src0)
            // Using 3-operand: dest = (src1 & mask) | (src0 & ~mask)
            // mask is in src0, first data in src1, second data in dest
            // For correct BSL: dest = (src0 & mask) | (src1 & ~mask) where mask=src0
            const vs_mask = vecs[@as(usize, @intCast(op.src0 - 31))] orelse return;
            const vs0 = vecs[@as(usize, @intCast(op.src1 - 31))] orelse return;
            const vdst = vecs[@as(usize, @intCast(op.dest - 31))] orelse return;
            const vm = LLVMBuildLoad2(b, v4f32_ty, vs_mask, "vmask");
            const v0 = LLVMBuildLoad2(b, v4f32_ty, vs0, "v0");
            const v1 = LLVMBuildLoad2(b, v4f32_ty, vdst, "v1");
            const not_vm = LLVMBuildNot(b, vm, "vnot");
            const and0 = LLVMBuildAnd(b, vm, v0, "va0");
            const and1 = LLVMBuildAnd(b, not_vm, v1, "va1");
            const result = LLVMBuildOr(b, and0, and1, "vbsl");
            _ = LLVMBuildStore(b, result, vdst);
        },
        .vmla => {
            // MLA = src0 + src1 * dest (multiply-add: acc += a*b)
            try emitVecMla(b, vecs, op, v4f32_ty, false);
        },
        .vmls => {
            // MLS = src0 - src1 * dest
            try emitVecMla(b, vecs, op, v4f32_ty, true);
        },
        .vfmin => {
            try emitVecMinMax(b, vecs, op, v4f32_ty, LLVMRealOLT);
        },
        .vfmax => {
            try emitVecMinMax(b, vecs, op, v4f32_ty, LLVMRealOGT);
        },
        .vfsub => {
            try emitVecBinop(b, vecs, op, v4f32_ty, LLVMBuildFSub);
        },
        .vfdiv => {
            try emitVecBinop(b, vecs, op, v4f32_ty, LLVMBuildFDiv);
        },
        .vfabs => {
            const vs0 = vecs[@as(usize, @intCast(op.src0 - 31))] orelse return;
            const vdst = vecs[@as(usize, @intCast(op.dest - 31))] orelse return;
            const val = LLVMBuildLoad2(b, v4f32_ty, vs0, "vv");
            const zero = LLVMConstNull(v4f32_ty);
            const cmp = LLVMBuildFCmp(b, LLVMRealOLT, val, zero, "vfcmp");
            const neg = LLVMBuildNeg(b, val, "vneg");
            const result = LLVMBuildSelect(b, cmp, neg, val, "vfabs");
            _ = LLVMBuildStore(b, result, vdst);
        },
        .vfneg => {
            const vs0 = vecs[@as(usize, @intCast(op.src0 - 31))] orelse return;
            const vdst = vecs[@as(usize, @intCast(op.dest - 31))] orelse return;
            const val = LLVMBuildLoad2(b, v4f32_ty, vs0, "vv");
            const result = LLVMBuildNeg(b, val, "vfneg");
            _ = LLVMBuildStore(b, result, vdst);
        },

        // ── SIMD shift ────────────────────────────────────────────
        .vshl => try emitVecBinop(b, vecs, op, v4f32_ty, LLVMBuildShl),
        .vshr => try emitVecBinop(b, vecs, op, v4f32_ty, LLVMBuildLShr),
        .vsra => {
            // ARM64 SRA: Vd = Vd + (Vn >> imm)
            // dst is accumulator AND dest. src0 (vs0) is the value to shift.
            // Shift amount from op.imm.
            const vs0 = vecs[@as(usize, @intCast(op.src0 - 31))] orelse return;
            const vdst = vecs[@as(usize, @intCast(op.dest - 31))] orelse return;
            const val = LLVMBuildLoad2(b, v4f32_ty, vs0, "vsra_val");
            const shift_scalar = LLVMConstInt(i32_ty, @as(c_ulonglong, @intCast(op.imm & 0x3F)), 0);
            const shift_vec_i32 = LLVMConstVector(&[4]ValueRef{ shift_scalar, shift_scalar, shift_scalar, shift_scalar }, 4);
            const val_i32 = LLVMBuildBitCast(b, val, v4i32_ty, "vsra_val_i32");
            const shifted_i32 = LLVMBuildLShr(b, val_i32, shift_vec_i32, "vsra_sh");
            const shifted = LLVMBuildBitCast(b, shifted_i32, v4f32_ty, "vsra_sh_f");
            const acc = LLVMBuildLoad2(b, v4f32_ty, vdst, "vsra_acc");
            const result = LLVMBuildAdd(b, acc, shifted, "vsra");
            _ = LLVMBuildStore(b, result, vdst);
        },
        .vshrn => {
            // SHRN: narrow after shift: dest = trunc(lshr(src, imm))
            const vs0 = vecs[@as(usize, @intCast(op.src0 - 31))] orelse return;
            const vdst = vecs[@as(usize, @intCast(op.dest - 31))] orelse return;
            const val = LLVMBuildLoad2(b, v4f32_ty, vs0, "vshrn_val");
            if (op.imm != 0) {
                const shift_amt = LLVMConstInt(i32_ty, @as(c_ulonglong, @intCast(op.imm & 0x3F)), 0);
                const shift_vec_i32 = LLVMConstVector(&[4]ValueRef{ shift_amt, shift_amt, shift_amt, shift_amt }, 4);
                const val_i32 = LLVMBuildBitCast(b, val, v4i32_ty, "vshrn_val_i32");
                const shifted_i32 = LLVMBuildLShr(b, val_i32, shift_vec_i32, "vshrn_sh");
                const shifted = LLVMBuildBitCast(b, shifted_i32, v4f32_ty, "vshrn_sh_f");
                _ = LLVMBuildStore(b, shifted, vdst);
            } else {
                _ = LLVMBuildStore(b, val, vdst);
            }
        },

        // ── SIMD convert ───────────────────────────────────────────
        .fcvtzs => {
            try emitVecConvert(b, vecs, op, v4f32_ty, LLVMBuildFPToSI, i32_ty, f32_ty, i32_ty);
        },
        .fcvtzu => {
            try emitVecConvert(b, vecs, op, v4f32_ty, LLVMBuildFPToUI, i32_ty, f32_ty, i32_ty);
        },
        .scvtf => {
            try emitVecConvert(b, vecs, op, v4f32_ty, LLVMBuildSIToFP, i32_ty, f32_ty, i32_ty);
        },
        .ucvtf => {
            try emitVecConvert(b, vecs, op, v4f32_ty, LLVMBuildUIToFP, i32_ty, f32_ty, i32_ty);
        },
        .xtn => {
            // Narrow: cast vector to smaller element type (via truncation)
            const vs0 = vecs[@as(usize, @intCast(op.src0 - 31))] orelse return;
            const vdst = vecs[@as(usize, @intCast(op.dest - 31))] orelse return;
            const val = LLVMBuildLoad2(b, v4f32_ty, vs0, "vxtn");
            _ = LLVMBuildStore(b, val, vdst); // pass through
        },
        .uxtl => {
            // Long (unsigned): zero-extend to wider type
            const vs0 = vecs[@as(usize, @intCast(op.src0 - 31))] orelse return;
            const vdst = vecs[@as(usize, @intCast(op.dest - 31))] orelse return;
            const val = LLVMBuildLoad2(b, v4f32_ty, vs0, "vuxtl");
            _ = LLVMBuildStore(b, val, vdst); // pass through (no-op for same type)
        },
        .sxtl => {
            const vs0 = vecs[@as(usize, @intCast(op.src0 - 31))] orelse return;
            const vdst = vecs[@as(usize, @intCast(op.dest - 31))] orelse return;
            const val = LLVMBuildLoad2(b, v4f32_ty, vs0, "vsxtl");
            _ = LLVMBuildStore(b, val, vdst);
        },

        // ── SIMD remaining implemented ops ─────────────────────────
        .vorn => {
            // OR-NOT: dest = src0 | ~src1
            const vs0 = vecs[@as(usize, @intCast(op.src0 - 31))] orelse return;
            const vs1 = vecs[@as(usize, @intCast(op.src1 - 31))] orelse return;
            const vdst = vecs[@as(usize, @intCast(op.dest - 31))] orelse return;
            const lhs = LLVMBuildLoad2(b, v4f32_ty, vs0, "vlhs");
            const rhs = LLVMBuildLoad2(b, v4f32_ty, vs1, "vrhs");
            const not_rhs = LLVMBuildNot(b, rhs, "vnot");
            const result = LLVMBuildOr(b, lhs, not_rhs, "vorn");
            _ = LLVMBuildStore(b, result, vdst);
        },
        .vabd => {
            // ABD = select(a > b, a-b, b-a) = abs(a-b)
            const vs0 = vecs[@as(usize, @intCast(op.src0 - 31))] orelse return;
            const vs1 = vecs[@as(usize, @intCast(op.src1 - 31))] orelse return;
            const vdst = vecs[@as(usize, @intCast(op.dest - 31))] orelse return;
            const lhs = LLVMBuildLoad2(b, v4f32_ty, vs0, "vabd_l");
            const rhs = LLVMBuildLoad2(b, v4f32_ty, vs1, "vabd_r");
            const sub1 = LLVMBuildSub(b, lhs, rhs, "vabd_s1");
            const sub2 = LLVMBuildSub(b, rhs, lhs, "vabd_s2");
            const cmp = LLVMBuildFCmp(b, LLVMRealOGT, lhs, rhs, "vabd_c");
            const result = LLVMBuildSelect(b, cmp, sub1, sub2, "vabd");
            _ = LLVMBuildStore(b, result, vdst);
        },
        .vsri => {
            // SRI = (a >> shift) | (b & ~mask) — shift right and insert
            try emitVecBinop(b, vecs, op, v4f32_ty, LLVMBuildOr);
        },
        .vsli => {
            // SLI = (a << shift) | (b & mask_not)
            try emitVecBinop(b, vecs, op, v4f32_ty, LLVMBuildOr);
        },
        .vfcmp => {
            try emitVecMinMax(b, vecs, op, v4f32_ty, LLVMRealOGT); // placeholder: uses FCmp+Select
        },
        .load_v64 => {
            // 64-bit vector load — load as 64-bit scalar, insert into v4f32 slot 0
            const base = regs[@as(usize, @intCast(op.src0))] orelse return;
            const vdst = vecs[@as(usize, @intCast(op.dest - 31))] orelse return;
            const base_val = LLVMBuildLoad2(b, i64_ty, base, "base");
            const off = LLVMConstInt(i64_ty, @as(u64, @bitCast(@as(i64, @intCast(@as(i32, @bitCast(op.imm)))))), 0);
            const addr = LLVMBuildAdd(b, base_val, off, "addr");
            const i64ptr = LLVMBuildIntToPtr(b, addr, LLVMPointerType(i64_ty, 0), "i64ptr");
            const loaded_i64 = LLVMBuildLoad2(b, i64_ty, i64ptr, "ld64");
            const as_f64 = LLVMBuildBitCast(b, loaded_i64, LLVMDoubleType(), "as_f64");
            const zero_idx = [_]ValueRef{LLVMConstInt(i32_ty, 0, 0)};
            const zero_vec = LLVMConstNull(v4f32_ty);
            const inserted = LLVMBuildInsertElement(b, zero_vec, as_f64, zero_idx[0], "vins");
            _ = LLVMBuildStore(b, inserted, vdst);
        },
        .store_v64 => {
            const base = regs[@as(usize, @intCast(op.src0))] orelse return;
            const vsrc = vecs[@as(usize, @intCast(op.src1 - 31))] orelse return;
            const base_val = LLVMBuildLoad2(b, i64_ty, base, "base");
            const src_val = LLVMBuildLoad2(b, v4f32_ty, vsrc, "vsrc");
            const off = LLVMConstInt(i64_ty, @as(u64, @bitCast(@as(i64, @intCast(@as(i32, @bitCast(op.imm)))))), 0);
            const addr = LLVMBuildAdd(b, base_val, off, "addr");
            // Extract element 0 and store as i64
            const zero_idx = [_]ValueRef{LLVMConstInt(i32_ty, 0, 0)};
            const elem = LLVMBuildExtractElement(b, src_val, zero_idx[0], "vext64");
            const as_i64 = LLVMBuildBitCast(b, elem, i64_ty, "as_i64");
            const i64ptr = LLVMBuildIntToPtr(b, addr, LLVMPointerType(i64_ty, 0), "i64ptr");
            _ = LLVMBuildStore(b, as_i64, i64ptr);
        },

        // ── vfma/vfms: (src0 * src1) ± dest ────────────────────────
        .vfma => {
            const vs0 = vecs[@as(usize, @intCast(op.src0 - 31))] orelse return;
            const vs1 = vecs[@as(usize, @intCast(op.src1 - 31))] orelse return;
            const vdst = vecs[@as(usize, @intCast(op.dest - 31))] orelse return;
            const acc = LLVMBuildLoad2(b, v4f32_ty, vdst, "vfma_acc");
            const a = LLVMBuildLoad2(b, v4f32_ty, vs0, "vfma_a");
            const b_ = LLVMBuildLoad2(b, v4f32_ty, vs1, "vfma_b");
            const prod = LLVMBuildFMul(b, a, b_, "vfma_m");
            const result = LLVMBuildFAdd(b, acc, prod, "vfma");
            _ = LLVMBuildStore(b, result, vdst);
        },
        .vfms => {
            const vs0 = vecs[@as(usize, @intCast(op.src0 - 31))] orelse return;
            const vs1 = vecs[@as(usize, @intCast(op.src1 - 31))] orelse return;
            const vdst = vecs[@as(usize, @intCast(op.dest - 31))] orelse return;
            const acc = LLVMBuildLoad2(b, v4f32_ty, vdst, "vfms_acc");
            const a = LLVMBuildLoad2(b, v4f32_ty, vs0, "vfms_a");
            const b_ = LLVMBuildLoad2(b, v4f32_ty, vs1, "vfms_b");
            const prod = LLVMBuildFMul(b, a, b_, "vfms_m");
            const result = LLVMBuildFSub(b, acc, prod, "vfms");
            _ = LLVMBuildStore(b, result, vdst);
        },

        // ── vrshr: rounding shift right = (val + (1<<(n-1))) >> n ──
        .vrshr => {
            const vs0 = vecs[@as(usize, @intCast(op.src0 - 31))] orelse return;
            const vdst = vecs[@as(usize, @intCast(op.dest - 31))] orelse return;
            const val = LLVMBuildLoad2(b, v4f32_ty, vs0, "vrshr_v");
            _ = LLVMBuildStore(b, val, vdst); // pass through — proper rounding needs element type
        },

        // ── vdup: broadcast scalar to all lanes ────────────────────
        .vdup => {
            const vs0 = vecs[@as(usize, @intCast(op.src0 - 31))] orelse return;
            const vdst = vecs[@as(usize, @intCast(op.dest - 31))] orelse return;
            // Load element 0 from source, then insert across all lanes
            const zero_idx = [_]ValueRef{LLVMConstInt(i32_ty, 0, 0)};
            const src_val = LLVMBuildLoad2(b, v4f32_ty, vs0, "vdup_s");
            const elem = LLVMBuildExtractElement(b, src_val, zero_idx[0], "vdup_e");
            // Build a vector by inserting the same element 4 times
            const zero_vec = LLVMConstNull(v4f32_ty);
            const v1 = LLVMBuildInsertElement(b, zero_vec, elem, zero_idx[0], "vdup_0");
            const one_idx = [_]ValueRef{LLVMConstInt(i32_ty, 1, 0)};
            const v2 = LLVMBuildInsertElement(b, v1, elem, one_idx[0], "vdup_1");
            const two_idx = [_]ValueRef{LLVMConstInt(i32_ty, 2, 0)};
            const v3 = LLVMBuildInsertElement(b, v2, elem, two_idx[0], "vdup_2");
            const three_idx = [_]ValueRef{LLVMConstInt(i32_ty, 3, 0)};
            const v4 = LLVMBuildInsertElement(b, v3, elem, three_idx[0], "vdup_3");
            _ = LLVMBuildStore(b, v4, vdst);
        },

        // ── vrev64: reverse elements within 64-bit halves ──────────
        .vrev64 => {
            const vs0 = vecs[@as(usize, @intCast(op.src0 - 31))] orelse return;
            const vdst = vecs[@as(usize, @intCast(op.dest - 31))] orelse return;
            const val = LLVMBuildLoad2(b, v4f32_ty, vs0, "vrev_v");
            // vrev64 for v4f32: [1, 0, 3, 2] — reverse pairs within 64-bit halves
            const z0 = LLVMConstInt(i32_ty, 0, 0);
            const z1 = LLVMConstInt(i32_ty, 1, 0);
            const z2 = LLVMConstInt(i32_ty, 2, 0);
            const z3 = LLVMConstInt(i32_ty, 3, 0);
            const e0 = LLVMBuildExtractElement(b, val, z1, "vre64_0"); // index 1
            const e1 = LLVMBuildExtractElement(b, val, z0, "vre64_1"); // index 0
            const e2 = LLVMBuildExtractElement(b, val, z3, "vre64_2"); // index 3
            const e3 = LLVMBuildExtractElement(b, val, z2, "vre64_3"); // index 2
            const zvec = LLVMConstNull(v4f32_ty);
            const r0 = LLVMBuildInsertElement(b, zvec, e0, z0, "vre64_r0");
            const r1 = LLVMBuildInsertElement(b, r0, e1, z1, "vre64_r1");
            const r2 = LLVMBuildInsertElement(b, r1, e2, z2, "vre64_r2");
            const r3 = LLVMBuildInsertElement(b, r2, e3, z3, "vre64_r3");
            _ = LLVMBuildStore(b, r3, vdst);
        },

        // ── vrev32: reverse 32-bit elements → shuffle [1,0,3,2] ──
        .vrev32 => {
            const vs0 = vecs[@as(usize, @intCast(op.src0 - 31))] orelse return;
            const vdst = vecs[@as(usize, @intCast(op.dest - 31))] orelse return;
            const val = LLVMBuildLoad2(b, v4f32_ty, vs0, "vrev32_v");
            _ = LLVMBuildStore(b, val, vdst); // same as vrev64 for f32 element type
        },
        // ── vrev16: pass-through (no sub-16-bit elements in v4f32) ──
        .vrev16 => {
            const vs0 = vecs[@as(usize, @intCast(op.src0 - 31))] orelse return;
            const vdst = vecs[@as(usize, @intCast(op.dest - 31))] orelse return;
            const val = LLVMBuildLoad2(b, v4f32_ty, vs0, "vrev16_v");
            _ = LLVMBuildStore(b, val, vdst);
        },
        // ── vqadd/vqsub: saturating add/sub via LLVM intrinsics ────────
        .vqadd => {
            const vs0 = vecs[@as(usize, @intCast(op.src0 - 31))] orelse return;
            const vs1 = vecs[@as(usize, @intCast(op.src1 - 31))] orelse return;
            const vdst = vecs[@as(usize, @intCast(op.dest - 31))] orelse return;
            const a = LLVMBuildLoad2(b, v4f32_ty, vs0, "vqa");
            const b_ = LLVMBuildLoad2(b, v4f32_ty, vs1, "vqb");
            // Bitcast to v4i32, use sadd_sat, bitcast back
            const ai = LLVMBuildBitCast(b, a, v4i32_ty, "vqai");
            const bi = LLVMBuildBitCast(b, b_, v4i32_ty, "vqbi");
            const intr = getIntrinsic(mod, "llvm.sadd_sat.v4i32", v4i32_ty, &.{ v4i32_ty, v4i32_ty });
            var args = [_]ValueRef{ ai, bi };
            const res = LLVMBuildCall2(b, v4i32_ty, intr, &args, 2, "vqadd");
            const res_f = LLVMBuildBitCast(b, res, v4f32_ty, "vqaddf");
            _ = LLVMBuildStore(b, res_f, vdst);
        },
        .vqsub => {
            const vs0 = vecs[@as(usize, @intCast(op.src0 - 31))] orelse return;
            const vs1 = vecs[@as(usize, @intCast(op.src1 - 31))] orelse return;
            const vdst = vecs[@as(usize, @intCast(op.dest - 31))] orelse return;
            const a = LLVMBuildLoad2(b, v4f32_ty, vs0, "vqs");
            const b_ = LLVMBuildLoad2(b, v4f32_ty, vs1, "vqsb");
            const ai = LLVMBuildBitCast(b, a, v4i32_ty, "vqsai");
            const bi = LLVMBuildBitCast(b, b_, v4i32_ty, "vqsbi");
            const intr = getIntrinsic(mod, "llvm.ssub_sat.v4i32", v4i32_ty, &.{ v4i32_ty, v4i32_ty });
            var args = [_]ValueRef{ ai, bi };
            const res = LLVMBuildCall2(b, v4i32_ty, intr, &args, 2, "vqsub");
            const res_f = LLVMBuildBitCast(b, res, v4f32_ty, "vqsubf");
            _ = LLVMBuildStore(b, res_f, vdst);
        },

        // ── vqshl/vqshlu: saturating shift left (approximate: no clamping) ──
        .vqshl => {
            const vs0 = vecs[@as(usize, @intCast(op.src0 - 31))] orelse return;
            const vs1 = vecs[@as(usize, @intCast(op.src1 - 31))] orelse return;
            const vdst = vecs[@as(usize, @intCast(op.dest - 31))] orelse return;
            const lhs = LLVMBuildLoad2(b, v4f32_ty, vs0, "vqsl");
            const rhs = LLVMBuildLoad2(b, v4f32_ty, vs1, "vqsr");
            const result = LLVMBuildShl(b, lhs, rhs, "vqshl");
            _ = LLVMBuildStore(b, result, vdst);
        },
        .vqshlu => {
            const vs0 = vecs[@as(usize, @intCast(op.src0 - 31))] orelse return;
            const vdst = vecs[@as(usize, @intCast(op.dest - 31))] orelse return;
            const val = LLVMBuildLoad2(b, v4f32_ty, vs0, "vqshlu");
            _ = LLVMBuildStore(b, val, vdst);
        },

        // ── vfsqrt: @llvm.sqrt.v4f32 ───────────────────────────────
        .vfsqrt => {
            const vs0 = vecs[@as(usize, @intCast(op.src0 - 31))] orelse return;
            const vdst = vecs[@as(usize, @intCast(op.dest - 31))] orelse return;
            const val = LLVMBuildLoad2(b, v4f32_ty, vs0, "vsqrt_v");
            const fn_ = getIntrinsic(mod, "llvm.sqrt.v4f32", v4f32_ty, &.{v4f32_ty});
            var args = [_]ValueRef{val};
            const result = LLVMBuildCall2(b, v4f32_ty, fn_, &args, 1, "vsqrt");
            _ = LLVMBuildStore(b, result, vdst);
        },

        // ── vpadd: pairwise add [a0+a1, a2+a3, b0+b1, b2+b3] ────
        .vpadd => {
            const vs0 = vecs[@as(usize, @intCast(op.src0 - 31))] orelse return;
            const vs1 = vecs[@as(usize, @intCast(op.src1 - 31))] orelse return;
            const vdst = vecs[@as(usize, @intCast(op.dest - 31))] orelse return;
            const v0 = LLVMBuildLoad2(b, v4f32_ty, vs0, "vp0");
            const v1 = LLVMBuildLoad2(b, v4f32_ty, vs1, "vp1");
            const zero_i = LLVMConstInt(i32_ty, 0, 0);
            const one_i = LLVMConstInt(i32_ty, 1, 0);
            const two_i = LLVMConstInt(i32_ty, 2, 0);
            const three_i = LLVMConstInt(i32_ty, 3, 0);
            const z0 = LLVMBuildExtractElement(b, v0, zero_i, "vp_e0");
            const z1 = LLVMBuildExtractElement(b, v0, one_i, "vp_e1");
            const z2 = LLVMBuildExtractElement(b, v0, two_i, "vp_e2");
            const z3 = LLVMBuildExtractElement(b, v0, three_i, "vp_e3");
            const z4 = LLVMBuildExtractElement(b, v1, zero_i, "vp_e4");
            const z5 = LLVMBuildExtractElement(b, v1, one_i, "vp_e5");
            const z6 = LLVMBuildExtractElement(b, v1, two_i, "vp_e6");
            const z7 = LLVMBuildExtractElement(b, v1, three_i, "vp_e7");
            const s0 = LLVMBuildAdd(b, z0, z1, "vp_s0");
            const s1 = LLVMBuildAdd(b, z2, z3, "vp_s1");
            const s2 = LLVMBuildAdd(b, z4, z5, "vp_s2");
            const s3 = LLVMBuildAdd(b, z6, z7, "vp_s3");
            const zv = LLVMConstNull(v4f32_ty);
            const r0 = LLVMBuildInsertElement(b, zv, s0, zero_i, "vp_r0");
            const r1 = LLVMBuildInsertElement(b, r0, s1, one_i, "vp_r1");
            const r2 = LLVMBuildInsertElement(b, r1, s2, two_i, "vp_r2");
            const r3 = LLVMBuildInsertElement(b, r2, s3, three_i, "vp_r3");
            _ = LLVMBuildStore(b, r3, vdst);
        },

        // ── vext: extract bytes from concatenated vn:vm ────────────
        .vext => {
            const vs0 = vecs[@as(usize, @intCast(op.src0 - 31))] orelse return;
            const vs1 = vecs[@as(usize, @intCast(op.src1 - 31))] orelse return;
            const vdst = vecs[@as(usize, @intCast(op.dest - 31))] orelse return;
            _ = vs1;
            // Pass-through: forward src0
            const val = LLVMBuildLoad2(b, v4f32_ty, vs0, "vext");
            _ = LLVMBuildStore(b, val, vdst);
        },

        // ── vfmulx: multiply extended ≈ FMUL (same for normal values) ──
        .vfmulx => {
            try emitVecBinop(b, vecs, op, v4f32_ty, LLVMBuildFMul);
        },
        // ── vfrecpe: reciprocal estimate ≈ 1.0 / x ────────────────────
        .vfrecpe => {
            const vs0 = vecs[@as(usize, @intCast(op.src0 - 31))] orelse return;
            const vdst = vecs[@as(usize, @intCast(op.dest - 31))] orelse return;
            const val = LLVMBuildLoad2(b, v4f32_ty, vs0, "vrec_v");
            const one = v4f32Const(b, f32_ty, v4f32_ty, 1.0);
            const result = LLVMBuildFDiv(b, one, val, "vrecpe");
            _ = LLVMBuildStore(b, result, vdst);
        },
        // ── vfrecps: Newton-Raphson step = 2.0 - a * b ──────────────
        .vfrecps => {
            const vs0 = vecs[@as(usize, @intCast(op.src0 - 31))] orelse return;
            const vs1 = vecs[@as(usize, @intCast(op.src1 - 31))] orelse return;
            const vdst = vecs[@as(usize, @intCast(op.dest - 31))] orelse return;
            const a = LLVMBuildLoad2(b, v4f32_ty, vs0, "vrec_a");
            const b_ = LLVMBuildLoad2(b, v4f32_ty, vs1, "vrec_b");
            const prod = LLVMBuildFMul(b, a, b_, "vrec_p");
            const two = v4f32Const(b, f32_ty, v4f32_ty, 2.0);
            const result = LLVMBuildFSub(b, two, prod, "vrecps");
            _ = LLVMBuildStore(b, result, vdst);
        },
        // ── vabdl: absolute difference long ≈ vabd (same-size, no widening) ──
        .vabdl => {
            const vs0 = vecs[@as(usize, @intCast(op.src0 - 31))] orelse return;
            const vs1 = vecs[@as(usize, @intCast(op.src1 - 31))] orelse return;
            const vdst = vecs[@as(usize, @intCast(op.dest - 31))] orelse return;
            const lhs = LLVMBuildLoad2(b, v4f32_ty, vs0, "vab_l");
            const rhs = LLVMBuildLoad2(b, v4f32_ty, vs1, "vab_r");
            const sub1 = LLVMBuildSub(b, lhs, rhs, "vab_s1");
            const sub2 = LLVMBuildSub(b, rhs, lhs, "vab_s2");
            const cmp = LLVMBuildFCmp(b, LLVMRealOGT, lhs, rhs, "vab_c");
            const result = LLVMBuildSelect(b, cmp, sub1, sub2, "vabdl");
            _ = LLVMBuildStore(b, result, vdst);
        },

        // ── vpmin/vpmax: pairwise min/max (same pattern as vpadd) ──
        .vpmin => {
            try emitVecPairwise(b, vecs, op, v4f32_ty, i32_ty, emitPairMin);
        },
        .vpmax => {
            try emitVecPairwise(b, vecs, op, v4f32_ty, i32_ty, emitPairMax);
        },
        // ── vaddlp: pairwise add long (same as vpadd for same-size storage) ──
        .vaddlp => {
            try emitVecPairwise(b, vecs, op, v4f32_ty, i32_ty, emitPairAdd);
        },

        // ── vaddlv: pairwise add across ────────────────────────────
        .vaddlv => {
            const vs0 = vecs[@as(usize, @intCast(op.src0 - 31))] orelse return;
            const vdst = vecs[@as(usize, @intCast(op.dest - 31))] orelse return;
            const v = LLVMBuildLoad2(b, v4f32_ty, vs0, "valv_v");
            const z0 = LLVMConstInt(i32_ty, 0, 0);
            const z1 = LLVMConstInt(i32_ty, 1, 0);
            const z2 = LLVMConstInt(i32_ty, 2, 0);
            const z3 = LLVMConstInt(i32_ty, 3, 0);
            const e0 = LLVMBuildExtractElement(b, v, z0, "valv_0");
            const e1 = LLVMBuildExtractElement(b, v, z1, "valv_1");
            const e2 = LLVMBuildExtractElement(b, v, z2, "valv_2");
            const e3 = LLVMBuildExtractElement(b, v, z3, "valv_3");
            const s01 = LLVMBuildAdd(b, e0, e1, "valv_01");
            const s23 = LLVMBuildAdd(b, e2, e3, "valv_23");
            const total = LLVMBuildAdd(b, s01, s23, "valv_t");
            const zvec = LLVMConstNull(v4f32_ty);
            const r0 = LLVMBuildInsertElement(b, zvec, total, z0, "valv_r");
            _ = LLVMBuildStore(b, r0, vdst);
        },

        // ── vtrn/vuzp/vzip: permutation via shuffle ──────────────────
        .vtrn => {
            // Transpose: {a0,a1,a2,a3},{b0,b1,b2,b3} → {a0,b0,a2,b2},{a1,b1,a3,b3}
            const vs0 = vecs[@as(usize, @intCast(op.src0 - 31))] orelse return;
            const vs1 = vecs[@as(usize, @intCast(op.src1 - 31))] orelse return;
            const vdst = vecs[@as(usize, @intCast(op.dest - 31))] orelse return;
            const va = LLVMBuildLoad2(b, v4f32_ty, vs0, "vtr_a");
            const vb = LLVMBuildLoad2(b, v4f32_ty, vs1, "vtr_b");
            // Even: [0,2,4,6] → but we only have 4+4 elements → interleave
            // For v4f32: interleave elements: a0,b0,a1,b1 (stored as result)
            const z0 = LLVMConstInt(i32_ty, 0, 0);
            const z1 = LLVMConstInt(i32_ty, 1, 0);
            const z2 = LLVMConstInt(i32_ty, 2, 0);
            const z3 = LLVMConstInt(i32_ty, 3, 0);
            const e0 = LLVMBuildExtractElement(b, va, z0, "vtr_0");
            const e1 = LLVMBuildExtractElement(b, vb, z0, "vtr_1");
            const e2 = LLVMBuildExtractElement(b, va, z2, "vtr_2");
            const e3 = LLVMBuildExtractElement(b, vb, z2, "vtr_3");
            const zv = LLVMConstNull(v4f32_ty);
            const r0 = LLVMBuildInsertElement(b, zv, e0, z0, "vtr_r0");
            const r1 = LLVMBuildInsertElement(b, r0, e1, z1, "vtr_r1");
            const r2 = LLVMBuildInsertElement(b, r1, e2, z2, "vtr_r2");
            const r3 = LLVMBuildInsertElement(b, r2, e3, z3, "vtr_r3");
            _ = LLVMBuildStore(b, r3, vdst);
        },
        .vuzp => {
            // De-interleave: {a0,a1,a2,a3},{b0,b1,b2,b3} → {a0,a2,b0,b2},{a1,a3,b1,b3}
            const vs0 = vecs[@as(usize, @intCast(op.src0 - 31))] orelse return;
            const vs1 = vecs[@as(usize, @intCast(op.src1 - 31))] orelse return;
            const vdst = vecs[@as(usize, @intCast(op.dest - 31))] orelse return;
            const va = LLVMBuildLoad2(b, v4f32_ty, vs0, "vuz_a");
            const vb = LLVMBuildLoad2(b, v4f32_ty, vs1, "vuz_b");
            const z0 = LLVMConstInt(i32_ty, 0, 0);
            const z1 = LLVMConstInt(i32_ty, 1, 0);
            const z2 = LLVMConstInt(i32_ty, 2, 0);
            const z3 = LLVMConstInt(i32_ty, 3, 0);
            const e0 = LLVMBuildExtractElement(b, va, z0, "vuz_0");
            const e1 = LLVMBuildExtractElement(b, va, z2, "vuz_1");
            const e2 = LLVMBuildExtractElement(b, vb, z0, "vuz_2");
            const e3 = LLVMBuildExtractElement(b, vb, z2, "vuz_3");
            const zv = LLVMConstNull(v4f32_ty);
            const r0 = LLVMBuildInsertElement(b, zv, e0, z0, "vuz_r0");
            const r1 = LLVMBuildInsertElement(b, r0, e1, z1, "vuz_r1");
            const r2 = LLVMBuildInsertElement(b, r1, e2, z2, "vuz_r2");
            const r3 = LLVMBuildInsertElement(b, r2, e3, z3, "vuz_r3");
            _ = LLVMBuildStore(b, r3, vdst);
        },
        .vzip => {
            // Interleave: {a0,a1,a2,a3},{b0,b1,b2,b3} → {a0,b0,a1,b1},{a2,b2,a3,b3}
            const vs0 = vecs[@as(usize, @intCast(op.src0 - 31))] orelse return;
            const vs1 = vecs[@as(usize, @intCast(op.src1 - 31))] orelse return;
            const vsrcd = vecs[@as(usize, @intCast(op.dest - 31))] orelse return;
            const va = LLVMBuildLoad2(b, v4f32_ty, vs0, "vzi_a");
            const vb = LLVMBuildLoad2(b, v4f32_ty, vs1, "vzi_b");
            const z0 = LLVMConstInt(i32_ty, 0, 0);
            const z1 = LLVMConstInt(i32_ty, 1, 0);
            const z2 = LLVMConstInt(i32_ty, 2, 0);
            const z3 = LLVMConstInt(i32_ty, 3, 0);
            const e0 = LLVMBuildExtractElement(b, va, z0, "vzi_0");
            const e1 = LLVMBuildExtractElement(b, vb, z0, "vzi_1");
            const e2 = LLVMBuildExtractElement(b, va, z1, "vzi_2");
            const e3 = LLVMBuildExtractElement(b, vb, z1, "vzi_3");
            const zv = LLVMConstNull(v4f32_ty);
            const r0 = LLVMBuildInsertElement(b, zv, e0, z0, "vzi_r0");
            const r1 = LLVMBuildInsertElement(b, r0, e1, z1, "vzi_r1");
            const r2 = LLVMBuildInsertElement(b, r1, e2, z2, "vzi_r2");
            const r3 = LLVMBuildInsertElement(b, r2, e3, z3, "vzi_r3");
            _ = LLVMBuildStore(b, r3, vsrcd);
        },

        // ── vtbl/vtbx: table lookup via memory (approximate) ────────
        .vtbl => {
            const vs0 = vecs[@as(usize, @intCast(op.src0 - 31))] orelse return;
            const vs1 = vecs[@as(usize, @intCast(op.src1 - 31))] orelse return;
            const vdst = vecs[@as(usize, @intCast(op.dest - 31))] orelse return;
            // Approximate: allocate stack memory, store table, index, load
            // For now, pass src0 through (simplified)
            const val = LLVMBuildLoad2(b, v4f32_ty, vs0, "vtbl");
            _ = vs1;
            _ = LLVMBuildStore(b, val, vdst);
        },
        .vtbx => {
            const vs0 = vecs[@as(usize, @intCast(op.src0 - 31))] orelse return;
            const vdst = vecs[@as(usize, @intCast(op.dest - 31))] orelse return;
            const val = LLVMBuildLoad2(b, v4f32_ty, vs0, "vtbx");
            _ = LLVMBuildStore(b, val, vdst);
        },

        // ── vrshrn: rounding shift right narrow (pass-through) ──────
        .vrshrn => {
            const vs0 = vecs[@as(usize, @intCast(op.src0 - 31))] orelse return;
            const vdst = vecs[@as(usize, @intCast(op.dest - 31))] orelse return;
            const val = LLVMBuildLoad2(b, v4f32_ty, vs0, "vrshrn");
            _ = LLVMBuildStore(b, val, vdst);
        },

        // ── fcvtl/fcvtn: float long/narrow (pass-through) ──────────
        .fcvtl => {
            const vs0 = vecs[@as(usize, @intCast(op.src0 - 31))] orelse return;
            const vdst = vecs[@as(usize, @intCast(op.dest - 31))] orelse return;
            const val = LLVMBuildLoad2(b, v4f32_ty, vs0, "fcvtl");
            _ = LLVMBuildStore(b, val, vdst);
        },
        .fcvtn => {
            const vs0 = vecs[@as(usize, @intCast(op.src0 - 31))] orelse return;
            const vdst = vecs[@as(usize, @intCast(op.dest - 31))] orelse return;
            const val = LLVMBuildLoad2(b, v4f32_ty, vs0, "fcvtn");
            _ = LLVMBuildStore(b, val, vdst);
        },

        else => {},
    }
}

fn emitBinop(
    b: BuilderRef,
    regs: *[31]?ValueRef,
    op: IROp,
    ty: TypeRef,
    comptime fn_binop: fn (BuilderRef, ValueRef, ValueRef, [*:0]const u8) callconv(.c) ValueRef,
    flags: *FlagState,
) void {
    const zero_val = LLVMConstInt(ty, 0, 0);
    const src0 = if (op.src0 >= 31) null else regs[@as(usize, @intCast(op.src0))];
    const src0val = if (src0) |s| LLVMBuildLoad2(b, ty, s, "lhs") else zero_val;
    const rhs_val: ValueRef = if (op.imm != 0) blk: {
        // Sign-extend i32 imm to i64 for negative offsets (LDP/STP etc).
        break :blk LLVMConstInt(ty, @as(u64, @bitCast(@as(i64, @intCast(@as(i32, @bitCast(op.imm)))))), 0);
    } else if (op.src1 < 31) blk: {
        const src1 = regs[@as(usize, @intCast(op.src1))] orelse return;
        break :blk LLVMBuildLoad2(b, ty, src1, "rhs");
    } else zero_val;
    const result = fn_binop(b, src0val, rhs_val, "res");
    // Record ALU state for nzcv_update (CMP/CMN with dest=31 still sets flags).
    flags.* = .{
        .lhs = src0val,
        .rhs = rhs_val,
        .result = result,
        .kind = switch (fn_binop) {
            LLVMBuildSub => .sub,
            LLVMBuildAdd => .add,
            else => .logical,
        },
    };
    // XZR dest (31) = CMP/CMN: discard result, only set flags
    if (op.dest >= 31) return;
    const dst = regs[@as(usize, @intCast(op.dest))] orelse return;
    _ = LLVMBuildStore(b, result, dst);
}

fn emitVecBinop(
    b: BuilderRef,
    vecs: *[32]?ValueRef,
    op: IROp,
    vty: TypeRef,
    comptime fn_binop: fn (BuilderRef, ValueRef, ValueRef, [*:0]const u8) callconv(.c) ValueRef,
) !void {
    const vs0 = vecs[@as(usize, @intCast(op.src0 - 31))] orelse return;
    const vdst = vecs[@as(usize, @intCast(op.dest - 31))] orelse return;
    if (op.src1 >= 31 and op.src1 < 128) {
        const vs1 = vecs[@as(usize, @intCast(op.src1 - 31))] orelse return;
        const lhs = LLVMBuildLoad2(b, vty, vs0, "vlhs");
        const rhs = LLVMBuildLoad2(b, vty, vs1, "vrhs");
        const result = fn_binop(b, lhs, rhs, "vres");
        _ = LLVMBuildStore(b, result, vdst);
    } else {
        const val = LLVMBuildLoad2(b, vty, vs0, "val");
        _ = LLVMBuildStore(b, val, vdst);
    }
}

fn emitVecCmp(
    b: BuilderRef,
    vecs: *[32]?ValueRef,
    op: IROp,
    vty: TypeRef,
    pred: c_uint,
) !void {
    const vs0 = vecs[@as(usize, @intCast(op.src0 - 31))] orelse return;
    const vs1 = vecs[@as(usize, @intCast(op.src1 - 31))] orelse return;
    const vdst = vecs[@as(usize, @intCast(op.dest - 31))] orelse return;
    const lhs = LLVMBuildLoad2(b, vty, vs0, "vc_lhs");
    const rhs = LLVMBuildLoad2(b, vty, vs1, "vc_rhs");
    // ICmp on vector types returns <N x i1> mask; select expands to all-0/all-1
    const cmp = LLVMBuildICmp(b, pred, lhs, rhs, "vcmp");
    const zero = LLVMConstNull(vty);
    const ones = LLVMBuildNot(b, zero, "vones");
    const result = LLVMBuildSelect(b, cmp, ones, zero, "vcres");
    _ = LLVMBuildStore(b, result, vdst);
}

fn emitVecMinMax(
    b: BuilderRef,
    vecs: *[32]?ValueRef,
    op: IROp,
    vty: TypeRef,
    pred: c_uint,
) !void {
    const vs0 = vecs[@as(usize, @intCast(op.src0 - 31))] orelse return;
    const vs1 = vecs[@as(usize, @intCast(op.src1 - 31))] orelse return;
    const vdst = vecs[@as(usize, @intCast(op.dest - 31))] orelse return;
    const lhs = LLVMBuildLoad2(b, vty, vs0, "vm_lhs");
    const rhs = LLVMBuildLoad2(b, vty, vs1, "vm_rhs");
    // Check element type from op.flags bits 0-2
    const elem_tag = Ir.VecFlags.elemType(op.flags);
    const is_float = switch (elem_tag) {
        .f32, .f64 => true,
        else => false,
    };
    if (is_float) {
        const cmp = LLVMBuildFCmp(b, pred, lhs, rhs, "vm_cmp");
        const result = LLVMBuildSelect(b, cmp, lhs, rhs, "vm_res");
        _ = LLVMBuildStore(b, result, vdst);
    } else {
        // Map FCmp predicate to ICmp signed predicate
        const icmp_pred: c_uint = switch (pred) {
            LLVMRealOLT => LLVMIntSLT,
            LLVMRealOGT => LLVMIntSGT,
            LLVMRealOLE => LLVMIntSLE,
            LLVMRealOGE => LLVMIntSGE,
            LLVMRealOEQ => LLVMIntEQ,
            LLVMRealONE => LLVMIntNE,
            else => pred,
        };
        const cmp = LLVMBuildICmp(b, icmp_pred, lhs, rhs, "vm_cmp");
        const result = LLVMBuildSelect(b, cmp, lhs, rhs, "vm_res");
        _ = LLVMBuildStore(b, result, vdst);
    }
}

fn emitVecConvert(
    b: BuilderRef,
    vecs: *[32]?ValueRef,
    op: IROp,
    vty: TypeRef,
    comptime fn_cvt: fn (BuilderRef, ValueRef, TypeRef, [*:0]const u8) callconv(.c) ValueRef,
    scalar_ty: TypeRef,
    f32_ty: TypeRef,
    i32_ty: TypeRef,
) !void {
    const vs0 = vecs[@as(usize, @intCast(op.src0 - 31))] orelse return;
    const vdst = vecs[@as(usize, @intCast(op.dest - 31))] orelse return;
    const val = LLVMBuildLoad2(b, vty, vs0, "vcvt_in");
    // Element-wise conversion: extract each 4-wide element, apply fn_cvt, insert back.
    const z0 = LLVMConstInt(i32_ty, 0, 0);
    const z1 = LLVMConstInt(i32_ty, 1, 0);
    const z2 = LLVMConstInt(i32_ty, 2, 0);
    const z3 = LLVMConstInt(i32_ty, 3, 0);
    const e0 = LLVMBuildExtractElement(b, val, z0, "cvt0");
    const e1 = LLVMBuildExtractElement(b, val, z1, "cvt1");
    const e2 = LLVMBuildExtractElement(b, val, z2, "cvt2");
    const e3 = LLVMBuildExtractElement(b, val, z3, "cvt3");
    // For int→float conversions (scvtf/ucvtf), input is stored as f32 bit pattern,
    // so bitcast each element to i32 before converting.
    const is_int_to_float = comptime (fn_cvt == LLVMBuildSIToFP or fn_cvt == LLVMBuildUIToFP);
    const out_ty: TypeRef = if (is_int_to_float) f32_ty else scalar_ty;
    const bc0 = if (is_int_to_float) LLVMBuildBitCast(b, e0, i32_ty, "bc0") else e0;
    const c0 = fn_cvt(b, bc0, out_ty, "cvt_r0");
    const bc1 = if (is_int_to_float) LLVMBuildBitCast(b, e1, i32_ty, "bc1") else e1;
    const c1 = fn_cvt(b, bc1, out_ty, "cvt_r1");
    const bc2 = if (is_int_to_float) LLVMBuildBitCast(b, e2, i32_ty, "bc2") else e2;
    const c2 = fn_cvt(b, bc2, out_ty, "cvt_r2");
    const bc3 = if (is_int_to_float) LLVMBuildBitCast(b, e3, i32_ty, "bc3") else e3;
    const c3 = fn_cvt(b, bc3, out_ty, "cvt_r3");
    const zv = LLVMConstNull(vty);
    const r0 = LLVMBuildInsertElement(b, zv, c0, z0, "cvt_w0");
    const r1 = LLVMBuildInsertElement(b, r0, c1, z1, "cvt_w1");
    const r2 = LLVMBuildInsertElement(b, r1, c2, z2, "cvt_w2");
    const r3 = LLVMBuildInsertElement(b, r2, c3, z3, "cvt_w3");
    _ = LLVMBuildStore(b, r3, vdst);
}

fn emitVecMla(
    b: BuilderRef,
    vecs: *[32]?ValueRef,
    op: IROp,
    vty: TypeRef,
    subtract: bool,
) !void {
    // MLA = dest + src0 * src1  (dest is the accumulator, src0*src1 is the product)
    // For BSL-like MLA: dest = src0 + src1 * dest → meaning: dest *= src1; dest += src0
    // ARM64 MLA: Vd = Va + Vn * Vm  (Vd is both dest and accumulator)
    const vs0 = vecs[@as(usize, @intCast(op.src0 - 31))] orelse return;
    const vs1 = vecs[@as(usize, @intCast(op.src1 - 31))] orelse return;
    const vdst = vecs[@as(usize, @intCast(op.dest - 31))] orelse return;
    const acc = LLVMBuildLoad2(b, vty, vdst, "v_acc");
    const a = LLVMBuildLoad2(b, vty, vs0, "v_mla_a");
    const b_ = LLVMBuildLoad2(b, vty, vs1, "v_mla_b");
    const prod = LLVMBuildMul(b, a, b_, "v_prod");
    const result = if (subtract)
        LLVMBuildSub(b, acc, prod, "v_mls")
    else
        LLVMBuildAdd(b, acc, prod, "v_mla");
    _ = LLVMBuildStore(b, result, vdst);
}
/// Pairwise operation helper: extract 4 elements from each vec, apply op, insert results
fn emitVecPairwise(
    b: BuilderRef,
    vecs: *[32]?ValueRef,
    op: IROp,
    vty: TypeRef,
    i32_ty: TypeRef,
    comptime pair_op: fn (BuilderRef, ValueRef, ValueRef, [*:0]const u8) callconv(.c) ValueRef,
) !void {
    const vs0 = vecs[@as(usize, @intCast(op.src0 - 31))] orelse return;
    const vs1 = vecs[@as(usize, @intCast(op.src1 - 31))] orelse return;
    const vdst = vecs[@as(usize, @intCast(op.dest - 31))] orelse return;
    const v0 = LLVMBuildLoad2(b, vty, vs0, "pp_v0");
    const v1 = LLVMBuildLoad2(b, vty, vs1, "pp_v1");
    const z0 = LLVMConstInt(i32_ty, 0, 0);
    const z1 = LLVMConstInt(i32_ty, 1, 0);
    const z2 = LLVMConstInt(i32_ty, 2, 0);
    const z3 = LLVMConstInt(i32_ty, 3, 0);
    const e0 = LLVMBuildExtractElement(b, v0, z0, "pp_e0");
    const e1 = LLVMBuildExtractElement(b, v0, z1, "pp_e1");
    const e2 = LLVMBuildExtractElement(b, v0, z2, "pp_e2");
    const e3 = LLVMBuildExtractElement(b, v0, z3, "pp_e3");
    const e4 = LLVMBuildExtractElement(b, v1, z0, "pp_e4");
    const e5 = LLVMBuildExtractElement(b, v1, z1, "pp_e5");
    const e6 = LLVMBuildExtractElement(b, v1, z2, "pp_e6");
    const e7 = LLVMBuildExtractElement(b, v1, z3, "pp_e7");
    const p0 = pair_op(b, e0, e1, "pp_p0");
    const p1 = pair_op(b, e2, e3, "pp_p1");
    const p2 = pair_op(b, e4, e5, "pp_p2");
    const p3 = pair_op(b, e6, e7, "pp_p3");
    const zv = LLVMConstNull(vty);
    const r0 = LLVMBuildInsertElement(b, zv, p0, z0, "pp_r0");
    const r1 = LLVMBuildInsertElement(b, r0, p1, z1, "pp_r1");
    const r2 = LLVMBuildInsertElement(b, r1, p2, z2, "pp_r2");
    const r3 = LLVMBuildInsertElement(b, r2, p3, z3, "pp_r3");
    _ = LLVMBuildStore(b, r3, vdst);
}

fn emitPairAdd(b: BuilderRef, a: ValueRef, b_: ValueRef, name: [*:0]const u8) callconv(.c) ValueRef {
    return LLVMBuildAdd(b, a, b_, name);
}
fn emitPairMin(b: BuilderRef, a: ValueRef, b_: ValueRef, name: [*:0]const u8) callconv(.c) ValueRef {
    const cmp = LLVMBuildFCmp(b, LLVMRealOLT, a, b_, "min_c");
    return LLVMBuildSelect(b, cmp, a, b_, name);
}
fn emitPairMax(b: BuilderRef, a: ValueRef, b_: ValueRef, name: [*:0]const u8) callconv(.c) ValueRef {
    const cmp = LLVMBuildFCmp(b, LLVMRealOGT, a, b_, "max_c");
    return LLVMBuildSelect(b, cmp, a, b_, name);
}

fn emitUnop(
    b: BuilderRef,
    regs: *[31]?ValueRef,
    op: IROp,
    ty: TypeRef,
    comptime fn_unop: fn (BuilderRef, ValueRef, [*:0]const u8) callconv(.c) ValueRef,
) void {
    if (op.dest >= 31) return;
    const dst = regs[@as(usize, @intCast(op.dest))] orelse return;
    if (op.src0 >= 31) {
        const zero_val = LLVMConstInt(ty, 0, 0);
        const result = fn_unop(b, zero_val, "res");
        _ = LLVMBuildStore(b, result, dst);
    } else if (regs[@as(usize, @intCast(op.src0))]) |src0| {
        const val = LLVMBuildLoad2(b, ty, src0, "val");
        const result = fn_unop(b, val, "res");
        _ = LLVMBuildStore(b, result, dst);
    }
}

// ── saSLP combined SIMD op emission ──────────────────────────────
// Concatenates two v4f32 operands into v8f32, applies the operation
// on the wider type, then splits and stores results back to v4f32 allocas.
// This gives LLVM the opportunity to generate 256-bit AVX instructions.

/// Emit a combined SIMD op that processes two adjacent vreg pairs as one
/// 256-bit vector operation. Called from compileBlock when tryCombineSimdPair
/// detects a combinable pair.
fn emitCombinedSimdOp(
    b: BuilderRef,
    vecs: *[32]?ValueRef,
    op0: IROp,
    op1: IROp,
    v4f32_ty: TypeRef,
    v8f32_ty: TypeRef,
    concat_mask: ValueRef,
    split_low_mask: ValueRef,
    split_high_mask: ValueRef,
) void {
    const tag = op0.tag;
    const dst_low_idx = @as(usize, @intCast(op0.dest - 31));
    const dst_high_idx = @as(usize, @intCast(op1.dest - 31));
    const src0_low_idx = @as(usize, @intCast(op0.src0 - 31));
    const src0_high_idx = @as(usize, @intCast(op1.src0 - 31));

    const vdst_low = vecs[dst_low_idx] orelse return;
    const vdst_high = vecs[dst_high_idx] orelse return;
    const vs0_low = vecs[src0_low_idx] orelse return;
    const vs0_high = vecs[src0_high_idx] orelse return;

    // Load both v4f32 source halves
    const s0_low_v = LLVMBuildLoad2(b, v4f32_ty, vs0_low, "cs_low");
    const s0_high_v = LLVMBuildLoad2(b, v4f32_ty, vs0_high, "cs_high");

    // Concatenate into v8f32 via shufflevector
    // shufflevector <4 x float> %s0_low, <4 x float> %s0_high, <8 x i32> <0,1,2,3,4,5,6,7>
    // Result: low[0..3] followed by high[0..3] = one contiguous 256-bit vector
    const s0_v8 = LLVMBuildShuffleVector(b, s0_low_v, s0_high_v, concat_mask, "cs_concat");

    const has_src1 = op0.src1 >= 31 and op0.src1 < 128;
    const null_v8 = LLVMConstNull(v8f32_ty);

    if (has_src1) {
        const src1_low_idx = @as(usize, @intCast(op0.src1 - 31));
        const src1_high_idx = @as(usize, @intCast(op1.src1 - 31));
        const vs1_low = vecs[src1_low_idx] orelse return;
        const vs1_high = vecs[src1_high_idx] orelse return;

        const s1_low_v = LLVMBuildLoad2(b, v4f32_ty, vs1_low, "cs_s1l");
        const s1_high_v = LLVMBuildLoad2(b, v4f32_ty, vs1_high, "cs_s1h");
        const s1_v8 = LLVMBuildShuffleVector(b, s1_low_v, s1_high_v, concat_mask, "cs_s1");

        // Dispatch to the right LLVM build function based on op tag.
        // The operation is applied to the full 256-bit v8f32 vector, enabling
        // LLVM to emit a single AVX instruction instead of two SSE instructions.
        const result_v8 = dispatchCombinedOp(b, tag, s0_v8, s1_v8);

        // Extract v4f32 halves from v8f32 result
        const res_low = LLVMBuildShuffleVector(b, result_v8, null_v8, split_low_mask, "cs_rlow");
        const res_high = LLVMBuildShuffleVector(b, result_v8, null_v8, split_high_mask, "cs_rhigh");
        _ = LLVMBuildStore(b, res_low, vdst_low);
        _ = LLVMBuildStore(b, res_high, vdst_high);
    } else {
        // Unary combined op (pass-through: forward the concatenated value)
        const res_low = LLVMBuildShuffleVector(b, s0_v8, null_v8, split_low_mask, "cs_rlow");
        const res_high = LLVMBuildShuffleVector(b, s0_v8, null_v8, split_high_mask, "cs_rhigh");
        _ = LLVMBuildStore(b, res_low, vdst_low);
        _ = LLVMBuildStore(b, res_high, vdst_high);
    }
}

/// Dispatch a combined v8f32 operation to the correct LLVM build function.
/// Follows the same pattern as the per-lane operations in emitOp but operates
/// on the full 256-bit vector.
fn dispatchCombinedOp(
    b: BuilderRef,
    tag: Tag,
    lhs: ValueRef,
    rhs: ValueRef,
) ValueRef {
    return switch (tag) {
        .vfadd => LLVMBuildFAdd(b, lhs, rhs, "cadd"),
        .vfsub => LLVMBuildFSub(b, lhs, rhs, "csub"),
        .vfmul => LLVMBuildFMul(b, lhs, rhs, "cmul"),
        .vfmin => blk: {
            const cmp = LLVMBuildFCmp(b, LLVMRealOLT, lhs, rhs, "cmin_c");
            break :blk LLVMBuildSelect(b, cmp, lhs, rhs, "cmin");
        },
        .vfmax => blk: {
            const cmp = LLVMBuildFCmp(b, LLVMRealOGT, lhs, rhs, "cmax_c");
            break :blk LLVMBuildSelect(b, cmp, lhs, rhs, "cmax");
        },
        .vadd => LLVMBuildAdd(b, lhs, rhs, "cadd"),
        .vsub => LLVMBuildSub(b, lhs, rhs, "csub"),
        .vmul => LLVMBuildMul(b, lhs, rhs, "cmul"),
        .vand => LLVMBuildAnd(b, lhs, rhs, "cand"),
        .vorr => LLVMBuildOr(b, lhs, rhs, "cor"),
        .veor => LLVMBuildXor(b, lhs, rhs, "cxor"),
        else => lhs,
    };
}
