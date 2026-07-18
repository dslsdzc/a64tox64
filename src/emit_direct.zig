//! Direct emitter — fast path for trivial 1:1 ARM64→x86-64 translations.
//!
//! Bypasses the IR entirely for simple instructions that map directly
//! to a single x86-64 instruction. The IR-based pipeline handles
//! everything else.

const Emit = @import("emit.zig");
const Decode = @import("decode.zig");
const EmitContext = Emit.EmitContext;
const A64Inst = Decode.A64Inst;
const Opcode = Decode.Opcode;

/// Try to emit x86-64 code directly from a decoded ARM64 instruction.
/// Returns `true` if the instruction was handled, `false` to fall
/// back to the IR path.
pub fn tryEmitDirect(ctx: *EmitContext, inst: A64Inst) bool {
    switch (inst.opcode) {
        .nop => {
            ctx.byte(0x90); // NOP
            return true;
        },
        .ret_ => {
            ctx.byte(0xC3); // RET
            return true;
        },
        else => return false,
    }
}
