const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const root_module = b.createModule(.{
        .root_source_file = .{ .src_path = .{ .owner = b, .sub_path = "src/main.zig" } },
        .target = target,
        .optimize = optimize,
    });

    // ── Executable ──────────────────────────────────────────────────
    const exe = b.addExecutable(.{
        .name = "a64tox64",
        .root_module = root_module,
    });
    exe.linkSystemLibrary("dl");
    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());

    const run_step = b.step("run", "Run a64tox64 CLI");
    run_step.dependOn(&run_cmd.step);

    // ── Unit tests ──────────────────────────────────────────────────
    const test_module = b.createModule(.{
        .root_source_file = .{ .src_path = .{ .owner = b, .sub_path = "src/main.zig" } },
        .target = target,
        .optimize = optimize,
    });

    const unit_tests = b.addTest(.{
        .root_module = test_module,
    });

    const test_step = b.step("test", "Run all unit tests");
    test_step.dependOn(&unit_tests.step);
}
