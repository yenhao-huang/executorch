#!/usr/bin/env python3
"""Recompile the reviewed kernel/tests against an existing portable CMake build."""
import argparse
import shlex
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--runtime-build", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--kernel-source", type=Path)
parser.add_argument("--sanitize", action="store_true")
args = parser.parse_args()
source = Path(__file__).resolve().parents[2]
build = args.runtime_build.resolve()
out = args.out.resolve()
out.mkdir(parents=True, exist_ok=True)
include = out / "include"
include.mkdir(exist_ok=True)
if not (include / "executorch").exists():
    (include / "executorch").symlink_to(source, target_is_directory=True)
flags = ["-I" + str(include)]
for line in (
    (build / "kernels/test/CMakeFiles/portable_kernels_test.dir/flags.make")
    .read_text()
    .splitlines()
):
    if line.startswith(("CXX_DEFINES =", "CXX_INCLUDES =", "CXX_FLAGS =")):
        flags.extend(shlex.split(line.split("=", 1)[1]))
sanitize = ["-fsanitize=address,undefined"] if args.sanitize else []
if args.sanitize:
    flags.extend(sanitize + ["-fno-omit-frame-pointer", "-O1", "-g"])
for name, path in [
    ("test", source / "kernels/test/op_embedding_bag_test.cpp"),
    (
        "kernel",
        args.kernel_source or source / "kernels/portable/cpu/op_embedding_bag.cpp",
    ),
]:
    subprocess.run(
        ["c++", *flags, "-c", str(path), "-o", str(out / (name + ".o"))], check=True
    )
libs = [
    "lib/libgtest_main.a",
    "lib/libgtest.a",
    "lib/libgmock.a",
    "kernels/portable/libportable_kernels.a",
    "kernels/portable/cpu/util/libkernels_util_all_deps.a",
    "libexecutorch_core.a",
]
objects = [
    "kernels/test/CMakeFiles/portable_kernels_test.dir/__/__/runtime/core/exec_aten/testing_util/tensor_util.cpp.o",
    "kernels/test/CMakeFiles/portable_kernels_test.dir/include/portable/executorch/kernels/test/supported_features.cpp.o",
]
subprocess.run(
    [
        "c++",
        str(out / "test.o"),
        str(out / "kernel.o"),
        *[str(build / p) for p in objects],
        "-Wl,--start-group",
        *[str(build / p) for p in libs],
        "-Wl,--end-group",
        *sanitize,
        "-lpthread",
        "-o",
        str(out / "test"),
    ],
    check=True,
)
