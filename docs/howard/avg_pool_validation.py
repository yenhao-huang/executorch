# Rebuild focused tests against the existing CMake runtime cache described in 11325-11326.md.
from pathlib import Path
import shlex, subprocess, os

src = Path("/workspace/executorch-worktrees/avg-pool-kernels")
b = Path("/workspace/executorch-worktrees/batch-2026-10/executorch/cmake-out")
out = Path(os.environ.get("AVG_BUILD", "/tmp/avg-build"))
out.mkdir(exist_ok=True)
inc = out / "source"
inc.mkdir(exist_ok=True)
if not (inc / "executorch").exists():
    (inc / "executorch").symlink_to(src, target_is_directory=True)
s = (b / "kernels/test/CMakeFiles/portable_kernels_test.dir/flags.make").read_text()
flags = []
for line in s.splitlines():
    if line.startswith(("CXX_DEFINES =", "CXX_INCLUDES =", "CXX_FLAGS =")):
        flags += shlex.split(line.split("=", 1)[1])
flags = (
    ([] if os.environ.get("BASELINE") else ["-I/tmp/avg-build/include"])
    + ["-I" + str(inc)]
    + flags
)
if os.environ.get("SANITIZE"):
    flags += ["-fsanitize=address,undefined", "-fno-omit-frame-pointer", "-O1", "-g"]
objs = []
for name in (
    ["test3"]
    if os.environ.get("BASELINE3")
    else ["test1", "test3", "kernel1", "kernel3"]
):
    n = name[-1]
    file = src / (
        f"kernels/test/op_avg_pool{n}d_test.cpp"
        if name.startswith("test")
        else f"kernels/portable/cpu/op_avg_pool{n}d.cpp"
    )
    file = (
        Path(os.environ.get("TEST_SOURCE" + n, str(file)))
        if name.startswith("test")
        else file
    )
    obj = out / (name + ".o")
    subprocess.run(
        ["/usr/bin/c++", *flags, "-c", str(file), "-o", str(obj)], check=True
    )
    objs.append(obj)
libs = [
    b / "lib/libgtest_main.a",
    b / "lib/libgtest.a",
    b / "lib/libgmock.a",
    b / "kernels/portable/libportable_kernels.a",
    b / "kernels/portable/cpu/util/libkernels_util_all_deps.a",
    b / "libexecutorch_core.a",
]
extras = [
    b
    / "kernels/test/CMakeFiles/portable_kernels_test.dir/__/__/runtime/core/exec_aten/testing_util/tensor_util.cpp.o",
    b
    / "kernels/test/CMakeFiles/portable_kernels_test.dir/include/portable/executorch/kernels/test/supported_features.cpp.o",
]
subprocess.run(
    [
        "/usr/bin/c++",
        *map(str, objs + extras),
        "-Wl,--start-group",
        *map(str, libs),
        "-Wl,--end-group",
        *(["-fsanitize=address,undefined"] if os.environ.get("SANITIZE") else []),
        "-lpthread",
        "-o",
        str(out / "test"),
    ],
    check=True,
)
