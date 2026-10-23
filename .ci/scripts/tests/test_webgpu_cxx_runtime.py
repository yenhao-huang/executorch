# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Exercise the sourced setup and native linker without network/package writes.

A linker-script fixture adds Dawn's missing symbol beside the host libstdc++,
so tests do not require downloading Dawn or upgrading the machine's compiler.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1]
SETUP_SCRIPT = Path(
    os.environ.get("WEBGPU_TEST_SETUP_SCRIPT", SCRIPTS / "setup-webgpu-linux-deps.sh")
)
SYMBOL = (
    "_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE15_M_replace_coldEPcmPKcmm"
)


@unittest.skipUnless(
    sys.platform.startswith("linux") and shutil.which("cmake") and shutil.which("c++"),
    "needs Linux, CMake and C++",
)
class WebGpuRuntimeTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="webgpu-runtime-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.prefix = self.root / "conda with spaces"
        lib = self.prefix / "lib"
        lib.mkdir(parents=True)
        system = subprocess.check_output(
            ["c++", "-print-file-name=libstdc++.so.6"], text=True
        ).strip()
        self.system = str(Path(system).resolve())
        source = self.root / "symbol.cpp"
        source.write_text(
            f'extern "C" void needed() asm("{SYMBOL}");\nvoid needed() {{}}\n'
        )
        subprocess.run(
            [
                "c++",
                "-shared",
                "-fPIC",
                str(source),
                "-Wl,-soname,libdawn_symbol.so",
                "-o",
                str(lib / "libdawn_symbol.so"),
            ],
            check=True,
        )
        (lib / "libstdc++.so").write_text(
            f'GROUP ( "{self.system}" "{lib / "libdawn_symbol.so"}" )\n'
        )
        (lib / "libstdc++.so.6").symlink_to(self.system)
        self.env = os.environ.copy()
        for name in ("CONDA_PREFIX", "LDFLAGS", "LD_LIBRARY_PATH", "CXX", "CXXFLAGS"):
            self.env.pop(name, None)
        self.env.update(
            CONDA_PREFIX=str(self.prefix),
            CXX="c++",
            DAWN_PREBUILT_DIR=str(self.root / "dawn"),
        )

    def run_helper(self, env=None):
        return subprocess.run(
            [
                "bash",
                "-ec",
                'source "$1"; python3 -c \'import json,os; print(json.dumps({k:os.environ.get(k) for k in ["LDFLAGS","LD_LIBRARY_PATH"]}))\'',
                "test",
                str(SCRIPTS / "setup-webgpu-cxx-runtime.sh"),
            ],
            env=env or self.env,
            text=True,
            capture_output=True,
        )

    def test_selected_runtime_links_and_loads_and_preserves_flags(self):
        self.env.update(LDFLAGS="-Wl,-z,relro", LD_LIBRARY_PATH="/existing/runtime")
        result = self.run_helper()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        exported = json.loads(result.stdout.splitlines()[-1])
        self.assertIn("-Wl,-z,relro", exported["LDFLAGS"])
        self.assertEqual(
            exported["LD_LIBRARY_PATH"], f"{self.prefix}/lib:/existing/runtime"
        )

    def test_missing_link_library_fails(self):
        (self.prefix / "lib/libstdc++.so").unlink()
        result = self.run_helper()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("libstdc++.so is missing", result.stderr)

    def test_incompatible_runtime_fails_before_backend_build(self):
        source = self.root / "incompatible.cpp"
        source.write_text("void unused() {}\n")
        # Force a library without the required symbol, even on GCC 13+ hosts.
        bad = self.prefix / "lib/libbadcxx.so"
        subprocess.run(
            ["c++", "-shared", "-fPIC", str(source), "-o", str(bad)], check=True
        )
        (self.prefix / "lib/libstdc++.so").write_text(f'INPUT ( "{bad}" )\n')
        result = self.run_helper()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("_M_replace_cold", result.stderr)

    def test_non_conda_compatible_flags_are_preserved(self):
        self.env.pop("CONDA_PREFIX")
        self.env["LDFLAGS"] = f'-L"{self.prefix}/lib" -Wl,-rpath,"{self.prefix}/lib"'
        self.env["LD_LIBRARY_PATH"] = str(self.prefix / "lib")
        result = self.run_helper()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(
            json.loads(result.stdout.splitlines()[-1])["LDFLAGS"], self.env["LDFLAGS"]
        )

    def test_full_setup_selects_runtime_before_downloading_dawn(self):
        probe = self.root / "consumer"
        probe.mkdir()
        (probe / "main.cpp").write_text(
            f'extern "C" void needed() asm("{SYMBOL}");\n'
            "int main() { void (*volatile symbol)() = &needed; return symbol == nullptr; }\n"
        )
        (probe / "CMakeLists.txt").write_text(
            "cmake_minimum_required(VERSION 3.18)\nproject(consumer LANGUAGES CXX)\n"
            "add_executable(consumer main.cpp)\n"
        )
        self.env.update(
            DAWN_PREBUILT_DIR=str(self.root / "dawn"), TEST_CONSUMER=str(probe)
        )
        result = subprocess.run(
            [
                "bash",
                "-ec",
                """
                sudo() { return 0; }
                apt-get() { return 0; }
                add-apt-repository() { return 0; }
                conda() { return 0; }
                curl() { echo "Unexpected download" >&2; return 99; }
                mkdir() {
                    if [[ "${@: -1}" == "$DAWN_PREBUILT_DIR" ]]; then
                        cmake -S "$TEST_CONSUMER" -B "$TEST_CONSUMER/build"
                        cmake --build "$TEST_CONSUMER/build"
                        "$TEST_CONSUMER/build/consumer"
                        exit 0
                    fi
                    command mkdir "$@"
                }
                source "$1"
            """,
                "test",
                str(SETUP_SCRIPT),
            ],
            env=self.env,
            text=True,
            capture_output=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn("Unexpected download", result.stderr)

    def test_conda_install_failure_is_not_ignored(self):
        result = subprocess.run(
            [
                "bash",
                "-ec",
                """
                sudo() { return 0; }
                apt-get() { return 0; }
                add-apt-repository() { return 0; }
                conda() { return 42; }
                curl() { return 99; }
                source "$1"
            """,
                "test",
                str(SETUP_SCRIPT),
            ],
            env=self.env,
            text=True,
            capture_output=True,
        )
        self.assertEqual(result.returncode, 42, result.stdout + result.stderr)
        self.assertNotIn("+++ curl", result.stderr)


if __name__ == "__main__":
    unittest.main()
