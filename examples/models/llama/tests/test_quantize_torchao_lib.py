# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import tempfile
import unittest
from unittest import mock

import torch

from executorch.examples.models.llama.source_transformation.quantize import (
    _load_torchao_aten_lib,
    _torchao_aten_lib_candidate_dirs,
)


class TorchaoAtenLibLookupTest(unittest.TestCase):
    """The lookup used to resolve to "./lib" relative to the current directory
    whenever CMAKE_INSTALL_PREFIX was unset, and then reported "Expected 1
    library but got 0". See https://github.com/pytorch/executorch/issues/8937.
    """

    def test_candidate_dirs_do_not_depend_on_cwd(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            dirs = _torchao_aten_lib_candidate_dirs()
        self.assertTrue(dirs)
        for directory in dirs:
            self.assertTrue(os.path.isabs(directory), directory)
        self.assertTrue(
            any(d.endswith(os.path.join("cmake-out", "lib")) for d in dirs), dirs
        )

    def test_install_prefix_is_searched_first(self):
        with mock.patch.dict(os.environ, {"CMAKE_INSTALL_PREFIX": "/tmp/prefix"}):
            dirs = _torchao_aten_lib_candidate_dirs()
        self.assertEqual(dirs[0], os.path.join("/tmp/prefix", "lib"))

    def test_missing_library_names_every_directory_searched(self):
        with self.assertRaises(FileNotFoundError) as caught:
            _load_torchao_aten_lib("libtorchao_ops_that_does_not_exist")
        message = str(caught.exception)
        for directory in _torchao_aten_lib_candidate_dirs():
            self.assertIn(directory, message)

    def test_library_found_in_a_candidate_dir_is_loaded(self):
        with tempfile.TemporaryDirectory() as prefix:
            libdir = os.path.join(prefix, "lib")
            os.makedirs(libdir)
            libpath = os.path.join(libdir, "libtorchao_ops_fake.so")
            open(libpath, "wb").close()
            with mock.patch.dict(os.environ, {"CMAKE_INSTALL_PREFIX": prefix}):
                with mock.patch.object(torch.ops, "load_library") as load_library:
                    _load_torchao_aten_lib("libtorchao_ops_fake")
            load_library.assert_called_once_with(libpath)


if __name__ == "__main__":
    unittest.main()
