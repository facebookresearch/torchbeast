# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#
#   CXX=c++ python3 setup.py build develop
# or
#   CXX=c++ pip install . -vv
#
# Potentially also set TORCHBEAST_LIBS_PREFIX.

import os
import subprocess
import sys
import unittest

import numpy as np
import setuptools
from torch.utils import cpp_extension


PREFIX = os.getenv("CONDA_PREFIX")

if os.getenv("TORCHBEAST_LIBS_PREFIX"):
    PREFIX = os.getenv("TORCHBEAST_LIBS_PREFIX")
if not PREFIX:
    PREFIX = "/usr/local"


def build_pb():
    protoc = f"{PREFIX}/bin/protoc"

    # Hard-code client.proto for now.
    source = os.path.join(os.path.dirname(__file__), "libtorchbeast", "rpcenv.proto")
    output = source.replace(".proto", ".pb.cc")

    if os.path.exists(output) and (
        os.path.exists(source) and os.path.getmtime(source) < os.path.getmtime(output)
    ):
        return

    print("calling protoc")
    if (
        subprocess.call(
            [protoc, "--cpp_out=libtorchbeast", "-Ilibtorchbeast", "rpcenv.proto"]
        )
        != 0
    ):
        sys.exit(-1)
    if (
        subprocess.call(
            protoc + " --grpc_out=libtorchbeast -Ilibtorchbeast"
            " --plugin=protoc-gen-grpc=`which grpc_cpp_plugin`"
            " rpcenv.proto",
            shell=True,
        )
        != 0
    ):
        sys.exit(-1)


def test_suite():
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover("tests", pattern="*_test.py")
    return test_suite


class build_ext(cpp_extension.BuildExtension):
    def run(self):
        build_pb()
        cpp_extension.BuildExtension.run(self)


def main():
    extra_compile_args = []
    extra_link_args = []

    grpc_objects = [
        f"{PREFIX}/lib/libgrpc++.a",
        f"{PREFIX}/lib/libgrpc.a",
        f"{PREFIX}/lib/libgpr.a",
        f"{PREFIX}/lib/libaddress_sorting.a",
    ]

    include_dirs = cpp_extension.include_paths() + [
        np.get_include(),
        f"{PREFIX}/include",
    ]
    libraries = []

    if sys.platform == "darwin":
        extra_compile_args += ["-stdlib=libc++", "-mmacosx-version-min=10.14"]
        extra_link_args += ["-stdlib=libc++", "-mmacosx-version-min=10.14"]

        # Relevant only when c-cares is not embedded in grpc, e.g. when
        # installing grpc via homebrew.
        libraries.append("cares")
    elif sys.platform == "linux":
        libraries.append("z")

    grpc_objects.append(f"{PREFIX}/lib/libprotobuf.a")

    libtorchbeast = cpp_extension.CppExtension(
        name="libtorchbeast._C",
        sources=[
            "libtorchbeast/libtorchbeast.cc",
            "libtorchbeast/actorpool.cc",
            "libtorchbeast/rpcenv.cc",
            "libtorchbeast/rpcenv.pb.cc",
            "libtorchbeast/rpcenv.grpc.pb.cc",
        ],
        include_dirs=include_dirs,
        libraries=libraries,
        language="c++",
        extra_compile_args=["-std=c++17"] + extra_compile_args,
        extra_link_args=extra_link_args,
        extra_objects=grpc_objects,
    )

    setuptools.setup(
        name="libtorchbeast",
        packages=["libtorchbeast"],
        version="0.0.14",
        ext_modules=[libtorchbeast],
        cmdclass={"build_ext": build_ext},
        test_suite="setup.test_suite",
    )


if __name__ == "__main__":
    main()
