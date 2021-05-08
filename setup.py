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


extra_compile_args = []
extra_link_args = []

protoc = f"{PREFIX}/bin/protoc"

grpc_objects = [
    f"{PREFIX}/lib/libgrpc++.a",
    f"{PREFIX}/lib/libgrpc.a",
    f"{PREFIX}/lib/libgpr.a",
    f"{PREFIX}/lib/libaddress_sorting.a",
]

include_dirs = cpp_extension.include_paths() + [np.get_include(), f"{PREFIX}/include"]
libraries = []

if sys.platform == "darwin":
    extra_compile_args += ["-stdlib=libc++", "-mmacosx-version-min=10.14"]
    extra_link_args += ["-stdlib=libc++", "-mmacosx-version-min=10.14"]

    # Relevant only when c-cares is not embedded in grpc, e.g. when
    # installing grpc via homebrew.
    libraries.append("cares")
elif sys.platform == "linux":
    libraries.append("z")

grpc_objects.append(f"{PREFIX}/lib/libprotobuf.so")


rpc = cpp_extension.CppExtension(
    name="torchbeast.rpc",
    sources=[
        "torchbeast/rpc.cc",
        "torchbeast/server.cc",
        "torchbeast/client.cc",
        "torchbeast/rpc.pb.cc",
        "torchbeast/rpc.grpc.pb.cc",
    ],
    include_dirs=include_dirs,
    libraries=libraries,
    language="c++",
    extra_compile_args=["-std=c++17"] + extra_compile_args,
    extra_link_args=extra_link_args,
    extra_objects=grpc_objects,
)


def build_pb():
    # Hard-code client.proto for now.
    source = os.path.join(os.path.dirname(__file__), "torchbeast", "rpc.proto")
    output = source.replace(".proto", ".pb.cc")

    if os.path.exists(output) and (
        os.path.exists(source) and os.path.getmtime(source) < os.path.getmtime(output)
    ):
        return

    print("calling protoc")
    if (
        subprocess.call([protoc, "--cpp_out=torchbeast", "-Itorchbeast", "rpc.proto"])
        != 0
    ):
        sys.exit(-1)
    if (
        subprocess.call(
            protoc + " --grpc_out=torchbeast -Itorchbeast"
            " --plugin=protoc-gen-grpc=`which grpc_cpp_plugin`"
            " rpc.proto",
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


setuptools.setup(
    name="torchbeast",
    packages=["torchbeast"],
    version="0.1.21",
    ext_modules=[rpc],
    cmdclass={"build_ext": build_ext},
    test_suite="setup.test_suite",
)
