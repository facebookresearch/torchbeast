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
import pathlib
import subprocess
import sys

import setuptools
from setuptools.command import build_ext
from distutils import spawn
from distutils import sysconfig


class CMakeBuild(build_ext.build_ext):
    def run(self):  # Necessary for pip install -e.
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        source_path = pathlib.Path(__file__).parent.resolve()
        output_path = (
            pathlib.Path(self.get_ext_fullpath(ext.name))
            .parent.joinpath("libtorchbeast")
            .resolve()
        )

        os.makedirs(self.build_temp, exist_ok=True)

        build_type = "Debug" if self.debug else "Release"
        generator = "Ninja" if spawn.find_executable("ninja") else "Unix Makefiles"

        cmake_cmd = [
            "cmake",
            str(source_path),
            "-G%s" % generator,
            "-DPYTHON_SRC_PARENT=%s" % source_path,
            "-DPYTHON_EXECUTABLE=%s" % sys.executable,
            "-DPYTHON_INCLUDE_DIR=%s" % sysconfig.get_python_inc(),
            "-DPYTHON_LIBRARY=%s" % sysconfig.get_config_var("LIBDIR"),
            "-DCMAKE_BUILD_TYPE=%s" % build_type,
            "-DCMAKE_INSTALL_PREFIX=%s" % sys.base_prefix,
            "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=%s" % output_path,
        ]

        build_cmd = ["cmake", "--build", ".", "--parallel"]
        install_cmd = build_cmd + ["--target", "install"]

        try:
            subprocess.check_call(cmake_cmd, cwd=self.build_temp)
            subprocess.check_call(build_cmd, cwd=self.build_temp)
            subprocess.check_call(install_cmd, cwd=self.build_temp)
        except subprocess.CalledProcessError:
            # Don't obscure the error with a setuptools backtrace.
            sys.exit(1)


def main():
    setuptools.setup(
        name="libtorchbeast",
        packages=["libtorchbeast"],
        ext_modules=[setuptools.Extension("libtorchbeast", sources=[])],
        package_dir={"libtorchbeast": "src/py/"},
        install_requires=["torch>=1.4.0"],
        version="0.0.20",
        cmdclass={"build_ext": CMakeBuild},
        zip_safe=False,
    )


if __name__ == "__main__":
    main()
