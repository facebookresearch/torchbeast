#!/usr/bin/env bash

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

set -e
set -x

if [ -z ${GRPC_DIR+x} ]; then
    GRPC_DIR=$(pwd)/third_party/grpc;
fi

PREFIX=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}

NPROCS=$(getconf _NPROCESSORS_ONLN)

pushd ${GRPC_DIR}

## This requires libprotobuf to be installed in the conda env.
## Otherwise, we could also do this:
# cd ${GRPC_DIR}/third_party/grpc/third_party/protobuf
# ./autogen.sh && ./configure --prefix=${PREFIX}
# make && make install && ldconfig

# Make make find libprotobuf
export CPATH=${PREFIX}/include:${CPATH}
export LIBRARY_PATH=${PREFIX}/lib:${LIBRARY_PATH}
export LD_LIBRARY_PATH=${PREFIX}/lib:${LD_LIBRARY_PATH}

make -j ${NPROCS} prefix=${PREFIX} \
     HAS_SYSTEM_PROTOBUF=true HAS_SYSTEM_CARES=false
make prefix=${PREFIX} \
     HAS_SYSTEM_PROTOBUF=true HAS_SYSTEM_CARES=false install

popd
