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


if [ $# -eq 0 ]
then
    tag='latest'
else
    tag=$1
fi

docker build -t torchbeast/ci-polybeast-cpu37:$tag -f .circleci/docker/polybeast/Dockerfile_cpu37 .
docker push torchbeast/ci-polybeast-cpu37:$tag
