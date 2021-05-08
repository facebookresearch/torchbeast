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

# Open with Jupyter + jupytext:
# ```
# $ pip install jupyter
# $ pip install jupytext
# $ jupyter notebook
# ```
#
# See https://github.com/mwouts/jupytext

# +
import csv
import os

import torch

import matplotlib.pyplot as plt
import numpy as np


# +
def moving_average_cumsum(a, n=20):
    # Fast, but doesn't play well with NaNs
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1 :] / n


def moving_average(a, n=20):
    return np.convolve(a, np.ones((n,)) / n, mode="valid")


def rolling_xs_ys(xs, ys, window_size=20):
    return xs[window_size - 1 :], moving_average(ys, window_size)


# -


def mean_xs_ys(xys):
    res_xs = []
    res_ys = []

    cur_x, sum_y = next(xys)
    n = 1

    for x, y in xys:
        if x == cur_x:
            sum_y += y
            n += 1
            continue
        res_xs.append(cur_x)
        res_ys.append(sum_y / n)

        cur_x = x
        sum_y = y
        n = 1
    res_xs.append(cur_x)
    res_ys.append(sum_y / n)

    return res_xs, res_ys


# +
tableau20 = [
    (31, 119, 180),
    (174, 199, 232),
    (255, 127, 14),
    (255, 187, 120),
    (44, 160, 44),
    (152, 223, 138),
    (214, 39, 40),
    (255, 152, 150),
    (148, 103, 189),
    (197, 176, 213),
    (140, 86, 75),
    (196, 156, 148),
    (227, 119, 194),
    (247, 182, 210),
    (127, 127, 127),
    (199, 199, 199),
    (188, 189, 34),
    (219, 219, 141),
    (23, 190, 207),
    (158, 218, 229),
]

tableau20 = ["#%02x%02x%02x" % c for c in tableau20]


def str2color(s, colors=tableau20):
    return colors[hash(s) % len(colors)]


# -


def read(filename, xkey="step", ykey="episode_return", window_size=20):
    def xy_iter(filename):
        delimiters = {".tsv": "\t", ".csv": ","}
        with open(filename) as f:
            for row in csv.DictReader(
                f, delimiter=delimiters[os.path.splitext(filename)[-1]]
            ):
                yield float(row[xkey]), float(row[ykey])

    xs, ys = mean_xs_ys(xy_iter(filename))
    xs, ys = rolling_xs_ys(xs, ys)
    return filename, xs, ys


def plot(data, columns=1):
    keys = list(data.keys())
    rows = -(-len(keys) // columns)  # Ceil division.

    fig, axes = plt.subplots(rows, columns, figsize=(96, 40), squeeze=False)

    for r in range(rows):
        for c in range(columns):
            if not keys:
                break

            key = keys.pop(0)
            ax = axes[r][c]

            name, xs, ys = data[key]
            ax.title.set_text(name)

            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)

            color = str2color(name)
            linewidth = 10.0
            alpha = 1.0
            label = name

            ax.plot(xs, ys, color=color, linewidth=linewidth, alpha=alpha, label=label)
    fig.show()


cache = {}

font = {"family": "DejaVu Sans Mono", "weight": "normal", "size": 100}
plt.rc("font", **font)

cache["amidar"] = read("/tmp/torchbeast/logs.tsv")

plot(cache, 1)
