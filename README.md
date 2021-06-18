
# TorchBeast
A PyTorch implementation of [IMPALA: Scalable Distributed
Deep-RL with Importance Weighted Actor-Learner Architectures
by Espeholt, Soyer, Munos et al.](https://arxiv.org/abs/1802.01561)

TorchBeast comes in two variants:
[MonoBeast](#getting-started-monobeast) and
[PolyBeast](#faster-version-polybeast). While
PolyBeast is more powerful (e.g. allowing training across machines),
it's somewhat harder to install. MonoBeast requires only Python and
PyTorch (we suggest using PyTorch version 1.2 or newer).

For further details, see our [paper](https://arxiv.org/abs/1910.03552).


## BibTeX

```
@article{torchbeast2019,
  title={{TorchBeast: A PyTorch Platform for Distributed RL}},
  author={Heinrich K\"{u}ttler and Nantas Nardelli and Thibaut Lavril and Marco Selvatici and Viswanath Sivakumar and Tim Rockt\"{a}schel and Edward Grefenstette},
  year={2019},
  journal={arXiv preprint arXiv:1910.03552},
  url={https://github.com/facebookresearch/torchbeast},
}
```

## Getting started: MonoBeast

MonoBeast is a pure Python + PyTorch implementation of IMPALA.

To set it up, create a new conda environment and install MonoBeast's
requirements:

```bash
$ conda create -n torchbeast
$ conda activate torchbeast
$ conda install pytorch -c pytorch
$ pip install -r requirements.txt
```

Then run MonoBeast, e.g. on the [Pong Atari
environment](https://gym.openai.com/envs/Pong-v0/):

```shell
$ python -m torchbeast.monobeast --env PongNoFrameskip-v4
```

By default, MonoBeast uses only a few actors (each with their instance
of the environment). Let's change the default settings (try this on a
beefy machine!):

```shell
$ python -m torchbeast.monobeast \
     --env PongNoFrameskip-v4 \
     --num_actors 45 \
     --total_steps 30000000 \
     --learning_rate 0.0004 \
     --epsilon 0.01 \
     --entropy_cost 0.01 \
     --batch_size 4 \
     --unroll_length 80 \
     --num_buffers 60 \
     --num_threads 4 \
     --xpid example
```

Results are logged to `~/logs/torchbeast/latest` and a checkpoint file is
written to `~/logs/torchbeast/latest/model.tar`.

Once training finished, we can test performance on a few episodes:

```shell
$ python -m torchbeast.monobeast \
     --env PongNoFrameskip-v4 \
     --mode test \
     --xpid example
```

MonoBeast is a simple, single-machine version of IMPALA.
Each actor runs in a separate process with its dedicated instance of
the environment and runs the PyTorch model on the CPU to create
actions. The resulting rollout trajectories
(environment-agent interactions) are sent to the learner. In the main
process, the learner consumes these rollouts and uses them to update
the model's weights.


## Faster version: PolyBeast

PolyBeast provides a faster and more scalable implementation of
IMPALA.

The easiest way to build and install all of PolyBeast's dependencies
and run it is to use Docker:

```shell
$ docker build -t torchbeast .
$ docker run --name torchbeast torchbeast
```

To run PolyBeast directly on Linux or MacOS, follow this guide.


### Installing PolyBeast

#### Linux

Create a new Conda environment, and install PolyBeast's requirements:

```shell
$ conda create -n torchbeast python=3.7
$ conda activate torchbeast
$ pip install -r requirements.txt
```

Install PyTorch either [from
source](https://github.com/pytorch/pytorch#from-source) or as per its
[website](https://pytorch.org/get-started/locally/) (select Conda).

PolyBeast also requires gRPC and other third-party software, which can
be installed by running:

```shell
$ git submodule update --init --recursive
```

Finally, let's compile the C++ parts of PolyBeast:

```
$ pip install nest/
$ python setup.py install
```

#### MacOS

Create a new Conda environment, and install PolyBeast's requirements:

```shell
$ conda create -n torchbeast
$ conda activate torchbeast
$ pip install -r requirements.txt
```

PyTorch can be installed as per its
[website](https://pytorch.org/get-started/locally/) (select Conda).

PolyBeast also requires gRPC and other third-party software, which can
be installed by running:

```shell
$ git submodule update --init --recursive
```

Finally, let's compile the C++ parts of PolyBeast:

```
$ pip install nest/
$ python setup.py install
```

### Running PolyBeast

To start both the environment servers and the learner process, run

```shell
$ python -m torchbeast.polybeast
```

The environment servers and the learner process can also be started separately:

```shell
python -m torchbeast.polybeast_env --num_servers 10
```

Start another terminal and run:

```shell
$ python3 -m torchbeast.polybeast_learner
```


## (Very rough) overview of the system

```
|-----------------|     |-----------------|                  |-----------------|
|     ACTOR 1     |     |     ACTOR 2     |                  |     ACTOR n     |
|-------|         |     |-------|         |                  |-------|         |
|       |  .......|     |       |  .......|     .   .   .    |       |  .......|
|  Env  |<-.Model.|     |  Env  |<-.Model.|                  |  Env  |<-.Model.|
|       |->.......|     |       |->.......|                  |       |->.......|
|-----------------|     |-----------------|                  |-----------------|
   ^     I                 ^     I                              ^     I
   |     I                 |     I                              |     I Actors
   |     I rollout         |     I rollout               weights|     I send
   |     I                 |     I                     /--------/     I rollouts
   |     I          weights|     I                     |              I (frames,
   |     I                 |     I                     |              I  actions
   |     I                 |     v                     |              I  etc)
   |     L=======>|--------------------------------------|<===========J
   |              |.........      LEARNER                |
   \--------------|..Model.. Consumes rollouts, updates  |
     Learner      |.........       model weights         |
      sends       |--------------------------------------|
     weights
```

The system has two main components, actors and a learner.

Actors generate rollouts (tensors from a number of steps of
environment-agent interactions, including environment frames, agent
actions and policy logits, and other data).

The learner consumes that experience, computes a loss and updates the
weights. The new weights are then propagated to the actors.


## Learning curves on Atari

We ran TorchBeast on Atari, using the same hyperparamaters and neural
network as in the [IMPALA
paper](https://arxiv.org/abs/1802.01561). For comparison, we also ran
the [open source TensorFlow implementation of
IMPALA](https://github.com/deepmind/scalable_agent), using the [same
environment
preprocessing](https://github.com/heiner/scalable_agent/releases/tag/gym). The
results are equivalent; see our paper for details.

![deep_network](./plot.png)


## Repository contents

`libtorchbeast`: C++ library that allows efficient learner-actor
communication via queueing and batching mechanisms. Some functions are
exported to Python using pybind11. For PolyBeast only.

`nest`: C++ library that allows to manipulate complex
nested structures. Some functions are exported to Python using
pybind11.

`tests`: Collection of python tests.

`third_party`: Collection of third-party dependencies as Git
submodules. Includes [gRPC](https://grpc.io/).

`torchbeast`: Contains `monobeast.py`, and `polybeast.py`,
`polybeast_learner.py` and `polybeast_env.py`.


## Hyperparamaters

Both MonoBeast and PolyBeast have flags and hyperparameters. To
describe a few of them:

* `num_actors`: The number of actors (and environment instances). The
  optimal number of actors depends on the capabilities of the machine
  (e.g. you would not have 100 actors on your laptop). In default
  PolyBeast this should match the number of servers started.
* `batch_size`: Determines the size of the learner inputs.
* `unroll_length`: Length of a rollout (i.e., number of steps that an
  actor has to be perform before sending its experience to the
  learner). Note that every batch will have dimensions
  `[unroll_length, batch_size, ...]`.


## Contributing

We would love to have you contribute to TorchBeast or use it for your
research. See the [CONTRIBUTING.md](CONTRIBUTING.md) file for how to help
out.

## License

TorchBeast is released under the Apache 2.0 license.
