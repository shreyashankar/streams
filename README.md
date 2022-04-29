# STREAMS: A Benchmark of Naturalistic Streaming Data for Online Continual Learning

[![STREAMS](https://github.com/shreyashankar/streams/actions/workflows/python-package.yml/badge.svg)](https://github.com/shreyashankar/streams/actions/workflows/python-package.yml/badge.svg)
[![Documentation Status](https://readthedocs.org/projects/streamscl/badge/?version=latest)](https://streamscl.readthedocs.io/en/latest/?badge=latest)

TODO: fill out :)

## Avalanche Compatibility Instructions

The continual learning library Avalanche provides a stream dataset abstraction for classification tasks and baseline CL algorithms. Our API has a `get_benchmark` method that returns an Avalanche benchmark. For example:

```python
from streams import STREAMSDataset

ds = STREAMSDataset("iwildcam", T=100)
ds.advance(10)
benchmark = ds.get_benchmark()
```

Our benchmarks have both a train and test stream. The train stream has only 1 experience. The [Avalanche tutorial](https://avalanche.continualai.org/from-zero-to-hero-tutorial/03_benchmarks) has documentation for how to work with a benchmark, but here is some boilerplate code:

```python
# recovering the train and test streams
train_stream = benchmark.train_stream
test_stream = benchmark.test_stream

# iterating over the train stream
# only one experience in the stream
for experience in train_stream:
  print("Start of task ", experience.task_label)
  print('Classes in this task:', experience.classes_in_this_experience)

  # The current Pytorch training set can be easily recovered through the
  # experience
  current_training_set = experience.dataset
  # ...as well as the task_label
  print('Task {}'.format(experience.task_label))
  print('This task contains', len(current_training_set), 'training examples')

  # we can recover the corresponding test experience in the test stream
  current_test_set = test_stream[experience.current_experience].dataset
  print('This task contains', len(current_test_set), 'test examples')
```
