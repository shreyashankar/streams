.. streamscl documentation master file, created by
   sphinx-quickstart on Fri Apr 29 12:16:30 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Introduction
============

``streamscl`` is a Python package to generate streams of publicly available data. 9 static datasets (with real-world OOD or subpopulation data) are used to create ``streamscl``, a benchmark of 9 data streams that cover regression, classification, and generation tasks across many modalities: images, text, speech, time series, tabular data, and human-agent interactions.


Motivation
##########

ML practitioners in the wild often face a constant stream of data with an ever-changing distribution. To maintain performance, they must adapt to the data stream, a problem called online continual learning (OCL). Despite its practical relevance, existing benchmarks for OCL suffer from many issues: (1) distribution shifts are abrupt and known beforehand, while in the real world, they can also be gradual and arrive without forewarning; (2) shifts are synthetic and unrealistic (e.g., pixel permutations); (3) benchmarks only cover a single modality and task type (typically few-class image classification). To address these issues, **we introduce a new multimodal benchmark for OCL called streamscl.** Given the scarcity of publicly available data streams, and the potential infrequency of adverse shifts that are worth simulating, we first propose a method to controllably generate streaming data from static data. Then taking static datasets containing real out-of-domain data (e.g., IWildCam) or multiple subpopulations (e.g., CivilComments), we apply this approach across a variety of modalities -- images, text, speech, time series, tabular data, and human-agent interactions -- to create ``streamscl``.

Usage
#####

The following datasets are supported:

- iwildcam
- civilcomments
- poverty
- jeopardy
- airquality
- zillow
- coauthor
- census
- nuimages

Data storage
************

The default data storage folder is ``~/.streams_data``. You can override it by setting the environment variables ``DOWNLOAD_HOME`` and ``DOWNLOAD_PREFIX``. If you do not already have the data downloaded, the ``STREAMSDataset`` utilities will do so for you -- with the exception of ``nuimages``, which requires manual download.

To download ``nuimages``, manually download the Metadata and Samples from NuImages_. Extract them to a folder named ``nuimages`` in the ``~/.streams_data`` or ``DOWNLOAD_HOME`` folder. Make sure the directory structure of ``nuimages`` looks like:

- ``nuimages/``
   - ``samples/``
   - ``v1.0-test/``
   - ``v1.0-train/``
   - ``v1.0-val/``

.. _NuImages: https://www.nuscenes.org/nuimages#download

Dataset configuration
*********************

To use the ``STREAMSDataset`` class, you can either pass in your own stream configuration parameters or use a preset stream configuration. To use the preset configuration, you can do the following:

.. code-block::

   from streams import STREAMSDataset

   dataset_name = "iwildcam"
   ds = STREAMSDataset.from_config(dataset_name)

To pass in your own stream configuration parameters, you can do the following:

.. code-block::

   from streams import STREAMSDataset

   dataset_name = "iwildcam"
   ds = STREAMSDataset(
      dataset_name,
      T=10,
      gamma=0.5,
      num_peaks=5,
      start_max=10,
      duration=1,
      log_step=1,
      inference_window=1
   )

See parameter descriptions in :func:`streams.utils.create_logits` for more details. The ``inference_window`` parameter tells how many steps you want to be able to "look ahead" in the stream as a "test set" to evaluate on.

Dataset iteration
*****************

Any instance of the ``STREAMSDataset`` class has a ``step`` property that tells you what timestep you are in the stream. Initially, the ``step`` is 0. To iterate through the dataset, you can call the following methods:

.. code-block::

   from streams import STREAMSDataset

   ds = STREAMSDataset.from_config("iwildcam")

   train_data, test_data = ds.get_data(include_test=True)
   for step, (x, y) in enumerate(train_data):
      print(step, x, y)

   # Or load the data into Pytorch data loaders
   train_dl, test_dl = ds.get_loaders(batch_size=32, include_test=True)

   # Advance time step in the stream
   ds.advance(step_size=1)

   # Reset to the beginning
   ds.reset()

The ``STREAMSDataset`` class also has some helper methods:

.. code-block::

   from streams import STREAMSDataset

   ds = STREAMSDataset.from_config("iwildcam")

   # Visualize signals for how likely domain values are to occur
   ds.visualize(domain_type_index=0, domain_value_indices=[0, 1, 2])

   # Get data from specific points in the stream
   ds.get(step_indices=[4, 5], future_ok=True)

   # Get length of dataset
   len(ds)

Check out :class:`streams.STREAMSDataset` for more details.

Avalanche Integration
*********************

TODO(shreyashankar)

Training Example
#################

TODO(shreyashankar)


Contributing
############

TODO(shreyashankar)



.. toctree::
   :maxdepth: 2
   :caption: Contents:
