# STREAMS API Designdoc

9 March 2022


## Description

We want users to be able to query streams of data from one of 8 datasets (as described in the above doc). Additionally, we want users to be able to add their own datasets to turn into streams.


## System


### Requirements



* API for users to load data (based on time steps) from one of our supported datasets
    * Multiple data formats (`get_data`)
    * Dataloader for Pytorch (`get_loader`)
    * Ability to advance time steps (`step(..)`)
* Easy credential loading
    * Kaggle, imagenet, etc
* API for users to register datasets to turn into streams (TODO: flush out)
* Nice-to-have: caching layer (TODO: flush out)


### User Interface



* `STREAMSDataset` class
    * Constructor: takes in one of the dataset names & timestep to advance to
        * Other parameters (max time steps, sampling function, etc) have defaults designed by us
        * Optional parameter: data_dir (if users has already downloaded data and doesnt want to default to our env var), initial_buffer (int number of data points to initially load)
        * Instance variables: current timestep, permutation of data points
        * Downloads dataset if
    * `get_data`
        * backend: pytorch dataset, pandas dataframe, TFData
        * Returns data until current timestep
    * `get_loader`
        * Returns dataloader wrapped around get_data
    * `get_stream`
        * Test_window (int) – do we want a default here?
        * Returns avalanche stream for CL
    * `step`
        * Size: int (default is 1)
        * Lazily computes permutation order
        * Doesnt return anything
    * `vizualize`
        * Visualizes signals
        * Domain: int, values: ints
* Utilities
    * Visualize signals
    * See supported datasets & statistics
    * API credentials
    * Directory to download data to (env var)


### Execution Layer



* Waveform creation
    * See original doc for details
    * Computed in the constructor
        * Initially done for n timesteps
    * Lazily done when user calls `step`


### Storage Layer



* Raw data (disk)
* Domain matrices
    * in-memory int matrices
    * Hosted in S3 so we can reuse across people (?)
* Permutation (in-memory vector)
* Caching? TODO


## Notes – Avalanche Integration?



* Needs: give our own permutation
* Training stream, test stream. Each stream has a number of “experiences” where each experience has a number of data points. There is a “fixed_exp_assignment” param
* Reproducibility_data param? TODO investigate
* Lazy load doesnt work lol. Can revisit
