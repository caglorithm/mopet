import ray
import numpy as np
import pandas as pd
import tables
import tqdm

import copy
import itertools
import datetime
import logging
import time

from tables import NoSuchNodeError, NodeError
from mopet.exceptions import (
    ExplorationNotFoundError,
    Hdf5FileNotExistsError,
    ExplorationExistsError,
)


class Exploration:
    RUN_PREFIX = "run_"

    ##############################################
    ## USER FUNCTIONS
    ##############################################

    def __init__(
        self,
        function,
        explore_params,
        default_params=None,
        exploration_name=None,
        hdf_filename=None,
        num_cpus: int = None,
        num_gpus: int = None,
    ):
        """Defines a parameter exploration of a given `function`.

        :param function: Function to evaluate at each run
        :type function: function
        :param explore_params: Exploration parameters (individual) for each run
        :type explore_params: dict
        :param default_params: Default (shared) parameters to load for each run, optional, defaults to None
        :type default_params: dict
        :param exploration_name: Name of the run, will create a name if left empty, defaults to None
        :type exploration_name: str, optional
        :param hdf_filename: Filename of the hdf storage file, defaults to None
        :type hdf_filename: str, optional
        :param num_cpus: Number of desired CPU cores passed to ray, defaults to None
        :type num_cpus: int, optional
        :param num_gpus: Number of desired GPUs passed to ray, defaults to None
        :type num_gpus: int, optional
        :return: Exploration instance
        """

        self.function = function
        self.results = {}
        self.results_params = []

        if default_params is not None:
            self.default_params = copy.deepcopy(default_params)
            self.full_params = True
        else:
            self.default_params = None
            self.full_params = False

        self.explore_params = copy.deepcopy(explore_params)

        if exploration_name is None:
            exploration_name = "exploration" + datetime.datetime.now().strftime("_%Y_%m_%d_%HH_%MM_%SS")
        self.exploration_name = exploration_name

        if hdf_filename is None:
            hdf_filename = "exploration.h5"
        self.hdf_filename = hdf_filename

        self.dfResults = None

        # status
        self._hdf_open_for_reading = False

        # List of all parameter combinations generated when exploration starts
        self.explore_params_list = None

        # Dict with runId as keys and explored parameter dict as value.
        # Will be filled when exploration starts.
        self.run_params_dict = {}

        # Dict with runId as keys and explored parameter dict as value.
        # Will be filled when calling `load_results`.
        self.params = {}

        # Ray configuration
        self.num_gpus = num_gpus
        self.num_cpus = num_cpus

    def run(self):
        """Start parameter exploration.

        TODO: Pass kwargs in run() to the exploration function

        :raises ExplorationExistsError: if exploration with same name already exists in HDF5 file.
        """
        # Initialize ray
        self._init_ray(num_cpus=self.num_cpus, num_gpus=self.num_gpus)

        # Create a list of all combinations of parameters from explore_params
        self.explore_params_list = self._cartesian_product_dict(self.explore_params)

        # Initialize hdf storage
        self._pre_storage_routine()

        # -----------------------------
        # Set up all simulations
        # -----------------------------

        # remember the time
        start_time = time.time()
        # a unique id for each run
        run_id = 0

        # contains ray objects of each run
        ray_returns = {}
        # contains all exploration parameters of each run
        self.run_params_dict = {}
        logging.info(f"Starting {len(self.explore_params_list)} jobs.")
        # cycle through all parameter combinations
        for update_params in tqdm.tqdm(self.explore_params_list):

            if self.full_params and self.default_params is not None:
                # load the default parameters
                run_params = copy.deepcopy(self.default_params)
                # and update them with the explored parameters
                run_params.update(update_params)
            else:
                run_params = copy.deepcopy(update_params)

            # start all ray jobs and remember the ray object
            # pylint: disable=no-member
            ray_returns[run_id] = _ray_remote.remote(self.function, run_params)

            # store this runs explore parameters
            self.run_params_dict[run_id] = copy.deepcopy(update_params)

            # increment the run id
            run_id += 1

        # stop measuring time
        end_time = time.time() - start_time
        logging.info(f"Runs took {end_time} s to submit.")

        # -----------------------------
        # Reduce and store all results
        # -----------------------------

        # remember the time
        start_time = time.time()

        # cycle through all returned ray objects
        for run_id, ray_return in tqdm.tqdm(ray_returns.items()):
            # get the appropriate parameters for this run
            run_param = self.run_params_dict[run_id]
            # queue object for storage
            self._store_result(run_id, ray_return, run_param)
            # remove LOCAL_REFERENCE in form of ObjectId from ray's object store.
            ray_returns[run_id] = None

        # stop measuring time
        end_time = time.time() - start_time
        logging.info(f"Runs and storage took {end_time} s to complete.")

        # tear down hdf storage
        self._post_storage_routine()

        self._shutdown_ray()

    def load_results(self, filename=None, exploration_name=None, aggregate=True, all=False):
        """Load results from previous explorations. This function
        will open an HDF file and look for an exploration. It will
        create a Pandas `Dataframe` object (accessible through the
        attribute `.df`) with a list of all runs and their parameters.

        You can load the exploration results using following parameters:

        - If `aggregate==True`, all scalar results (such as `float`
        or `int`) from the exploration will be added to the Dataframe.
        - If `all==True`, then all results, including arrays and other
        types, will be saved in the attribute `.results`. This can take
        up a lot of RAM since all results will be available. Only
        use this option if you know that you have enough memory. Otherwise,
        you might want to skip this and load results separately using the
        method `.get_run()`.

        :param filename: Filename of HDF file, uses default filename or previously used filename if not given, defaults to None
        :type filename: str, optional
        :param exploration_name: Name of the exploration, same as the group names of the explorations in the HDF file, defaults to None
        :type exploration_name: str, optional
        :param aggregate: Aggregate scalar results into the results Dataframe. If this option is enabled, defaults to True
        :type aggregate: bool, optional
        :param all: Load all results into a dictionary available as the attribute `.results`. Can use a lot of RAM, defaults to False
        :type all: bool, optional
        :raises Hdf5FileNotExistsError: if file with `filename` does not exist.
        """
        if exploration_name is None:
            exploration_name = self.exploration_name
        else:
            self.exploration_name = exploration_name

        self._open_hdf(filename=filename)
        self._load_all_results(exploration_name, all=all)
        self._create_df()
        if aggregate:
            self._aggregate_results(exploration_name)
        self.close_hdf()

    def get_run(self, run_id=None, run_name=None, filename=None, exploration_name=None):
        """Get a single result from a previous exploration. This function
        will load a single result from the HDF file. Use this function
        if you want to avoid loading all results to memory, which you can
        do using `.load_results(all=True)`.

        Note: This function will open the HDF for reading but will not close
        it afterwards! This is to speed up many sequential loads but it also
        means that you have to close the HDF file yourself. You can do this
        by using `.close_hdf()`.

        :param run_id: Unique id of the run. Has to be given if run_name is not given, defaults to None
        :type run_id: int, optional
        :param run_name: The name of the run. Has to be given if run_id is not given, defaults to None
        :type run_name: str, optional
        :param filename: Filename of the HDF with previous exploration results. Previously used filename will be used if not given, defaults to None
        :type filename: str, optional
        :param exploration_name: Name of the exploration to load data from. Previously used exploration_name will be used if not given, defaults to None
        :type exploration_name: str, optional

        :return: Results of the run
        :rtype: dict
        :raises: NoSuchExplorationError if hdf5 file does not contain `exploration_name` group.
        """
        # get result by id or if not then by run_name (hdf_run)
        assert run_id is not None or run_name is not None, "Either use `run_id` or `run_name`."

        if exploration_name:
            self.exploration_name = exploration_name

        if run_id is not None:
            run_name = self.RUN_PREFIX + str(run_id)

        if not self._hdf_open_for_reading:
            self._open_hdf(filename)

        try:
            run_results_group = self.h5file.get_node("/" + self.exploration_name, "runs")[run_name]
        except NoSuchNodeError:
            raise ExplorationNotFoundError(
                "Exploration %s could not be found in HDF file %s".format(self.exploration_name, self.hdf_filename)
            )

        result = self._read_group_as_dict(run_results_group)
        return result

    def _cartesian_product_dict(self, input_dict):
        """Returns the cartesian product of the exploration parameters.

        :param input_dict: Parameter names and their values to explore
        :type input_dict: dict
        :return: List of dictionaries of all possible combinations
        :rtype: list
        """
        return [dict(zip(input_dict.keys(), values)) for values in itertools.product(*input_dict.values())]

    ##############################################
    ## MULTIPROCESSING
    ##############################################

    def _init_ray(self, num_cpus: int = None, num_gpus: int = None):
        """Initialize ray.

        :param num_cpus: Number of desired CPU cores used for this run, defaults to None
        :type num_cpus: int, optional
        :param num_gpus: Number of desired GPUs used for this run, defaults to None
        :type num_gpus: int, optional
        """
        if ray.is_initialized():
            self._shutdown_ray()

        ray.init(num_cpus=num_cpus, num_gpus=num_gpus)
        assert ray.is_initialized() is True, "Could not initialize ray."

    def _shutdown_ray(self):
        """Shutdown ray."""
        ray.shutdown()
        assert ray.is_initialized() is False, "Could not shutdown ray."

    ##############################################
    ## DATA STORAGE
    ##############################################

    def _store_dict_to_hdf(self, group, dict_data):
        """Stores a dictionary into a group of the hdf file.

        :param group: group in hdf file to store data in
        :type group: [type]
        :param dict_data: dictionary with data to store
        :type dict_data: dict
        """
        for r_key, r_val in dict_data.items():
            try:
                self.h5file.create_array(group, r_key, obj=r_val)
            except:
                logging.warning(f"Could not store dict entry {r_key} (type: {type(r_val)})")

    def _init_hdf(self):
        """Create hdf storage file and all necessary groups.

        :raises Hdf5FileNotExistsError, ExplorationAlreadyExists
        """
        try:
            self.h5file = tables.open_file(self.hdf_filename, mode="a")
        except IOError:
            raise Hdf5FileNotExistsError("Hdf5 file {} does not exist".format(self.hdf_filename))

        try:
            self.run_group = self.h5file.create_group("/", self.exploration_name)
        except NodeError as e:
            raise ExplorationExistsError(
                "Exploration with name {} already exists in HDF5 file {}".format(
                    self.exploration_name, self.hdf_filename
                )
            )

        # create group in which all data from runs will be saved
        self.runs_group = self.h5file.create_group(self.h5file.root[self.exploration_name], "runs")

        if self.default_params is not None:
            # create group in which all default parameters will be saved
            self.default_params_group = self.h5file.create_group(
                self.h5file.root[self.exploration_name], "default_params"
            )
            # store default parameters of this exploration
            self._store_dict_to_hdf(self.default_params_group, self.default_params)

        # create group in which exploration parameters will be saved
        self.explore_params_group = self.h5file.create_group(self.h5file.root[self.exploration_name], "explore_params")
        self._store_dict_to_hdf(self.explore_params_group, self.explore_params)

        # create group in which information about this run is saved
        # self.info_group = self.h5file.create_group("/", "info")

    def _pre_storage_routine(self):
        """Routines for preparing the hdf storage."""
        # initialize the hdf file
        self._init_hdf()

    def _post_storage_routine(self):
        """Routines for closing the hdf storage."""
        self.h5file.close()

    def _store_result(self, result_id, ray_object, run_params):
        """Resolves results from the ray object and stores the results.

        :param result_id: id of the run
        :type result_id: int
        :param ray_object: ray object
        :type ray_object: ray object
        :param run_params: explored parameters of the run
        :type run_params: dict
        """
        # set the name of this run for naming the hdf group
        run_result_name = self.RUN_PREFIX + str(result_id)

        # resolve the ray object and get the returned dictionary from the evaluation function
        result_dict = ray.get(ray_object)

        assert isinstance(result_dict, dict), f"Returned result must be a dictionary, is `{type(result_dict)}`."

        self._store_result_in_hdf(run_result_name, result_dict, run_params)

    def _store_result_in_hdf(self, run_result_name, result_dict, run_params):
        """Stores the results of a ray object of a single run and the parameters of the run.

        :param run_result_name: Name of the result
        :type run_result_name: str
        :param run_params: Explored parameters of the run
        :type run_params: dict
        """

        # create the results group
        run_results_group = self.h5file.create_group(self.runs_group, run_result_name)
        # store each item in the dictionary
        for r_key, r_val in result_dict.items():
            self.h5file.create_array(run_results_group, r_key, obj=r_val)

        # store parameters
        # create parameters group
        run_params_group = self.h5file.create_group(run_results_group, "params")
        # store the parameter dictionary
        self._store_dict_to_hdf(run_params_group, run_params)

    ##############################################
    ## READ DATA
    ##############################################

    def _create_df(self):
        """Create results dataframe and will it with columns according to the explored parameters.

        :return: dfResults
        :rtype: pd.DataFrame
        """
        logging.info("Creating new results DataFrame")
        self.explore_params = self._read_explore_params()
        self.dfResults = pd.DataFrame(columns=self.explore_params.keys(), index=self.run_ids, dtype=object)
        for key, value in self.params.items():
            self.dfResults.loc[key] = value
        return self.dfResults

    def _open_hdf(self, filename=None):
        """Open a hdf file as `.h5file` for reading later.

        :param filename: Filename of HDF file, defaults to None
        :type filename: str, optional
        :raises Hdf5FileNotExistsError
        """
        if filename is not None:
            self.hdf_filename = filename
        assert self.hdf_filename is not None, "No hdf filename was given or previously set."

        try:
            self.h5file = tables.open_file(self.hdf_filename, mode="r+")
        except OSError:
            raise Hdf5FileNotExistsError("Hdf5 file %s does not exist".format(self.hdf_filename))

        self._hdf_open_for_reading = True
        logging.info(f"{self.hdf_filename} opened for reading.")

    def close_hdf(self):
        """Close a previously opened HDF file."""
        self.h5file.close()
        self._hdf_open_for_reading = False
        logging.info(f"{self.hdf_filename} closed.")

    def _aggregate_results(self, exploration_name=None, arrays=True):
        """Go through all results saved in `.results` and store all floats in the results table.

        TODO: Direct reading from hdf without having to load it to memory, like in neurolib
        TODO: Storage of non-scalar values, like in neurolib

        :param exploration_name: [description], defaults to None
        :type exploration_name: [type], optional
        """
        nan_value = np.nan
        logging.info("Aggregating scalar results ...")
        for runId, parameters in tqdm.tqdm(self.dfResults.iterrows(), total=len(self.dfResults)):
            result = self.get_run(runId)
            for key, value in result.items():
                # we check for the type of the value and
                # save it to the datafram accordingly
                if isinstance(value, (float, int)):
                    self.dfResults.loc[runId, key] = value
                elif isinstance(value, np.ndarray) and arrays == True:
                    # to save a numpy array, convert column to object type
                    if key not in self.dfResults:
                        self.dfResults[key] = None
                    self.dfResults[key] = self.dfResults[key].astype(object)
                    self.dfResults.at[runId, key] = value
                else:
                    self.dfResults.loc[runId, key] = nan_value
        # drop nan columns
        self.dfResults = self.dfResults.dropna(axis="columns", how="all")

    def _load_all_results(self, exploration_name=None, all=True):
        """Load all results in hdf file into `.results` (if all=True). Load all explored parameters into `.params`.

        :param exploration_name: Name of the run, defaults to None
        :type exploration_name: str, optional
        :param all: Whether to load everything into ram or not, defaults to True
        :type all: bool, optional
        :raises Hdf5FileNotExistsError, ExplorationNotFoundError
        """
        if exploration_name is None:
            exploration_name = self.exploration_name
        else:
            self.exploration_name = exploration_name

        logging.info(f"Gettings runs of exploration ``{exploration_name}``")

        self.results = {}
        self.params = {}

        self.run_names = []
        self.run_ids = []

        hdf_runs_group = self.h5file.get_node("/" + self.exploration_name, "runs")
        logging.debug(f"Loading {len(list(hdf_runs_group))} results.")
        for hdf_run in tqdm.tqdm(hdf_runs_group, total=len(list(hdf_runs_group))):
            # get run name
            run_name = hdf_run._v_name
            self.run_names.append(run_name)
            # get id of run
            run_id = int(run_name[len(self.RUN_PREFIX) :])
            self.run_ids.append(run_id)
            # get results data
            if all:
                self.results[run_id] = self._read_group_as_dict(hdf_run)
            self.params[run_id] = self._get_run_parameters(hdf_run)

        logging.debug(f"{len(self.results)} results loaded to memory.")

    def _read_group_as_dict(self, group):
        """Read an HDF group and return a dictionary.

        :param group: hdf group
        :type group: tables.group.Group
        :return: Dictionary with data of the group.
        :rtype: dict
        """
        return_dict = {}
        # iterate through all arrays in that group
        for array in group:
            if isinstance(array, tables.array.Array):
                value = array.read()
                # unwrap 0-dim arrays
                if type(value) is np.ndarray and value.ndim == 0:
                    value = array.dtype.type(value)

                key = array._v_name
                return_dict[key] = value
        return return_dict

    def _get_run_parameters(self, hdf_run):
        """Get the parameters of a run

        :param hdf_run: Tables group
        :type hdf_run: tables.group.Group
        :return: Parameters as a dictionary
        :rtype: dict
        """
        run_params = {}
        if isinstance(hdf_run, tables.group.Group):
            run_params_group = self.h5file.get_node(hdf_run, "params")
            run_params = self._read_group_as_dict(run_params_group)
        return run_params

    def _read_explore_params(self):
        """Get explored parameters of from opened hdf file.

        :return: Dictionary with explored parameters
        :rtype: dict
        """
        explore_params_group = self.h5file.get_node("/" + self.exploration_name, "explore_params")
        self.explore_params = self._read_group_as_dict(explore_params_group)
        return self.explore_params

    @property
    def df(self):
        """Returns a dataframe with exploration results. Creates it from new if it doesn't exist yet.

        :return: Dataframe with exploration results
        :rtype: pandas.DataFrame
        """
        if hasattr(self, "dfResults"):
            if self.dfResults is None:
                return self._create_df()
            else:
                return self.dfResults
        else:
            return self._create_df()


@ray.remote
def _ray_remote(function, params):
    """This is a ray remote function (see ray documentation). It runs the `function` on each ray worker.

    :param function: function to be executed remotely.
    :type function: callable
    :param params: Parameters of the run.
    :type params: dict
    :return: ray object
    """
    r = function(params)
    return r
