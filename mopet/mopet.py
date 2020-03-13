import ray
import numpy as np
import pandas as pd
import tables

import copy
import itertools
import datetime
import logging
import time


class Exploration:

    RUN_PREFIX = "run_"

    def __init__(
        self,
        function,
        default_params,
        explore_params,
        exploration_name=None,
        hdf_filename=None,
    ):
        """Defines a parameter exploration of a given `function`.
        
        :param function: Function to evaluate at each run
        :type function: function
        :param default_params: Default (shared) parameters to load for each run
        :type default_params: dict
        :param explore_params: Exploration parameters (individual) for each run
        :type explore_params: dict
        :param exploration_name: Name of the run, will create a name if left empty, defaults to None
        :type exploration_name: str, optional
        :param hdf_filename: Filename of the hdf storage file, defaults to None
        :type hdf_filename: str, optional
        :return: Exploration instance
        """

        self.function = function
        self.results = {}
        self.results_params = []

        self.default_params = copy.deepcopy(default_params)
        self.explore_params = copy.deepcopy(explore_params)

        if exploration_name is None:
            exploration_name = "exploration" + datetime.datetime.now().strftime(
                "_%Y_%m_%d_%HH_%MM_%SS"
            )
        self.exploration_name = exploration_name

        if hdf_filename is None:
            hdf_filename = "exploration.h5"
        self.hdf_filename = hdf_filename

        self.run_id = None
        self.dfResults = None

    def run(self):
        """Start parameter exploration.
        """
        # Initialize ray
        self._init_ray()

        # Create a list of all combinations of parameters from explore_params
        self.explore_params_list = self._cartesian_product_dict(self.explore_params)

        # -----------------------------
        # Set up all simulations
        # -----------------------------

        # remember the time
        start_time = time.time()
        # a unique id for each run
        self.run_id = 0

        # contains ray objects of each run
        ray_returns = {}
        # contains all exploration parameters of each run
        self.run_params_dict = {}
        # cycle through all parameter combinations
        for update_params in self.explore_params_list:
            # load the default parameters
            run_params = copy.deepcopy(self.default_params)
            # and update them with the explored parameters
            run_params.update(update_params)

            # start all ray jobs and remember the ray object
            # pylint: disable=no-member
            ray_returns[self.run_id] = self._ray_remote.remote(self, run_params)

            # store this runs explore parameters
            self.run_params_dict[self.run_id] = copy.deepcopy(update_params)

            # increment the run id
            self.run_id += 1

        # stop measuring time
        end_time = time.time() - start_time
        logging.info(f"Runs took {end_time} s to submit.")

        # -----------------------------
        # Reduce and store all results
        # -----------------------------

        # initialize hdf storage
        self._pre_storage_routine()

        # remember the time
        start_time = time.time()

        # cycle through all returned ray objects
        for run_id, ray_return in ray_returns.items():
            # get the appropriate parameters for this run
            run_param = self.run_params_dict[run_id]
            # queue object for storage
            self._store_result(run_id, ray_return, run_param)

        # stop measuring time
        end_time = time.time() - start_time
        logging.info(f"Runs and storage took {end_time} s to complete.")

        # tear down hdf storage
        self._post_storage_routine()

    def _cartesian_product_dict(self, input_dict):
        """Returns the cartesian product of the exploration parameters.
        
        :param input_dict: Parameter names and their values to explore
        :type input_dict: dict
        :return: List of dictionaries of all possible combinations
        :rtype: list
        """
        return [
            dict(zip(input_dict.keys(), values))
            for values in itertools.product(*input_dict.values())
        ]

    ##############################################
    ## MULTIPROCESSING
    ##############################################

    @ray.remote
    def _ray_remote(self, params):
        """This is a ray remote function (see ray documentation). It runs the `function` on each ray worker.
        
        :param params: Parameters of the run.
        :type params: dict
        :return: ray object
        """
        r = self.function(params)
        return r

    def _init_ray(self):
        """Initialize ray.
        """
        ray.shutdown()
        ray.init()

    ##############################################
    ## DATA STORAGE
    ##############################################

    def _store_dict_to_hdf(self, group, dict_data):
        """Stores a dictionary into a group of the hdf file.
        
        :param group: group in hdf file to store data in
        :type group: [type]
        :param dict: dictionary with data to store
        :type dict: dict
        """
        for rkey, rval in dict_data.items():
            try:
                self.h5file.create_array(group, rkey, obj=rval)
            except:
                logging.warn(f"Could not store dict entry {rkey} (type: {type(rval)})")

    def _init_hdf(self):
        """Create hdf storage file and all necessary groups.
        """
        self.h5file = tables.open_file(self.hdf_filename, mode="a")
        self.run_group = self.h5file.create_group("/", self.exploration_name)

        # create group in which all data from runs will be saved
        self.runs_group = self.h5file.create_group(
            self.h5file.root[self.exploration_name], "runs"
        )

        # create group in which all default parameters will be saved
        self.default_params_group = self.h5file.create_group(
            self.h5file.root[self.exploration_name], "default_params"
        )
        # store default parameters of this exploration
        self._store_dict_to_hdf(self.default_params_group, self.default_params)

        # create group in which exploration parameters will be saved
        self.explore_params_group = self.h5file.create_group(
            self.h5file.root[self.exploration_name], "explore_params"
        )
        self._store_dict_to_hdf(self.explore_params_group, self.explore_params)

        # create group in which information about this run is saved
        # self.info_group = self.h5file.create_group("/", "info")

    def _pre_storage_routine(self):
        """Routines for preparing the hdf storage.
        """
        # initialize the hdf file
        self._init_hdf()

    def _post_storage_routine(self):
        """Routines for closing the hdf storage.
        """
        self.h5file.close()

    def _store_result(self, result_id, ray_object, run_params):
        """Resolves results from the ray object and stores the results.
        
        :param result_id: id of the run
        :type result_id: int
        :param ray_object: ray object
        :type ray_object: ray object
        :param run_param: explored parameters of the run
        :type run_param: dict
        """
        # set the name of this run for naming the hdf group
        run_result_name = self.RUN_PREFIX + str(result_id)

        # resolve the ray object and get the returned dictionary from the evaluation function
        result_dict = ray.get(ray_object)

        self._store_result_in_hdf(run_result_name, result_dict, run_params)
        # store all results in a dictionary
        # self._store_result_in_dictionary(result_id, result_dict)

    def _store_result_in_hdf(self, run_result_name, result_dict, run_params):
        """Stores the results of a ray object of a single run and the parameters of the run.
        
        :param run_result_name: Name of the result
        :type run_result_name: str
        :param run_return: ray object of the run
        :type run_return: ray object
        :param run_params: Explored parameters of the run
        :type run_params: dict
        """

        # create the results group
        run_results_group = self.h5file.create_group(self.runs_group, run_result_name)
        # store each item in the dictionary
        for rkey, rval in result_dict.items():
            self.h5file.create_array(run_results_group, rkey, obj=rval)

        # store parameters
        # create parameters group
        run_params_group = self.h5file.create_group(run_results_group, "params")
        # store the parameter dictionary
        self._store_dict_to_hdf(run_params_group, run_params)

    def _store_result_in_dictionary(self, result_id, result_dict):
        self.results[self.run_id] = copy.deepcopy(result_dict)

    ##############################################
    ## READ DATA
    ##############################################

    def load_results(self, exploration_name=None):
        self._open_hdf()
        self._hdf_get_data_from_exploration(exploration_name)
        self._close_hdf()

    def _open_hdf(self):
        self.h5file = tables.open_file(self.hdf_filename, mode="r")
        logging.info(f"{self.hdf_filename} opened for reading.")

    def _close_hdf(self):
        self.h5file.close()
        logging.info(f"{self.hdf_filename} closed.")

    def _hdf_get_data_from_exploration(self, exploration_name=None):
        if exploration_name is None:
            exploration_name = self.exploration_name
        else:
            self.exploration_name = exploration_name

        logging.info(f"Gettings runs of exploration ``{exploration_name}``")

        self.results = {}
        self.params = {}

        self.run_names = []
        self.run_ids = []

        runs_group = self.h5file.get_node("/" + self.exploration_name, "runs")

        for child in runs_group:
            # get run name
            run_name = child._v_name
            self.run_names.append(run_name)
            # get id of run
            run_id = int(run_name[len(self.RUN_PREFIX) :])
            self.run_ids.append(run_id)
            # get results data
            self._get_run_results(child, run_id)

        logging.info(f"{len(self.results)} results loaded.")

    def _get_run_results(self, hdf_run, run_id=None):
        # hack to get the name of the group, I bet there is a better way
        if run_id is None:
            run_id = str(hdf_run._v_name)
        self.results[run_id] = {}
        self.params[run_id] = {}
        self.results[run_id] = self._read_group_as_dict(hdf_run)
        self.params[run_id] = self._get_run_parameters(hdf_run)
        # for child in hdf_run:
        #     # if the child is another group (parameters)
        #     if isinstance(child, tables.group.Group):
        #         # param = {}
        #         param = self._read_group_as_dict(child)
        #         # if there was at leas one array in the group
        #         if len(param):
        #             self.params[run_id] = param

    def _read_group_as_dict(self, group):
        return_dict = {}
        # iterate through all arrays in that group
        for array in group:
            if isinstance(array, tables.array.Array):
                value = array.read()
                # convert 0-dim arrays to floats
                if value.ndim == 0:
                    value = np.float(value)
                key = array._v_name
                return_dict[key] = value
        return return_dict

    def _get_run_parameters(self, hdf_run):
        run_params = {}
        if isinstance(hdf_run, tables.group.Group):
            run_params_group = self.h5file.get_node(hdf_run, "params")
            run_params = self._read_group_as_dict(run_params_group)
        return run_params

    def _read_explore_params(self):
        explore_params_group = self.h5file.get_node(
            "/" + self.exploration_name, "explore_params"
        )
        self.explore_params = self._read_group_as_dict(explore_params_group)
        return self.explore_params

    def _create_df(self):
        self.explore_params = self._read_explore_params()
        self.dfResults = pd.DataFrame(
            columns=self.explore_params.keys(), index=self.run_ids, dtype=object
        )
        for key, value in self.params.items():
            self.dfResults.loc[key] = value
        return self.dfResults

    @property
    def df(self):
        if self.dfResults is None:
            logging.info("Creating new results DataFrame")
            return self._create_df()
        else:
            return self.dfResults
