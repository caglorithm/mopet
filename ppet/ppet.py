import ray
import copy
import numpy as np
import itertools
import tables
import datetime
import logging
import time


class Exploration:
    def __init__(
        self, function, default_params, explore_params, run_name=None, hdf_filename=None
    ):

        self.function = function
        self.results = {}
        self.results_params = []

        self.default_params = copy.deepcopy(default_params)
        self.explore_params = copy.deepcopy(explore_params)

        if run_name is None:
            run_name = "run" + datetime.datetime.now().strftime("_%Y_%m_%d_%HH_%MM_%SS")
        self.run_name = run_name

        if hdf_filename is None:
            hdf_filename = "exploration.h5"
        self.hdf_filename = hdf_filename

        self.run_id = None

        # self.init_ray()
        # self.init_hdf()

    def init_ray(self):
        ray.shutdown()
        ray.init()

    def init_hdf(self):
        self.h5file = tables.open_file(self.hdf_filename, mode="a")
        # self.info_group = self.h5file.create_group("/", "info")
        self.run_group = self.h5file.create_group("/", self.run_name)

        # create group in which all runs will be saved
        self.runs_group = self.h5file.create_group(
            self.h5file.root[self.run_name], "runs"
        )

        # create group in which all default parameters will be saved
        self.params_group = self.h5file.create_group(
            self.h5file.root[self.run_name], "params"
        )

        # store parameters
        for rkey, rval in self.default_params.items():
            try:
                ds = self.h5file.create_array(self.params_group, rkey, obj=rval)
            except:
                logging.warn(
                    f"Could not store default parameter {rkey}, type {type(rval)}"
                )

    def cartesian_product_dict(self, input_dict):
        return [
            dict(zip(input_dict.keys(), values))
            for values in itertools.product(*input_dict.values())
        ]

    def pre_run_routine(self):
        # initialize the hdf file
        self.init_hdf()

    def run(self):

        self.pre_run_routine()

        self.explore_params_list = self.cartesian_product_dict(self.explore_params)

        # run all simulations
        self.run_id = 0
        run_return = {}
        for update_params in self.explore_params_list:
            run_params = copy.deepcopy(self.default_params)
            run_params.update(update_params)
            start_time = time.time()
            result_dict = self.evaluate_run(run_params)
            end_time = time.time() - start_time
            print(end_time)
            # results.append(result_dict)
            run_return[self.run_id] = {}
            run_return[self.run_id]["params"] = update_params
            run_return[self.run_id]["result"] = result_dict
            self.run_id += 1

        # store all results
        for rkey, rval in run_return.items():
            self.store_result(rkey, rval)

        self.post_run_routine()

    def post_run_routine(self):
        self.h5file.close()

    def store_results_in_hdf(self, result_id, run_return):
        run_result_name = "run_" + str(result_id)

        result_dict = run_return["result"]
        update_params = run_return["params"]

        # store result
        run_results_group = self.h5file.create_group(self.runs_group, run_result_name)
        for rkey, rval in result_dict.items():
            ds = self.h5file.create_array(run_results_group, rkey, obj=rval)

        # store parameters
        run_params_group = self.h5file.create_group(run_results_group, "params")
        for rkey, rval in update_params.items():
            ds = self.h5file.create_array(run_params_group, rkey, obj=rval)

    def store_result(self, result_id, result_dict):
        self.store_results_in_hdf(result_id, result_dict)
        # store all results in a dictionary
        # self.store_result_in_dictionary(result_dict)

    @ray.remote
    def run_ray(self, params):
        return self.function(params)

    def evaluate_run(self, params):
        # if no ray:
        result = self.function(params)
        # if ray:
        # result = self.run_ray.remote(self, params)
        return result

    def store_result_in_dictionary(self, result_dict, params=None):
        self.results[self.run_id] = copy.deepcopy(result_dict)
