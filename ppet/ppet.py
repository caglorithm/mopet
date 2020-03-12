import ray
import copy
import numpy as np
import itertools
import tables
import datetime


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
        self.init_hdf()

    def init_ray(self):
        ray.shutdown()
        ray.init()

    def init_hdf(self):
        self.h5file = tables.open_file(
            self.hdf_filename, mode="a", title="ppet explorations"
        )
        # self.info_group = self.h5file.create_group("/", "info")
        self.run_group = self.h5file.create_group("/", self.run_name)
        self.results_group = self.h5file.create_group(
            self.h5file.root[self.run_name], "results"
        )

    def cartesian_product_dict(self, input_dict):
        return [
            dict(zip(input_dict.keys(), values))
            for values in itertools.product(*input_dict.values())
        ]

    def run(self):
        self.explore_params_list = self.cartesian_product_dict(self.explore_params)
        self.run_id = 0
        for p in self.explore_params_list:
            self.run_id += 1
            run_params = copy.deepcopy(self.default_params)
            run_params.update(p)
            result_dict = self.evaluate_run(run_params)
            self.store_result(result_dict, explore_params=p)

        self.post_run_routine()

    def post_run_routine(self):
        print(self.h5file)
        self.h5file.close()

    def store_results_in_hdf(self, result_dict):
        run_result_name = "run_" + str(self.run_id)
        run_results_group = self.h5file.create_group(
            self.results_group, run_result_name
        )
        for rkey, rval in result_dict.items():
            ds = self.h5file.create_array(run_results_group, rkey, obj=rval)

    def store_result(self, result_dict, explore_params):
        self.store_results_in_hdf(result_dict, explore_params)
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
