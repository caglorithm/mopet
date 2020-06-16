import unittest
import mopet
import numpy as np

from mopet.exceptions import Hdf5FileNotExistsError


class TestExploration(unittest.TestCase):
    """Test full exploration.
    """

    def test_exploration(self):
        def evalFunction(params):
            result_float = abs((params["x"] ** 2 + params["y"] ** 2) - 1)
            result_array = np.random.randn(
                np.random.randint(1, 131), np.random.randint(1, 5000)
            )
            result = {}
            result["float_result"] = result_float
            result["array_result"] = result_array
            return result

        params = {"x": 1.3, "y": 2.7, "z": 0.0}
        explore_params = {"x": np.linspace(-2, 2, 2), "y": np.linspace(-2, 2, 2)}

        ex = mopet.Exploration(evalFunction, explore_params, params)

        ex.run()
        ex.load_results(all=True)
        ex.df

    def test_hdf_file_not_exists(self):
        # TODO: clean up existing hdf5 file possibly created by other tests.
        def run(params):
            return {"my_list": [1, 2, 3]}

        params = {"a": np.arange(0, 1, 0.5)}

        ex = mopet.Exploration(run, params, exploration_name="testC")
        # load results without running exploration
        # hdf5 file has not been created, this should throw a mopet exception.
        self.assertRaises(Hdf5FileNotExistsError, ex.load_results)