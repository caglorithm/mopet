import unittest
import mopet
import numpy as np
import glob
import os

from mopet.exceptions import Hdf5FileNotExistsError, ExplorationExistsError


class TestExploration(unittest.TestCase):
    """ Test various exploration cases. """

    def tearDown(self) -> None:
        """ Clean up files created during tests. """

        # Remove h5 files.
        h5_files = glob.glob("*.h5", recursive=False)
        for f in h5_files:
            os.remove(f)

    def test_exploration(self):
        def eval_function(params):
            result_float = abs((params["x"] ** 2 + params["y"] ** 2) - 1)
            result_array = np.random.randn(
                np.random.randint(1, 131), np.random.randint(1, 5000)
            )
            result = {"float_result": result_float, "array_result": result_array}
            return result

        params = {"x": 1.3, "y": 2.7, "z": 0.0}
        explore_params = {"x": np.linspace(-2, 2, 2), "y": np.linspace(-2, 2, 2)}

        ex = mopet.Exploration(eval_function, explore_params, params)

        ex.run()
        ex.load_results(all=True)
        ex.df

    def test_hdf_file_not_exists(self):
        def run(params):
            return {"my_list": [1, 2, 3]}

        params = {"a": np.arange(0, 1, 0.5)}

        ex = mopet.Exploration(run, params, exploration_name="testC")
        # load results without running exploration
        # hdf5 file has not been created, this should throw a mopet exception.
        self.assertRaises(Hdf5FileNotExistsError, ex.load_results)

    def test_exploration_with_name_exists(self):
        """ Test if exception is thrown for second run of exploration with fixed name.

        Exceptions protects against accidental override of results.
        """
        explore_params = {"x": np.linspace(-2, 2, 2), "y": np.linspace(-2, 2, 2)}

        # Initialize, run and load.
        ex = mopet.Exploration(
            lambda params: {}, explore_params, exploration_name="test"
        )
        ex.run()
        ex.load_results(all=True)

        # Run again, but exploration already exists -> will throw exception.
        self.assertRaises(ExplorationExistsError, ex.run)
