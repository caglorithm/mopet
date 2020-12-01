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
            result_array = np.random.randn(np.random.randint(1, 131), np.random.randint(1, 5000))
            result = {"float_result": result_float, "array_result": result_array}
            return result

        params = {"x": 1.3, "y": 2.7, "z": 0.0}
        explore_params = {"x": np.linspace(-2, 2, 2), "y": np.linspace(-2, 2, 2)}

        ex = mopet.Exploration(eval_function, explore_params, params)

        ex.run()
        ex.load_results(all=True)
        # check if results are in the dataframe
        self.assertIn("array_result", ex.df)
        self.assertIn("float_result", ex.df)

    def test_hdf_file_not_exists(self):
        def run(params):
            return {"my_list": [1, 2, 3]}

        params = {"a": np.arange(0, 1, 0.5)}

        ex = mopet.Exploration(run, params, exploration_name="testC")
        # load results without running exploration
        # hdf5 file has not been created, this should throw a mopet exception.
        self.assertRaises(Hdf5FileNotExistsError, ex.load_results)

    def test_exploration_with_name_exists(self):
        """Test if exception is thrown for second run of exploration with fixed name.

        Exceptions protects against accidental override of results.
        """
        explore_params = {"x": np.linspace(-2, 2, 2), "y": np.linspace(-2, 2, 2)}

        # Initialize, run and load.
        ex = mopet.Exploration(lambda params: {}, explore_params, exploration_name="test")
        ex.run()
        ex.load_results(all=True)

        # Run again, but exploration already exists -> will throw exception.
        self.assertRaises(ExplorationExistsError, ex.run)

    def test_exploration_with_supported_types(self):
        """
        Tests if stored result dict containing all supported data types is loaded expected.

        Supported objects are:
            * NumPy array
            * Record or scalar
            * Homogeneous list or tuple
            * Integer, float, complex or bytes

        This is determined by supported types of array in pytables.

        See here for supported data types in PyTables (https://www.pytables.org/usersguide/datatypes.html).

        """
        result_dict = {
            "bool_true": True,
            "bool_false": False,
            "bool_array": [True, False],
            "int": np.int(1),
            "float": 42.0,
            "float_array": [1.0, 2.0],
            "complex": 4 + 3j,
            "tuple": (1, 2),
            "bytes": b"\x04",
        }

        def run(params):
            return result_dict

        ex = mopet.Exploration(run, {"a": np.arange(0, 1, 0.5)})
        ex.run()
        ex.load_results(all=True)

        print(ex.df)

        for key, result in ex.results.items():
            expected_result = dict(result_dict)

            # tuple is transformed to array by PyTables
            expected_result["tuple"] = [1, 2]
            self.assertDictEqual(result, expected_result)
