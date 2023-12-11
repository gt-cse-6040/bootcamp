# Tester framework

The Tester framework provides a generic interface for testing exercise solution functions. The provided Tester class is designed to be used with pre-built test cases stored in encrypted pickle files. Solution funcitons are run aginst the test cases and an error is raised if the outputs do not match or execution otherwise fails.

## Features
### Supported Data Types

- Basic Python types:
    - `str`, `int`, `bool` - must match exactly
    - `float` - must match within confugurable tolerance
- Basic Python collections: - all child objects must match
    - `list`, `tuple` - order must match
    - `set` - must contain same elements
    - `dict` - must have same keys and matching values
- Numpy:
    - `np.ndarray` - all values must match; array must be sorted correctly
- Pandas
    - `pd.DataFrame` - all values, index, and column must match; configurable column type and sort checking
    - `pd.Series` - all values, index must match
### Additional features

- Modification check: option to verify solution does not modify its inputs. 
- Data type check: option to verify top level data types of outputs.

## Usage

To use the framework instanciate the `Tester` class, then call the `run_test()` method. The constructor takes 3 arguments:
- conf - `dict` with configuration details. See [Configuration](#configuration)
- key - encryption key to decrypt the test case file
- path - directory for test case file

```
from tester_fw.testers import Tester

tester = Tester(conf, 
                key=b'ZYnA9mj-ZX9s_ZQ0z1EmzeeG3N5Rz8XYF_2Uajb0u9I=', 
                path='resource/asnlib/publicdata/')

    tester.run_test()
```
## Configuration

The `conf` argument of `Tester()` is expected to have the following structure.

```
conf = {
    'case_file':'tc_0', 
    'func': some_func, 
    'inputs':{ 
        'some_param':{                  # name of param.
            'dtype':'',                     # data type of param.
            'check_modified':True,          # whether to check if input modified
        },
    },
    'outputs':{
        'output_0':{                    # name of output (default 'object_i')
            'dtype':'dict',                 # type of output
            'check_dtype': True,            # whether to check the data type of output
            'check_col_dtypes': True,       # whether to check `DataFrame` column types
            'check_row_order': True,        # whether to enforce `DataFrame` sorting as a requirement
            'float_tolerance': 10 ** (-6)   # tolerance for floating point calculations
        },
    }
}
```

## Returning outputs to the user

There is an option to pass the variables from the test cases back to the student taking the exam. The `get_test_vars()` method returns the test result from the latest `run_test` execution.

- `input_vars` - original input variables.
- `original_input_vars` - copy of input variables to verify that funciton does not modify inputs.
- `returned_output_vars` - result of function run with `input_vars` used as parameters.
- `true_output_vars` - expected output pulled from test cases.

```
tester = Tester(conf, key=b'ZYnA9mj-ZX9s_ZQ0z1EmzeeG3N5Rz8XYF_2Uajb0u9I=', path='resource/asnlib/publicdata/')
for _ in range(20):
    try:
        tester.run_test()
        (input_vars, original_input_vars, returned_output_vars, true_output_vars) = tester.get_test_vars()
    except:
        (input_vars, original_input_vars, returned_output_vars, true_output_vars) = tester.get_test_vars()
        raise
```

Students are able to use these variables for debugging.
