from . import ExerciseTester

class Tester(ExerciseTester):
    def __init__(self, conf, key, path):
        import dill as pickle
        from cryptography.fernet import Fernet
        fernet = Fernet(key)
        with open(f"{path}{conf['case_file']}", 'rb') as fin:
            self.cases = pickle.loads(fernet.decrypt(fin.read()))
        self.func = conf['func']
        self.conf_inputs = conf['inputs']
        self.conf_outputs = conf['outputs']
        self.prevent_mod = True
        self.input_vars = dict()
        self.original_input_vars = dict()
        self.returned_output_vars = dict()
        self.true_output_vars = dict()
    
    def copy_vars(self):
        from copy import deepcopy
        self.original_input_vars = {k: deepcopy(v) 
                                        for k, v in self.input_vars.items()
                                        if self.conf_inputs[k]['check_modified']}   
    
    def check_modified(self):
        for var_name in self.input_vars:
            if not self.conf_inputs[var_name]['check_modified']: continue
            import pandas as pd
            import numpy as np
            if isinstance(self.original_input_vars[var_name], pd.DataFrame):
                try:
                    pd.testing.assert_frame_equal(self.original_input_vars[var_name], self.input_vars[var_name])
                except AssertionError:
                    assert False, f'Your solution modified the input variable `{var_name}`. You can use the testing variables for debugging.'
            elif isinstance(self.original_input_vars[var_name], np.ndarray):
                assert (self.input_vars[var_name] == self.original_input_vars[var_name]).all(), f'Your solution modified the input variable `{var_name}`. You can use the testing variables for debugging.'
            else:
                assert self.input_vars[var_name] == self.original_input_vars[var_name], f'Your solution modified the input variable `{var_name}`. You can use the testing variables for debugging.'
    
    def run_test(self, func=None):
        return super().run_test(self.func)

    def build_vars(self):
        from tester_fw.test_utils import dfs_to_conn
        from random import choice
        case = choice(self.cases)
        for input_key, input_dict in self.conf_inputs.items():
            if input_dict['dtype'] == 'db':
                temp_conn = dfs_to_conn(case[input_key])
                self.input_vars[input_key] = temp_conn
            else:
                self.input_vars[input_key] = case[input_key]
        for output_key in self.conf_outputs:
            self.true_output_vars[output_key] = case[output_key]

    def run_func(self, func):
        out = func(**self.input_vars)
        if not isinstance(out, tuple):
            out = (out,)
        out_keys = sorted(self.conf_outputs, key=lambda x: self.conf_outputs[x]['index'])
        self.returned_output_vars = dict(zip(out_keys, out))
    
    def check_type(self):
        import numpy as np
        import pandas as pd
        type_options = {
            'int': (int, np.integer), 'float': (float, np.floating), 'bool': (bool, np.bool_), 'str': (str,),
            'dict': (dict,), 'set': (set,), 'tuple': (tuple,),
            'df': (pd.DataFrame,), 'series': (pd.Series,),
            'array': (np.ndarray,)
        }
        for out_key, out_dict in self.conf_outputs.items():
            t = type_options.get(out_dict['dtype'])
            if (t is None) or (t == ''): continue
            o = self.returned_output_vars[out_key]
            assert isinstance(o, t), f'Type {" or ".join(str(t_) for t_ in t)} is required for {out_key} but {str(type(o))} was returned.'
             
    def check_matches(self):
        from . import test_utils
        for out_key, out_dict in self.conf_outputs.items():
            test_var = self.returned_output_vars[out_key]
            assert test_utils.compare_copies(a=test_var,
                                            b=self.true_output_vars[out_key],
                                            tol=out_dict['float_tolerance'],
                                            sort_df=not out_dict['check_row_order'],
                                            col_type=out_dict['check_col_dtypes']), \
            f'''
Output for {out_key} is incorrect.
The returned result is available as `returned_output_vars['{out_key}']`
The expected result is available as `true_output_vars['{out_key}']`
            '''
