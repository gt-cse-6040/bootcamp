class ExerciseTester():
    '''
    Object for generating inputs/outputs and testing solutions to CSE6040 exercises. ExerciseTester is expected to be extended by subclasses to handle testing individual exercises.
    '''
    ###
    ### These methods are "generic" and should work for most exercises. Think long and hard before overriding in a subclass.
    ###

    # Do not override this method
    def __init__(self,
        prevent_mod=True,
        tc_path=None):
        '''
        Instantiate an ExerciseTester object.
        '''
        print(f'initializing {__name__}')
        self.input_vars = dict()
        self.original_input_vars = dict()
        self.returned_output_vars = dict()
        self.true_output_vars = dict()
        self.prevent_mod = prevent_mod
        self.tc_path = tc_path

    # Do not override this method
    def get_test_vars(self):
        '''
        Get all relavent variables used in a test run. Purpose is to return variables to students for debugging.
        '''
        return self.input_vars, self.original_input_vars, self.returned_output_vars, self.true_output_vars

    # Do not override this method
    def run_test(self, func):
        '''
        Run the test. Call other methods to test a student solution.
        '''
        self.build_vars()        # Build inputs and true outputs
        if self.prevent_mod:
            self.copy_vars()         # Create copy of inputs
        else:
            self.original_input_vars = self.input_vars
        self.run_func(func)      # Run the function being tested and set the returned outputs
        if self.prevent_mod:     #  - can disable by setting `prevent_mod`to `False` in constructor
            self.check_modified()    # Check to verify inputs were not modified
        self.check_type()        # Check to verify correct output types
        self.check_matches()     # Check to verify correct output

    
    # Do not override this method
    def copy_vars(self):
        '''
        Copies `input_vars` to `original_input_vars`. The copy is needed to verify that the student solution does not modify any inputs.
        '''
        from copy import deepcopy
        self.original_input_vars = {k: deepcopy(v) for k, v in self.input_vars.items()}   
    
    # Do not override this method
    def check_modified(self):
        '''
        Checks `input_vars` against `original_input_vars` to verify that the student's solution did not modify any inputs.
        '''
        for var_name in self.input_vars:
            import pandas as pd
            import numpy as np
            if isinstance(self.original_input_vars[var_name], pd.DataFrame):
                from pandas.testing import assert_frame_equal
                assert_frame_equal(self.original_input_vars[var_name], self.input_vars[var_name])
            elif isinstance(self.original_input_vars[var_name], np.ndarray):
                assert (self.input_vars[var_name] == self.original_input_vars[var_name]).all()
            else:
                assert self.input_vars[var_name] == self.original_input_vars[var_name], f'Your solution modified the input variable `{var_name}`. You can use the testing variables for debugging.'

    ###
    ### These methods will vary based on the requirements of the exercise. They will need to be implemented in a subclass.
    ###
    
    def build_vars(self):
        '''
        Sets `input_vars` and `true_output_vars`.

        This method needs to be defined in a subclass. The exact details and requirements will vary between exercises.
        '''
        raise NotImplementedError

    def run_func(self, func):
        '''
        Sets `returned_output_vars` by running the student's solution.

        This method needs to be defined in a subclass. The exact details and requirements will vary between exercises.
        '''
        raise NotImplementedError
    
    def check_type(self):
        '''
        Checks that `returned_output_vars` are all the correct type using isinstance.

        This method needs to be defined in a subclass. The exact details and requirements will vary between exercises.
        The checks should be implemented using assertions which identify which variable does not match and the type expected.
        '''
        raise NotImplementedError

    def check_matches(self):
        '''
        Checks that `returned_output_vars` are consistent with `true_output_vars`.

        This method needs to be defined in a subclass. The exact details and requirements will vary between exercises.
        The checks should be implemented using assertions which identify which variable does not match.
        '''
        raise NotImplementedError


