from tester_6040 import *

class Tester_1_2_0(ExerciseTester):
    import pickle
    with open('test_cases.pkl', 'rb') as fin:
        cases = pickle.load(fin)

    def build_vars(self):
        import random
        case = random.choice(self.cases)
        self.input_vars['grades'] = case['grades']
        self.true_output_vars['students'] = case['students']

    def run_func(self, func):
        self.returned_output_vars['students'] = func(self.input_vars['grades'])

    def check_type(self):
        assert type(self.returned_output_vars['students']) is list, 'Output `students` is not the correct type. It should have type `list`.'

    def check_matches(self):
        assert self.returned_output_vars['students'] == self.true_output_vars['students'], 'Your output for `students` does not match the expected output.'

class Tester_1_2_1(ExerciseTester):
    import pickle
    with open('test_cases.pkl', 'rb') as fin:
        cases = pickle.load(fin)
    def build_vars(self):
        import random
        case = random.choice(self.cases)
        self.input_vars['grades'] = case['grades']
        self.true_output_vars['assignments'] = case['assignments']

    def run_func(self, func):
        self.returned_output_vars['assignments'] = func(self.input_vars['grades'])

    def check_type(self):
        assert type(self.returned_output_vars['assignments']) is list, 'Output `assignments` is not the correct type. It should have type `list`.'

    def check_matches(self):
        assert self.returned_output_vars['assignments'] == self.true_output_vars['assignments'], 'Your output for `assignments` does not match the expected output.'

class Tester_1_2_2(ExerciseTester):
    import pickle
    with open('test_cases.pkl', 'rb') as fin:
        cases = pickle.load(fin)
    def build_vars(self):
        import random
        case = random.choice(self.cases)
        self.input_vars['grades'] = case['grades']
        self.true_output_vars['grade_lists'] = case['grade_lists']

    def run_func(self, func):
        self.returned_output_vars['grade_lists'] = func(self.input_vars['grades'])

    def check_type(self):
        assert type(self.returned_output_vars['grade_lists']) is dict, 'Output `grade_lists` is not the correct type. It should have type `dict`.'

    def check_matches(self):
        assert check_dicts_match(self.returned_output_vars['grade_lists'], self.true_output_vars['grade_lists']), 'Your output for `grade_lists` does not match the expected output.'

class Tester_1_2_3(ExerciseTester):
    import pickle
    with open('test_cases.pkl', 'rb') as fin:
        cases = pickle.load(fin)
    def build_vars(self):
        import random
        case = random.choice(self.cases)
        self.input_vars['grades'] = case['grades']
        self.true_output_vars['grade_dicts'] = case['grade_dicts']

    def run_func(self, func):
        self.returned_output_vars['grade_dicts'] = func(self.input_vars['grades'])

    def check_type(self):
        assert type(self.returned_output_vars['grade_dicts']) is dict, 'Output `grade_dicts` is not the correct type. It should have type `dict`.'

    def check_matches(self):
        assert check_dicts_match(self.returned_output_vars['grade_dicts'], self.true_output_vars['grade_dicts']), 'Your output for `grade_dicts` does not match the expected output.'

class Tester_1_2_4(ExerciseTester):
    import pickle
    with open('test_cases.pkl', 'rb') as fin:
        cases = pickle.load(fin)
    def build_vars(self):
        import random
        case = random.choice(self.cases)
        self.input_vars['grades'] = case['grades']
        self.true_output_vars['avg_by_student'] = case['avg_by_student']

    def run_func(self, func):
        self.returned_output_vars['avg_by_student'] = func(self.input_vars['grades'])

    def check_type(self):
        assert type(self.returned_output_vars['avg_by_student']) is dict, 'Output `avg_by_student` is not the correct type. It should have type `dict`.'

    def check_matches(self):
        assert check_dicts_match(self.returned_output_vars['avg_by_student'], self.true_output_vars['avg_by_student'], 4e-14), 'Your output for `avg_by_student` does not match the expected output.'

class Tester_1_2_5(ExerciseTester):
    import pickle
    with open('test_cases.pkl', 'rb') as fin:
        cases = pickle.load(fin)
    def build_vars(self):
        import random
        case = random.choice(self.cases)
        self.input_vars['grades'] = case['grades']
        self.true_output_vars['grade_by_asn'] = case['grade_by_asn']

    def run_func(self, func):
        self.returned_output_vars['grade_by_asn'] = func(self.input_vars['grades'])

    def check_type(self):
        assert type(self.returned_output_vars['grade_by_asn']) is dict, 'Output `grade_by_asn` is not the correct type. It should have type `dict`.'

    def check_matches(self):
        assert check_dicts_match(self.returned_output_vars['grade_by_asn'], self.true_output_vars['grade_by_asn']), 'Your output for `grade_by_asn` does not match the expected output.'

class Tester_1_2_6(ExerciseTester):
    import pickle
    with open('test_cases.pkl', 'rb') as fin:
        cases = pickle.load(fin)
    def build_vars(self):
        import random
        case = random.choice(self.cases)
        self.input_vars['grades'] = case['grades']
        self.true_output_vars['avg_by_asn'] = case['avg_by_asn']

    def run_func(self, func):
        self.returned_output_vars['avg_by_asn'] = func(self.input_vars['grades'])

    def check_type(self):
        assert type(self.returned_output_vars['avg_by_asn']) is dict, 'Output `avg_by_asn` is not the correct type. It should have type `dict`.'

    def check_matches(self):
        assert check_dicts_match(self.returned_output_vars['avg_by_asn'], self.true_output_vars['avg_by_asn'], 4e-14), 'Your output for `avg_by_asn` does not match the expected output.'

class Tester_1_2_7(ExerciseTester):
    import pickle
    with open('test_cases.pkl', 'rb') as fin:
        cases = pickle.load(fin)
    def build_vars(self):
        import random
        case = random.choice(self.cases)
        while(len(case['avg_by_student']) != len(set(case['avg_by_student'].values()))):
            case = random.choice(self.cases)        
        self.input_vars['grades'] = case['grades']
        self.true_output_vars['ranked_students'] = case['ranked_students']

    def run_func(self, func):
        self.returned_output_vars['ranked_students'] = func(self.input_vars['grades'])

    def check_type(self):
        assert type(self.returned_output_vars['ranked_students']) is list, 'Output `ranked_students` is not the correct type. It should have type `list`.'

    def check_matches(self):
        assert self.returned_output_vars['ranked_students'] == self.true_output_vars['ranked_students'], 'Your output for `ranked_students` does not match the expected output.'