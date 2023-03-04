"""Data structures and read/write methods for input and output data.json file format

Author: Jesse Holzer, jesse.holzer@pnnl.gov
Author: Arun Veeramany, arun.veeramany@pnnl.gov
Author: Randy K Tran, randy.tran@pnnl.gov

Date: 2020-07-10

TODO update the default values supplied by the scrubber in view of 8-28 meeting
check cblocks in gen, load, pcblocks, qcblocks, scblocks
discard unneeded fields
In scrubber, set defaults for certain missing values
"""
# data_json.py
# module for input and output data
# including data structures
# and read and write functions

import inspect
import json
import sys

from jsonschema import Draft7Validator

# these are for fixing existing cost functions that do not cover the required range
cost_function_min_range = 1.0e12
cost_function_default_cost = 1.0e6

# set defaults here
# used in do_force_defaults
do_check_delta_default = True
do_check_deltactg_default = True
do_check_deltar_default = False
do_check_deltarctg_default = False
delta_tol = 1.0/3600.0
delta_default = 5.0/60.0
deltactg_default = 5.0/60.0
deltar_default = 5.0/60.0
deltarctg_default = 5.0/60.0
load_prumaxctg_default_equals_prumax = True
load_prdmaxctg_default_equals_prdmax = True
generator_prumaxctg_default_equals_prumax = True
generator_prdmaxctg_default_equals_prdmax = True
pcblocks_default = [{"pmax": 1000000000001.0, "c": 10000.0}] # extra step for numerical tolerance?
qcblocks_default = [{"qmax": 1000000000001.0, "c": 10000.0}] # extra step for numerical tolerance?
scblocks_default = [{"tmax": 0.05, "c": 50.0}, {"tmax": 0.05, "c": 200.0}, {"tmax": 0.2, "c": 1000.0}, {"tmax": 1000000000001.0, "c": 1000000.0}] # extra step for numerical tolerance? - added a step to cover essentially any possible quantity (i.e. up to 1e12+1) at cost 1e6USD/MWh

# easy solution for now - TODO fix this later
# do not change anything in systemparameters
# if load_prumaxctg is missing, set it equal to load_prumax
# if load_prdmaxctg is missing, set it equal to load_prdmax
# if generator_prumaxctg is missing, set it equal to generator_prumax
# if generator_prdmaxctg is missing, set it equal to generator_prdmax
# overwrite pcblocks, qcblocks, scblocks with defaults

def is_in(key, item, context):
    if key in item:
        return True
    else:
        message = "Error: Unable to find {context}{key}.".format(context=context, key=key)
        print(message) # JH. we need this message so that daata teams know they need to add this. maybe should exit too
        return False


def is_not_in(key, json_object):
    if key not in json_object:
        message = "Error: Unable to find {key}.".format(key=key)
        print(message)
        print("Exiting...")
        sys.exit()


def parse_keys(keys, json_object, context):
    for key in keys:
        if not is_in(key, json_object, context):
            keys.remove(key)

    return keys


def validate_json(keys, jsonobj):
    for key in keys:
        is_not_in(key, jsonobj)


class Sup:
    sup_jsonobj = None

    # All cached content here
    pcblocks_sorted = False
    qcblocks_sorted = False
    scblocks_sorted = False

    # ALl cached ids here
    generator_ids = set()
    load_ids = set()
    line_ids = set()
    transformer_ids = set()

    transformers = {}
    lines = {}
    generators = {}

    scblocks_tmax = []
    scblocks_c = []

    pcblocks_pmax = []
    pcblocks_c = []

    qcblocks_qmax = []
    qcblocks_c = []

    def __init__(self):
        pass

    scrub_mode = False
    do_force_defaults = False

    def scrub_pcblocks(self, cblock):
        cblock['pmax'] = 1e12
        cblock['c'] = 1000000
       
    def scrub_qcblocks(self, cblock):
        cblock['qmax'] = 1e12
        cblock['c'] = 1000000

    def scrub_scblocks(self, cblock):
        cblock['tmax'] = 1e12
        cblock['c'] = 1000000


    def validateJson(self, schema_path, instance_path):

        with open(schema_path, 'r') as f:
            schema = json.loads(  f.read() )


        with open(instance_path, 'r') as f:
            sup_data = f.read()
            #instance = json.loads(sup_data)
            instance = json.loads(sup_data, object_pairs_hook=dict_alert_on_duplicates)
            v = Draft7Validator(schema)
            errors = sorted(v.iter_errors(instance), key=lambda e: e.path)
            for error in errors:
                for suberror in sorted(error.context, key=lambda e: e.schema_path):
                    print(list(error.schema_path),list(suberror.schema_path), suberror.message)
        return len(errors)==0



    def assert_continue(self, condition, error_string, scrub_info=None):
        try:
            assert condition
            return True
        except:
            print("\t" + error_string)
            if self.scrub_mode and scrub_info is not None:
                print("\t\tscrubbing...")
                scrub_info['handler'](scrub_info)
            return False

    # def scrub_cblock(self, scrub_info):
    #     if 
    #     scrub_info['cblock']
    #     #pqs = 'p', 'q', or 's', or ''
    #     max_prefix = scrub_info['pqs']
    #     #pqs_new = ('t' if pqs == 's' else pqs)
    #     #cblocks_string = pqs + 'cblocks'
    #     max_string = max_prefix + 'max'
    #     #cblocks = self.sup_jsonobj[cblocks_string]
    #     cblocks += [{'c': cost_function_default_cost, max_string: (cost_function_min_range + 1.0)}]
    #     self.sup_jsonobj[cblocks_string] = cblocks

    def scrub_pqscblocks(self, scrub_info):
        #pqs = 'p', 'q', or 's'
        pqs = scrub_info['pqs']
        pqs_new = ('t' if pqs == 's' else pqs)
        cblocks_string = pqs + 'cblocks'
        max_string = pqs_new + 'max'
        cblocks = self.sup_jsonobj[cblocks_string]
        cblocks += [{'c': cost_function_default_cost, max_string: (cost_function_min_range + 1.0)}]
        self.sup_jsonobj[cblocks_string] = cblocks

    def scrub_delta(self, scrub_info):
        self.sup_jsonobj['systemparameters']['delta'] = delta_default

    def scrub_deltactg(self, scrub_info):
        self.sup_jsonobj['systemparameters']['deltactg'] = deltactg_default

    def scrub_deltar(self, scrub_info):
        self.sup_jsonobj['systemparameters']['deltar'] = deltar_default

    def scrub_deltarctg(self, scrub_info):
        self.sup_jsonobj['systemparameters']['deltarctg'] = deltarctg_default

    def scrub_load_tmin_tmax(self, scrub_info):
        scrub_info['load']['tmin'] = 0.0
        scrub_info['load']['tmax'] = 1.0

    def scrub_gen_prumax_nonneg(self, scrub_info):
        scrub_info["generator"]['prumax'] = 0.0

    def scrub_gen_prdmax_nonneg(self, scrub_info):
        scrub_info["generator"]['prdmax'] = 0.0

    def scrub_load_prumax_nonneg(self, scrub_info):
        scrub_info["load"]['prumax'] = 0.0

    def scrub_load_prdmax_nonneg(self, scrub_info):
        scrub_info["load"]['prdmax'] = 0.0

    def scrub_gen_prumaxctg_nonneg(self, scrub_info):
        scrub_info["generator"]['prumaxctg'] = 0.0

    def scrub_gen_prdmaxctg_nonneg(self, scrub_info):
        scrub_info["generator"]['prdmaxctg'] = 0.0

    def scrub_load_prumaxctg_nonneg(self, scrub_info):
        scrub_info["load"]['prumaxctg'] = 0.0

    def scrub_load_prdmaxctg_nonneg(self, scrub_info):
        scrub_info["load"]['prdmaxctg'] = 0.0

    def remove_loads(self, keys):
        json_map = {(r["bus"], r["id"]):r for r in self.sup_jsonobj['loads']}
        json_keys = json_map.keys()
        json_keys = set(json_keys).difference(set(keys))
        json_vals = [json_map[k] for k in json_keys]
        self.sup_jsonobj['loads'] = json_vals
        self.init_loads()
        self.load_ids = []
        self.get_load_ids()

    def remove_generators(self, keys):
        json_map = {(r["bus"], r["id"]):r for r in self.sup_jsonobj['generators']}
        json_keys = json_map.keys()
        json_keys = set(json_keys).difference(set(keys))
        json_vals = [json_map[k] for k in json_keys]
        self.sup_jsonobj['generators'] = json_vals
        self.init_generators()
        self.generator_ids = []
        self.get_generator_ids()

    def remove_lines(self, keys):
        json_map = {(r["origbus"], r["destbus"], r["id"]):r for r in self.sup_jsonobj['lines']}
        json_keys = json_map.keys()
        json_keys = set(json_keys).difference(set(keys))
        json_vals = [json_map[k] for k in json_keys]
        self.sup_jsonobj['lines'] = json_vals
        self.init_lines()
        self.line_ids = []
        self.get_line_ids()

    def remove_transformers(self, keys):
        json_map = {(r["origbus"], r["destbus"], r["id"]):r for r in self.sup_jsonobj['transformers']}
        json_keys = json_map.keys()
        json_keys = set(json_keys).difference(set(keys))
        json_vals = [json_map[k] for k in json_keys]
        self.sup_jsonobj['transformers'] = json_vals
        self.init_transformers()
        self.transformer_ids = []
        self.get_transformer_ids()

    ### Checks Start Here #######################

    def check(self, scrub_mode=False):
        self.scrub_mode = scrub_mode

        print("Validating JSON...")
        keys = ["systemparameters", "loads", "generators", "lines", "transformers", "pcblocks", "qcblocks",
                      "scblocks"]
        #CHALLENGE2 ABOVE MANDATORY

        validate_json(keys, self.sup_jsonobj)

        for key in keys:
            if key == "systemparameters":
                message = "Checking {key}...".format(key=key)
                #print(message)
                self.check_system_parameters(self.get_value(key))

            if key == "loads":
                message = "Checking {key}...".format(key=key)
                #print(message)
                [self.check_load(item) for item in self.get_value(key)]

            if key == "generators":
                message = "Checking {key}...".format(key=key)
                #print(message)
                [self.check_generator(item) for item in self.get_value(key)]

            if key == "lines":
                message = "Checking {key}...".format(key=key)
                #print(message)
                [self.check_line(item) for item in self.get_value(key)]

            if key == "transformers":
                message = "Checking {key}...".format(key=key)
                #print(message)
                [self.check_transformer(item) for item in self.get_value(key)]

            if key == "pcblocks":

                message = "Checking {key}...".format(key=key)
                print(message)
                [self.check_pcblock(item) for item in self.get_value(key)]
                #self.check_pcblocks(self.get_value(key))
                self.check_pqscblocks(self.get_value(key), 'p')
                #self.check_pqscblocks(cblocks, 'p')
                '''
                =======
                 message = "Checking {key}...".format(key=key)
                 #print(message)
                 [self.check_pcblock(item) for item in self.get_value(key)]
                 self.check_pcblocks(self.get_value(key))
            

                 '''

            if key == "qcblocks":
                message = "Checking {key}...".format(key=key)
                #print(message)
                [self.check_qcblock(item) for item in self.get_value(key)]
                #self.check_qcblocks(self.get_value(key))
                self.check_pqscblocks(self.get_value(key), 'q')
            
            if key == "scblocks":
                message = "Checking {key}...".format(key=key)
                #print(message)
                [self.check_scblock(item) for item in self.get_value(key)]
                #self.check_scblocks(self.get_value(key))
                self.check_pqscblocks(self.get_value(key), 's')

    def check_system_parameters(self, system_parameters):
        context = "{}: ".format(inspect.stack()[0][3])
        keys = ["delta", "deltactg", "deltar", "deltarctg"]
        #ALL MANDATORY ABOVE
        keys = parse_keys(keys, system_parameters, context)

        for key in keys:
            if key == "delta":
                condition = system_parameters[key] > 0.0
                scrub_info = {"handler": self.scrub_delta}
                message = context + "{key} must be greater than zero".format(key=key)
                self.assert_continue(condition, message, scrub_info)
                if do_check_delta_default:
                    condition = abs(system_parameters[key] - delta_default) <= delta_tol
                    message = context + "{key} should be equal to {default} except with intentional justification. Current value: {val}, tolerance: {tol}. Scrubbing does not remove this warning.".format(key=key, default=delta_default, val=system_parameters[key], tol=delta_tol)
                    self.assert_continue(condition, message)

            if key == "deltactg":
                condition = system_parameters[key] > 0.0
                scrub_info = {"handler": self.scrub_deltactg}
                message = context + "{key} must be greater than zero".format(key=key)
                self.assert_continue(condition, message, scrub_info)
                if do_check_deltactg_default:
                    condition = abs(system_parameters[key] - deltactg_default) <= delta_tol
                    message = context + "{key} should be equal to {default} except with intentional justification. Current value: {val}, tolerance: {tol}. Scrubbing does not remove this warning.".format(key=key, default=deltactg_default, val=system_parameters[key], tol=delta_tol)
                    self.assert_continue(condition, message)

            if key == "deltar":
                condition = system_parameters[key] > 0.0
                scrub_info = {"handler": self.scrub_deltar}
                message = context + "{key} must be greater than zero".format(key=key)
                self.assert_continue(condition, message, scrub_info)
                if do_check_deltar_default:
                    condition = abs(system_parameters[key] - deltar_default) <= delta_tol
                    message = context + "{key} should be equal to {default} except with intentional justification. Current value: {val}, tolerance: {tol}. Scrubbing does not remove this warning.".format(key=key, default=deltar_default, val=system_parameters[key], tol=delta_tol)
                    self.assert_continue(condition, message)

            if key == "deltarctg":
                condition = system_parameters[key] > 0.0
                scrub_info = {"handler": self.scrub_deltarctg}
                message = context + "{key} must be greater than zero".format(key=key)
                self.assert_continue(condition, message, scrub_info)
                if do_check_deltarctg_default:
                    condition = abs(system_parameters[key] - deltarctg_default) <= delta_tol
                    message = context + "{key} should be equal to {default} except with intentional justification. Current value: {val}, tolerance: {tol}. Scrubbing does not remove this warning.".format(key=key, default=deltarctg_default, val=system_parameters[key], tol=delta_tol)
                    self.assert_continue(condition, message)

    def check_load(self, load):
        context = "{} [bus {}]: ".format(inspect.stack()[0][3], load["bus"])
        keys = ["id", "prumax", "prdmax", "prumaxctg", "prdmaxctg", "tmin", "tmax", "cblocks"]
        keys = parse_keys(keys, load, context)

        for key in keys:
            # id
            if key == "id":
                condition = len(load[key]) in [1, 2]
                message = context + "{key} must be utmost 2 characters".format(key=key)
                self.assert_continue(condition, message)

            # prumax
            if key == "prumax":
                condition = load[key] >= 0.0
                message = context + "{key} must be non-negative".format(key=key)
                scrub_info = {"handler": self.scrub_load_prumax_nonneg, "load": load}
                #print('hello world 2')
                self.assert_continue(condition, message, scrub_info)

            # prdmax
            if key == "prdmax":
                condition = load[key] >= 0.0
                message = context + "{key} must be non-negative".format(key=key)
                scrub_info = {"handler": self.scrub_load_prdmax_nonneg, "load": load}
                self.assert_continue(condition, message, scrub_info)

            # prumaxctg
            if key == "prumaxctg":
                condition = load[key] >= 0.0
                message = context + "{key} must be non-negative".format(key=key)
                scrub_info = {"handler": self.scrub_load_prumaxctg_nonneg, "load": load}
                #print('hello world 2')
                self.assert_continue(condition, message, scrub_info)

            # prdmaxctg
            if key == "prdmaxctg":
                condition = load[key] >= 0.0
                message = context + "{key} must be non-negative".format(key=key)
                scrub_info = {"handler": self.scrub_load_prdmaxctg_nonneg, "load": load}
                self.assert_continue(condition, message, scrub_info)

            # cblocks
            if key == "cblocks":
                for b in load[key]:
                    #print(b)
                    self.check_cblock(b, max_prefix='p', cblock_prefix='')
                #self.check_pqscblocks(load[key], 'p') # todo: need to do this in Data.check(), as we do not have pmin,pmax here

        # tmin && tmax
        if "tmin" in keys and "tmax" in keys:
            condition = load["tmin"] <= load["tmax"]
            message = context + "tmin must be <= tmax"
            scrub_info = {"handler": self.scrub_load_tmin_tmax, "load": load}
            self.assert_continue(condition, message, scrub_info)

    def check_generator(self, generator):
        context = "{} [bus {}]: ".format(inspect.stack()[0][3], generator["bus"])
        keys = ["id", "suqual", "sdqual", "suqualctg", "sdqualctg", "prumax", "prdmax", "prumaxctg", "prdmaxctg", "oncost", "sucost",
                      "sdcost", "cblocks"]
        keys = parse_keys(keys, generator, context)

        for key in keys:
            if key == "id":
                condition = len(generator["id"]) in [1, 2]
                message = context + "id must be utmost 2 characters"
                self.assert_continue(condition, message)

            if key == "prumax":
                condition = generator[key] >= 0.0
                message = "{key} must be non-negative".format(key=key)
                scrub_info = {"handler": self.scrub_gen_prumax_nonneg, "generator": generator}
                self.assert_continue(condition, message, scrub_info)

            if key == "prdmax":
                condition = generator[key] >= 0.0
                message = "{key} must be non-negative".format(key=key)
                scrub_info = {"handler": self.scrub_gen_prdmax_nonneg, "generator": generator}
                self.assert_continue(condition, message, scrub_info)

            if key == "prumaxctg":
                condition = generator[key] >= 0.0
                message = "{key} must be non-negative".format(key=key)
                scrub_info = {"handler": self.scrub_gen_prumaxctg_nonneg, "generator": generator}
                self.assert_continue(condition, message, scrub_info)

            if key == "prdmaxctg":
                condition = generator[key] >= 0.0
                message = "{key} must be non-negative".format(key=key)
                scrub_info = {"handler": self.scrub_gen_prdmaxctg_nonneg, "generator": generator}
                self.assert_continue(condition, message, scrub_info)

            if key == "suqual":
                condition = generator[key] == 0 or generator[key] == 1
                message = context + "{key} must be either 0 or 1".format(key=key)
                self.assert_continue(condition, message)

            if key == "sdqual":
                condition = generator[key] == 0 or generator[key] == 1
                message = context + "{key} must be either 0 or 1".format(key=key)
                self.assert_continue(condition, message)

            if key == "suqualctg":
                condition = generator[key] == 0 or generator[key] == 1
                message = context + "{key} must be either 0 or 1".format(key=key)
                self.assert_continue(condition, message)

            if key == "sdqualctg":
                condition = generator[key] == 0 or generator[key] == 1
                message = context + "{key} must be either 0 or 1".format(key=key)
                self.assert_continue(condition, message)

            # cblocks
            if key == "cblocks":
                for b in generator[key]:
                    #print(b)
                    self.check_cblock(b, max_prefix='p', cblock_prefix='')
                #self.check_pqscblocks(generator[key], 'p') # todo: need to do this in Data.check(), as we do not have pmin,pmax here

    def check_line(self, line):
        context = "{} [origbus {}, destbus {}]: ".format(inspect.stack()[0][3], line["origbus"], line["destbus"])
        keys = ["origbus", "destbus", "id", "swqual", "csw"]
        keys = parse_keys(keys, line, context)

        for key in keys:
            if key == "swqual":
                condition = line[key] == 0 or line[key] == 1
                message = context + "{key} must be either 0 or 1".format(key=key)
                self.assert_continue(condition, message)

    def check_transformer(self, transformer):
        context = "{} [origbus {}, destbus {}]: ".format(inspect.stack()[0][3], transformer["origbus"],
                                                         transformer["destbus"])
        keys = ["origbus", "destbus", "id", "swqual", "csw"]
        keys = parse_keys(keys, transformer, context)

        for key in keys:
            if key == "swqual":
                condition = transformer[key] == 0 or transformer[key] == 1
                message = context + "{key} must be either 0 or 1".format(key=key)
                self.assert_continue(condition, message)

    def check_cblock(self, cblock, max_prefix=None, cblock_prefix=None):
        context = ("" if cblock_prefix is None else cblock_prefix) + "cblock"
        max_str = ("" if max_prefix is None else max_prefix) + "max"
        keys = [max_str, "c"]
        keys = parse_keys(keys, cblock, context)

        for key in keys:
            if key == max_str:
                condition = (cblock[key] >= 0.0)
                message = context + " {key} must be >= 0.0: {cblock}".format(key=key, cblock=str(cblock))
                self.assert_continue(condition, message)
            if key == "c":
                condition = None
                message = ""
        if self.scrub_mode:
            if not(cblock[max_str] >= 0.0):
                print('scrubbing {} {}. setting {} to 0.0'.format(context, str(cblock), max_str))
                cblock[max_str] = 0.0

    def check_pcblock(self, pcblock):
        self.check_cblock(pcblock, max_prefix='p', cblock_prefix='p')
        # context = "pcblock"
        # keys = ["pmax", "c"]
        # keys = parse_keys(keys, pcblock, context)

        # for key in keys:
        #     if key == "pmax":
        #         condition = (pcblock[key] >= 0.0)
        #         message = context + " {key} must be >= 0.0: {cblock}".format(key=key, cblock=str(pcblock))
        #         self.assert_continue(condition, message)
        #     if key == "c":
        #         condition = None
        #         message = ""

    def check_qcblock(self, qcblock):
        self.check_cblock(qcblock, max_prefix='q', cblock_prefix='q')
        # context = "qcblock"
        # keys = ["qmax", "c"]
        # keys = parse_keys(keys, qcblock, context)

        # for key in keys:
        #     if key == "qmax":
        #         condition = (qcblock[key] >= 0.0)
        #         message = context + " {key} must be >= 0.0: {cblock}".format(key=key, cblock=str(qcblock))
        #         self.assert_continue(condition, message)
        #     if key == "c":
        #         condition = None
        #         message = ""

    def check_scblock(self, scblock):
        self.check_cblock(scblock, max_prefix='t', cblock_prefix='s')
        # context = "scblock"
        # keys = ["tmax", "c"]
        # keys = parse_keys(keys, scblock, context)

        # for key in keys:
        #     if key == "tmax":
        #         condition = (scblock[key] >= 0.0)
        #         message = context + " {key} must be >= 0.0: {cblock}".format(key=key, cblock=str(scblock))
        #         self.assert_continue(condition, message)
        #     if key == "c":
        #         condition = None
        #         message = ""

    #<<<<<<< valid_aug
    def check_pqscblocks(self, cblocks, pqs):
        #pqs = 'p' or 'q' or 's'
        pqs_new = ('t' if pqs == 's' else pqs)
        max_string = pqs_new + 'max'
        total_max = sum([0.0] + [b[max_string] for b in cblocks])
        condition = (len(cblocks) > 0) and (total_max >= 1.0e12)
        message = "{}cblocks requires sum([b[pmax] for b in {}cblocks]) >= 1e12: {cblocks}".format(pqs, pqs, cblocks=str(cblocks))
        scrub_info = {"handler": self.scrub_pqscblocks, "pqs": pqs}
        self.assert_continue(condition, message, scrub_info)

    '''        
        =======
    def check_pcblocks(self, cblocks):

        total_max = sum([0.0] + [b['pmax'] for b in cblocks])
        condition = (total_max >= 1.0e12)
        message = "pcblocks requires sum([b[pmax] for b in pcblocks]) >= 1e12: {cblocks}".format(cblocks=str(cblocks))
        self.assert_continue(condition, message)
        if condition == False:
            print('\t\tscrubber setting: pmax:1e+12, c:1000000')
            self.scrub_pcblocks( cblocks[-1] )

    def check_qcblocks(self, cblocks):

        total_max = sum([0.0] + [b['qmax'] for b in cblocks])
        condition = (total_max >= 1.0e12)
        message = "qcblocks requires sum([b[qmax] for b in qcblocks]) >= 1e12: {cblocks}".format(cblocks=str(cblocks))
        self.assert_continue(condition, message)
        if condition == False:
            print('\t\tscrubber setting: pmax:1e+12, c:1000000')
            self.scrub_qcblocks( cblocks[-1] )

    def check_scblocks(self, cblocks):

        total_max = sum([0.0] + [b['tmax'] for b in cblocks])
        condition = (total_max >= 1.0e12)
        message = "scblocks requires sum([b[tmax] for b in scblocks]) >= 1e12: {cblocks}".format(cblocks=str(cblocks))
        self.assert_continue(condition, message)
        if condition == False:
            print('\t\tscrubber setting: pmax:1e+12, c:1000000')
            self.scrub_scblocks( cblocks[-1] )

    #>>>>>>> develop
    '''
        
    ### Getters Start Here #######################
    def get_value(self, key):
        try:
            value = self.sup_jsonobj[key]
            return value
        except KeyError:
            print("Error: Unable to get key - [{key}]".format(key=key))

    def get_system_parameters(self):
        return self.sup_jsonobj['system_parameters']

    def get_loads(self):
        return self.sup_jsonobj['loads']

    def get_generators(self):
        return self.sup_jsonobj['generators']

    def force_defaults(self):
        '''
        TODO refine this
        load_prumaxctg_default_equals_prumax = True
        load_prdmaxctg_default_equals_prdmax = True
        generator_prumaxctg_default_equals_prumax = True
        generator_prdmaxctg_default_equals_prdmax = True
        pcblocks_default = [{"pmax": 1000000000001.0, "c": 10000.0}] # extra step for numerical tolerance?
        qcblocks_default = [{"qmax": 1000000000001.0, "c": 10000.0}] # extra step for numerical tolerance?
        scblocks_default = [{"tmax": 0.05, "c": 50.0}, {"tmax": 0.05, "c": 200.0}, {"tmax": 0.2, "c": 1000.0}, {"tmax": 1000000000001.0, "c": 1000000.0}] # e       
        '''
        self.revert_pr_maxctg_to_pr_max()
        self.revert_pcblocks_to_default()
        self.revert_qcblocks_to_default()
        self.revert_scblocks_to_default()

    def init(self):
        self.init_generators()
        self.init_loads()
        self.init_lines()
        self.init_transformers()

        pcblocks = self.get_pcblocks()
        self.pcblocks_pmax = [ block['pmax']    for block in pcblocks   ]
        self.pcblocks_c = [ block['c']    for block in pcblocks   ]

        qcblocks = self.get_qcblocks()
        self.qcblocks_qmax = [ block['qmax']    for block in qcblocks   ]
        self.qcblocks_c = [ block['c']    for block in qcblocks   ]

        scblocks = self.get_scblocks()
        self.scblocks_tmax = [ block['tmax']    for block in scblocks   ]
        self.scblocks_c = [ block['c']    for block in scblocks   ]

    def init_generators(self):
        self.generators = { (g['bus'], g['id']): g  for g in self.sup_jsonobj["generators"] }
        self.generator_count = len(self.generators)
        self.check_repeated_keys_in_sup_jsonobj_section("generators", ['bus', 'id'])
        self.gen_cblock_count = { (g['bus'], g['id']): len(g["cblocks"]) for g in self.generators.values()  }

    def init_lines(self):
        #CHALLENGE2 REVIEW: id == ckt ?
        self.lines = { (g['origbus'], g['destbus'], g['id']): g  for g in self.sup_jsonobj["lines"] }
        self.line_count = len(self.lines)
        self.check_repeated_keys_in_sup_jsonobj_section("lines", ['origbus', 'destbus', 'id'])

    def init_transformers(self):
            #CHALLENGE2 REVIEW: id == ckt ?
        self.transformers = { (g['origbus'], g['destbus'], g['id']): g  for g in self.sup_jsonobj["transformers"] }
        self.xfmr_count = len(self.transformers)
        self.check_repeated_keys_in_sup_jsonobj_section("transformers", ['origbus', 'destbus', 'id'])

    def  convert_generator_cblock_units(self,base_mva):
        if self.scrub_mode == True:
            raise "Refusing to convert units in scrub mode"
            return
        for generator in self.generators.values():
            for cblock in generator['cblocks']:
                cblock['pmax'] /= base_mva
                cblock['c'] *= base_mva

    def  convert_load_cblock_units(self,base_mva):
        if self.scrub_mode == True:
            raise "Refusing to convert units in scrub mode"
            return
        for load in self.loads.values():
            for cblock in load['cblocks']:
                cblock['pmax'] /= base_mva
                cblock['c'] *= base_mva
                
    def convert_pcblock_units(self,base_mva):
        if self.scrub_mode == True:
            raise "Refusing to convert units in scrub mode"
            return
        for pcblock in self.sup_jsonobj['pcblocks']:
            pcblock['pmax'] /= base_mva
            pcblock['c'] *= base_mva

    def convert_qcblock_units(self,base_mva):
        if self.scrub_mode == True:
            raise "Refusing to convert units in scrub mode"
            return
        for qcblock in self.sup_jsonobj['qcblocks']:
            qcblock['qmax'] /= base_mva
            qcblock['c'] *= base_mva

    def convert_scblock_units(self,base_mva):
        if self.scrub_mode == True:
            raise "Refusing to convert units in scrub mode"
            return
        for scblock in self.sup_jsonobj['scblocks']:
            #scblock['tmax'] /= base_mva # no normalization needed
            scblock['c'] *= base_mva

    def check_repeated_keys_in_sup_jsonobj_section(self, section, key_fields):
        #num_entries = len(self.sup_jsonobj[section])
        keys_with_repeats = sorted([tuple([r[kf] for kf in key_fields]) for r in self.sup_jsonobj[section]])
        unique_keys = sorted(list(set(keys_with_repeats)))
        num_entries = len(keys_with_repeats)
        num_unique_keys = len(unique_keys)
        if num_unique_keys < num_entries:
            repeated_keys = []
            for i in range(len(keys_with_repeats) - 1):
                if keys_with_repeats[i] == keys_with_repeats[i + 1]:
                    repeated_keys.append(keys_with_repeats[i])
            repeated_keys = sorted(list(set(repeated_keys)))
            print(
                {'data_type': 'Sup', 'error_message': 'repeated key',
                 'diagnostics': {'section': section, 'entries': num_entries, 'unique keys': num_unique_keys, 'repeated keys': repeated_keys}})

    def init_loads(self):
        self.loads = { (v['bus'], v['id']): v  for v in self.sup_jsonobj["loads"] }
        self.load_count = len(self.loads)
        self.check_repeated_keys_in_sup_jsonobj_section("loads", ['bus', 'id'])
        self.load_cblock_count = { load['bus']: len(load["cblocks"]) for load in self.loads.values()  }
  
    def get_lines(self):
        return self.sup_jsonobj['lines']

    def get_transformers(self):
        return self.sup_jsonobj['transformers']

    def revert_pr_maxctg_to_pr_max(self):
        for v in self.sup_jsonobj["loads"]:
            if not "prumax" in v.keys():
                print('{} {} missing {}. setting to {}'.format('load', v, 'prumax', 0.0))
                v["prumax"] = 0.0
            if not "prdmax" in v.keys():
                print('{} {} missing {}. setting to {}'.format('load', v, 'prdmax', 'prumax'))
                v["prdmax"] = v["prumax"]
            if not "prumaxctg" in v.keys():
                print('{} {} missing {}. setting to {}'.format('load', v, 'prumaxctg', 'prumax'))
                v["prumaxctg"] = v["prumax"]
            if not "prdmaxctg" in v.keys():
                print('{} {} missing {}. setting to {}'.format('load', v, 'prdmaxctg', 'prdmax'))
                v["prdmaxctg"] = v["prdmax"]
        for v in self.sup_jsonobj["generators"]:
            if not "prumax" in v.keys():
                print('{} {} missing {}. setting to {}'.format('gen', v, 'prumax', 0.0))
                v["prumax"] = 0.0
            if not "prdmax" in v.keys():
                print('{} {} missing {}. setting to {}'.format('gen', v, 'prdmax', 'prumax'))
                v["prdmax"] = v["prumax"]
            if not "prumaxctg" in v.keys():
                print('{} {} missing {}. setting to {}'.format('gen', v, 'prumaxctg', 'prumax'))
                v["prumaxctg"] = v["prumax"]
            if not "prdmaxctg" in v.keys():
                print('{} {} missing {}. setting to {}'.format('gen', v, 'prdmaxctg', 'prdmax'))
                v["prdmaxctg"] = v["prdmax"]

    def revert_pcblocks_to_default(self):
        print('setting {} to {}'.format('pcblocks', pcblocks_default))
        self.sup_jsonobj['pcblocks'] = pcblocks_default

    def revert_qcblocks_to_default(self):
        print('setting {} to {}'.format('qcblocks', qcblocks_default))
        self.sup_jsonobj['qcblocks'] = qcblocks_default

    def revert_scblocks_to_default(self):
        print('setting {} to {}'.format('scblocks', scblocks_default))
        self.sup_jsonobj['scblocks'] = scblocks_default

    def get_pcblocks(self):
        if not self.pcblocks_sorted:
            self.sup_jsonobj['pcblocks'] = sorted(self.sup_jsonobj['pcblocks'], key=lambda o: o['c'])
            self.pcblocks_sorted = True
        return self.sup_jsonobj['pcblocks']

    def get_qcblocks(self):
        if not self.qcblocks_sorted:
            self.sup_jsonobj['qcblocks'] = sorted(self.sup_jsonobj['qcblocks'], key=lambda o: o['c'])
            self.qcblocks_sorted = True
        return self.sup_jsonobj['qcblocks']

    def get_scblocks(self):
        if not self.scblocks_sorted:
            self.sup_jsonobj['scblocks'] = sorted(self.sup_jsonobj['scblocks'], key=lambda o: o['c'])
            self.scblocks_sorted = True
        return self.sup_jsonobj['scblocks']
    


    # Get cached ids here

    def get_generator_ids(self):
        if len(self.generator_ids) == 0:
            self.generator_ids = set([ (g['bus'],g['id'].strip())  for g in self.get_generators()])
        return self.generator_ids

    def get_load_ids(self):
        if len(self.load_ids) == 0:
            self.load_ids = set([ (g['bus'],g['id'].strip())  for g in self.get_loads()])
        return self.load_ids

    def get_line_ids(self):
        if len(self.line_ids) == 0:
            self.line_ids = set([ (g['origbus'],g['destbus'],g['id'].strip())  for g in self.get_lines()])
        return self.line_ids

    def get_transformer_ids(self):
        if len(self.transformer_ids) == 0:
            self.transformer_ids = set([ (g['origbus'],g['destbus'],g['id'].strip())  for g in self.get_transformers()])
        return self.transformer_ids

    # Read/Write from/to JSON here

    def read(self, file_name):
        with open(file_name, "r") as case_json_file:
            #self.sup_jsonobj = json.load(case_json_file)
            self.sup_jsonobj = json.load(case_json_file, object_pairs_hook=dict_alert_on_duplicates)
            #print('json file: {}, size: {}'.format(case_json_file, len(self.sup_jsonobj) if self.sup_jsonobj is not None else 0))
            if self.do_force_defaults:
                self.force_defaults()
            self.init()

    def write(self, file_name):
        with open(file_name, "w") as case_json_file:
            json.dump(self.sup_jsonobj,  case_json_file, indent=2)
            case_json_file.write("\n")


#REAL AND REACTIVE IMBALANCES ARE CALCULATED AT THE END
#THESE ARE TRANSLATED TO COST BY APPLYING COST FUNCTIONS
#eval_piecewise_linear_penalty() FOR IMBALANCES AND GEN COST

def dict_alert_on_duplicates(pairs):
    """Alert on duplicate keys."""
    d = {}
    for k, v in pairs:
        if k in d:
            print({'data_type': 'Sup', 'error_message': 'repeated key', 'diagnostics': k})
        d[k] = v
    return d
