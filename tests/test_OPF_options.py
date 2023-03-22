from pyopf.run import run_opf


class TestOPFOptions:
    def test_scenario_options(self):
        _case = "NYISO"
        _objective = "min cost"
        _options = {
            "voltage bounds": (0.95, 1.05),
            "costs": "cases/NYISO/scenarios/offpeak/NYISO_offpeak2019_v23_shunts_as_gens_costs.json"
        }
        _scenario = {
            "dir": "scenarios/offpeak",
            "name": "offpeak2019_v23_shunts_as_gens",
            "timestamp": "None"
        }

        _nyiso_results = run_opf(case=_case, dir_cases="cases", objective=_objective, scenario=_scenario,
                                 options=_options)

        # get the solved status
        _solved = _nyiso_results["Solved"]["Status"]
        _solved_tc = _nyiso_results["Solved"]["Terminating Condition"]

        assert _solved and _solved_tc == "Feasible and Optimal"
