from pyopf.run import run


class TestOPFBasic:

    def test_ieee14(self):
        _case = "IEEE-14"
        _objective = "min cost"

        _ieee_14_results = run(case=_case, dir_cases="cases", objective=_objective)

        # get the solved status
        _solved = _ieee_14_results["Solved"]["Status"]
        _solved_tc = _ieee_14_results["Solved"]["Terminating Condition"]

        assert _solved and _solved_tc == "Feasible and Optimal"

    def test_ieee118(self):
        _case = "IEEE-118"
        _objective = "min cost"

        _ieee_118_results = run(case=_case, dir_cases="cases", objective=_objective)

        # get the solved status
        _solved = _ieee_118_results["Solved"]["Status"]
        _solved_tc = _ieee_118_results["Solved"]["Terminating Condition"]

        assert _solved and _solved_tc == "Feasible and Optimal"

    def test_Texas7k(self):
        _case = "Texas7k"
        _objective = "min cost"

        _texas7k_results = run(case=_case, dir_cases="cases", objective=_objective)

        # get the solved status
        _solved = _texas7k_results["Solved"]["Status"]
        _solved_tc = _texas7k_results["Solved"]["Terminating Condition"]

        assert _solved and _solved_tc == "Feasible and Optimal"
