import argparse
from typing import Optional

from pyopf.preprocess.argparse_actions import ReadJSON
from pyopf.run import run


def main(case: str,
         objective: Optional[str] = "min cost",
         scenario: Optional[dict] = None,
         options: Optional[dict] = None):
    """

    Args:
        case: the name of the network to optimize
        objective: the optimization objective
        scenario: the specific network scenario/case to optimize if any exists
        options: the optimization options

    Returns:
        A summary of the optimal power flow results as a dictionary
    """
    # # Run OPF # #
    _opf_results = run(case, "cases", objective, scenario, options)
    return _opf_results


if __name__ == "__main__":
    cli_parser = argparse.ArgumentParser(description="Parse command line inputs")
    cli_parser.add_argument('--case',
                            help='the name of the network to run',
                            type=str)

    cli_parser.add_argument('--obj',
                            help='the chosen optimization obj to min or max',
                            default='min cost',
                            type=str)

    cli_parser.add_argument('--scenario',
                            help='the scenario to run',
                            action=ReadJSON,
                            default=None)

    cli_parser.add_argument('--options',
                            help='options used to modify the optimization; for example changing voltage bounds',
                            action=ReadJSON,
                            default=None)

    args = cli_parser.parse_args()

    main(args.case, args.obj, args.scenario, args.options)
