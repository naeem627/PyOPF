import argparse
import json

__all__ = ['ReadJSON']

class ReadJSON(argparse.Action):

    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        if nargs is not None:
            raise ValueError("nargs not allowed")
        super(ReadJSON, self).__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        values_json = json.load(open(values))
        setattr(namespace, self.dest, values_json)
