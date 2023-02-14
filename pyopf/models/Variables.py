import numbers
from typing import Optional, overload

from sympy import symbols


class Var:

    @overload
    def __init__(self,
                 name=None,
                 active=None,
                 value=None,
                 domain=None,
                 lb=None,
                 ub=None,
                 units=None,
                 doc=None,
                 fixed=None):
        ...

    def __init__(self, **kwargs):
        self._name = kwargs.pop("name", None)
        self._active = kwargs.pop("active", True)
        self._value = kwargs.pop("value", None)
        self._domain = kwargs.pop("domain", "Reals")
        self._lb = kwargs.pop("lb", None)
        self._ub = kwargs.pop("ub", None)
        self._units = kwargs.pop("units", None)
        self._doc = kwargs.pop("doc", None)
        self._symbol = symbols(self._name)
        self._fixed = kwargs.pop("fixed", False)
        self._normalized_value = None

        self.validate_domain(self._value)
        self.validate_domain(self._lb)
        self.validate_domain(self._ub)

    def __repr__(self):
        return f"Var(name='{self._name}',active={self._active},value={self._value})"

    def __str__(self):
        return f"Variable {self._name} has value {self._value} {self._units} and active status is {self._active}."

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, val):
        self._name = val

    @property
    def active(self):
        return self._active

    @active.setter
    def active(self, val):
        self._active = val

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, val):
        self.validate_domain(val)
        self._value = val

    @property
    def fixed(self):
        return self._fixed

    @fixed.setter
    def fixed(self, val):
        self._fixed = val

    @property
    def symbol(self):
        return self._symbol

    @symbol.setter
    def symbol(self, val):
        self._symbol = symbols(val)

    @property
    def domain(self):
        return self._domain

    @domain.setter
    def domain(self, val):
        if not isinstance(val, str):
            raise TypeError("Incorrect type of domain provided.")

        self._domain = val

    @property
    def lb(self):
        return self._lb

    @lb.setter
    def lb(self, val):
        self.validate_domain(val)
        self._lb = val

    @property
    def ub(self):
        return self._ub

    @ub.setter
    def ub(self, val):
        self.validate_domain(val)
        self._ub = val

    @property
    def bounds(self):
        return self._lb, self._ub

    @property
    def units(self):
        return self._units

    @units.setter
    def units(self, val):
        self._units = val

    @property
    def doc(self):
        return self._doc

    @doc.setter
    def doc(self, val):
        self._doc = val

    @property
    def normalized_value(self):
        return self._normalized_value

    def normalize(self,
                  x_min: Optional[float] = None,
                  x_max: Optional[float] = None,
                  eps: float = 1e-5):
        if x_min is None:
            x_min = self._lb

        if x_max is None:
            x_max = self._ub

        x_norm = (self._value - x_min) / (x_max - x_min)

        if x_norm == 0:
            x_norm = eps

        if x_norm == 1:
            x_norm = 1 - eps

        self._normalized_value = x_norm

    def unnormalize(self,
                    val: float,
                    x_min: Optional[float] = None,
                    x_max: Optional[float] = None):
        if x_min is None:
            x_min = self._lb

        if x_max is None:
            x_max = self._ub
        return val * (x_max - x_min) + x_min

    def validate_domain(self, val: Optional[float] = None):
        def validate_real():
            _valid_domain = False
            if isinstance(val, numbers.Real):
                _valid_domain = True
            return _valid_domain

        def validate_nonnegative():
            _valid_domain = False
            if isinstance(val, numbers.Real) and val >= 0:
                _valid_domain = True
            return _valid_domain

        def validate_nonpositive():
            _valid_domain = False
            if isinstance(val, numbers.Real) and val <= 0:
                _valid_domain = True
            return _valid_domain

        def validate_positive():
            _valid_domain = False
            if isinstance(val, numbers.Real) and val > 0:
                _valid_domain = True
            return _valid_domain

        def validate_negative():
            _valid_domain = False
            if isinstance(val, numbers.Real) and val < 0:
                _valid_domain = True
            return _valid_domain

        domain_opts = {
            "Reals": validate_real,
            "NonNegativeReals": validate_nonnegative,
            "NonPositiveReals": validate_nonpositive,
            "PositiveReals": validate_positive,
            "NegativeReals": validate_negative
        }

        domain_status = True
        if val is not None:
            domain_status = domain_opts.get(self._domain)

        if not domain_status:
            raise ValueError("Variable value is not within the %s domain" % self._domain)


def _init_var(var_type: str,
              name: str,
              val: float,
              lb: Optional[float] = None,
              ub: Optional[float] = None,
              domain: Optional[str] = "NonNegativeReals",
              doc: Optional[str] = None) -> Var:
    """
    Turn a physical quantity or data into a variable
    Args:
        var_type: The type of variable (unused but could be useful)
        name: The name of the variable
        val: The present value of the variable
        lb: The lower bound value of the variable
        ub: The upper bound value of the variable
        domain: The domain of the variable which defines the acceptable variable values
        doc: Any specific documentation or notes about the variable

    Returns:

    """

    # check if the variable is fixed, if so fix the variable
    _is_fixed = False
    if val == 0.:
        if lb is None and ub is None:
            _is_fixed = True
        elif lb == 0. and ub == 0.:
            _is_fixed = True

    if "Qg" in name:
        if lb == 0.:
            lb = None
        if ub == 0.:
            ub = None
    var = Var(name=name, active=True, value=val, domain=domain, lb=lb, ub=ub, doc=doc, fixed=_is_fixed)
    return var
