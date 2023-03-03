from math import cos, radians, sin, sqrt
from typing import Optional, Union, overload

from pyomo.environ import Var as PyomoVar

from pyopf.models.Variables import _init_var


class Bus:
    @overload
    def __init__(self,
                 bus_data=None,
                 bus: Optional[int] = None,
                 name: Optional[str] = None,
                 bustype: Optional[int] = None,
                 basekv: Optional[float] = None,
                 v_mag: Optional[float] = None,
                 v_ang: Optional[float] = None,
                 v_nhi: Optional[float] = None,
                 v_nlo: Optional[float] = None,
                 v_ehi: Optional[float] = None,
                 v_elo: Optional[float] = None,
                 status: Optional[int] = None):
        ...

    def __init__(self, **kwargs):
        _bus_data = kwargs.pop("bus_data", None)

        self._bus = kwargs.pop("bus", _bus_data.i)
        self._name = kwargs.pop("name", _bus_data.name)
        self._type = kwargs.pop("bustype", _bus_data.ide)
        self._base_KV = kwargs.pop("basekv", _bus_data.baskv)
        self._v_mag = kwargs.pop("v_mag", _bus_data.vm)
        self._v_ang = kwargs.pop("v_ang", _bus_data.va)
        self._v_nhi = kwargs.pop("v_nhi", _bus_data.nvhi)
        self._v_nlo = kwargs.pop("v_nlo", _bus_data.nvlo)
        self._v_ehi = kwargs.pop("v_ehi", _bus_data.evhi)
        self._v_elo = kwargs.pop("v_elo", _bus_data.evlo)
        self._status = kwargs.pop("status", 1)

    def __repr__(self):
        msg = f"Bus(bus={self._bus},name={self._name}, type={self._type},status={self._status}, " \
              f"v_mag={self._v_mag}, v_ang={self._v_ang})"
        return msg

    def __str__(self):
        msg = f"Bus is {self._bus} of type {self._type}, status {self._status}, v_mag {self._v_mag:.3f}, " \
              f"v_ang {self._v_ang:.3f}."
        return msg

    @property
    def bus(self):
        return self._bus

    @bus.setter
    def bus(self, val):
        self._bus = val

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, val):
        self._name = val

    @property
    def type(self):
        return self._type

    @type.setter
    def type(self, val):
        self._type = val

    @property
    def base_KV(self):
        return self._base_KV

    @base_KV.setter
    def base_KV(self, val):
        self._base_KV = val

    @property
    def v_mag(self):
        return self._v_mag

    @v_mag.setter
    def v_mag(self, val):
        self._v_mag = val

    @property
    def v_ang(self):
        return self._v_ang

    @v_ang.setter
    def v_ang(self, val):
        self._v_ang = val

    @property
    def v_nhi(self):
        return self._v_nhi

    @v_nhi.setter
    def v_nhi(self, val):
        self._v_nhi = val

    @property
    def v_nlo(self):
        return self._v_nlo

    @v_nlo.setter
    def v_nlo(self, val):
        self._v_nlo = val

    @property
    def v_ehi(self):
        return self._v_ehi

    @v_ehi.setter
    def v_ehi(self, val):
        self._v_ehi = val

    @property
    def v_elo(self):
        return self._v_elo

    @v_elo.setter
    def v_elo(self, val):
        self._v_elo = val

    @property
    def status(self):
        return self._status

    @status.setter
    def status(self, val):
        self._status = val

    @property
    def v_r(self):
        return float(self._v_mag * cos(radians(self._v_ang)))

    @property
    def v_i(self):
        return float(self._v_mag * sin(radians(self._v_ang)))


class Load:

    @overload
    def __init__(self,
                 load_data=None,
                 base_mva: Optional[float] = None,
                 bus: Optional[int] = None,
                 id: Optional[str] = None,
                 status: Optional[int] = None,
                 P: Optional[float] = None,
                 Q: Optional[float] = None,
                 IP: Optional[float] = None,
                 IQ: Optional[float] = None,
                 ZP: Optional[float] = None,
                 ZQ: Optional[float] = None):
        ...

    def __init__(self, **kwargs):
        _load_data = kwargs.pop("load_data", None)
        self._base_mva = kwargs.pop("base_mva", 100)

        self._bus = kwargs.pop("bus", _load_data.i)
        self._id = kwargs.pop("id", _load_data.id)
        self.status = kwargs.pop("status", _load_data.status)

        self._P = _init_var("P", name=f"Pl_{self._bus}_{self._id}",
                            val=kwargs.pop("P", _load_data.pl / self._base_mva))
        self._Q = _init_var("Q", name=f"Ql_{self._bus}_{self._id}",
                            val=kwargs.pop("Q", _load_data.ql / self._base_mva),
                            domain="Reals")
        self._IP = _init_var("IP", name=f"Ipl_{self._bus}_{self._id}",
                             val=kwargs.pop("IP", _load_data.ip / self._base_mva))
        self._IQ = _init_var("IQ", name=f"Iql_{self._bus}_{self._id}",
                             val=kwargs.pop("IQ", _load_data.iq / self._base_mva))
        self._ZP = _init_var("ZP", name=f"Zpl_{self._bus}_{self._id}",
                             val=kwargs.pop("ZP", _load_data.yp / self._base_mva))
        self._ZQ = _init_var("ZQ", name=f"Zql_{self._bus}_{self._id}",
                             val=kwargs.pop("ZQ", _load_data.yq / self._base_mva),
                             domain="Reals")

    def __repr__(self):
        msg = (f"Load(bus={self._bus},id='{self._id}',status={self._status},P={self._P.value}, Q={self._Q.value},"
               f"IP={self._IP.value}, IQ={self._IQ.value}, ZP={self._ZP.value}, ZQ={self._ZQ.value})")
        return msg

    def __str__(self):
        msg = (f"Load {self._bus}, '{self._id}' has P={self._P.value}, Q={self._Q.value}, and IP={self._IP.value}, "
               f"IP={self._IP.value}, IQ={self._IQ.value}, ZP={self._ZP.value}, ZQ={self._ZQ.value}.")
        return msg

    @property
    def bus(self):
        return self._bus

    @property
    def id(self):
        return self._id

    @property
    def key(self):
        key = (self._bus, self._id)
        return key

    @property
    def status(self):
        return self._status

    @status.setter
    def status(self, val):
        self._status = val

    @property
    def P(self):
        return self._P

    @P.setter
    def P(self, val):
        self._P.value = val

    @property
    def Q(self):
        return self._Q

    @Q.setter
    def Q(self, val):
        self._Q.value = val

    @property
    def IP(self):
        return self._IP

    @IP.setter
    def IP(self, val):
        self._IP.value = val

    @property
    def IQ(self):
        return self._IQ

    @IQ.setter
    def IQ(self, val):
        self._IQ.value = val

    @property
    def ZP(self):
        return self._ZP

    @ZP.setter
    def ZP(self, val):
        self._ZP.value = val

    @property
    def ZQ(self):
        return self._ZQ

    @ZQ.setter
    def ZQ(self, val):
        self._ZQ.value = val

    def calc_real_current(self,
                          Vr: Union[PyomoVar, float],
                          Vi: Union[PyomoVar, float],
                          P: Union[PyomoVar, float],
                          Q: Union[PyomoVar, float],
                          ZP: Union[PyomoVar, float] = None,
                          ZQ: Union[PyomoVar, float] = None,
                          IP: Union[PyomoVar, float] = None,
                          IQ: Union[PyomoVar, float] = None):

        Ir = (P * Vr + Q * Vi) / (Vr ** 2 + Vi ** 2)

        if ZP is not None or ZQ is not None:
            Ir += ZP * Vr - ZQ * Vi
        if IP is not None or IQ is not None:
            Ir += (IP * Vr + IQ * Vi) / (Vr ** 2 + Vi ** 2)
        return Ir

    def calc_imag_current(self,
                          Vr: Union[PyomoVar, float],
                          Vi: Union[PyomoVar, float],
                          P: Union[PyomoVar, float],
                          Q: Union[PyomoVar, float],
                          ZP: Union[PyomoVar, float] = None,
                          ZQ: Union[PyomoVar, float] = None,
                          IP: Union[PyomoVar, float] = None,
                          IQ: Union[PyomoVar, float] = None):

        Ii = (P * Vi - Q * Vr) / (Vr ** 2 + Vi ** 2)

        if ZP is not None or ZQ is not None:
            Ii += ZP * Vi - ZQ * Vr

        if IP is not None or IQ is not None:
            Ii += (IP * Vi - IQ * Vr) / (Vr ** 2 + Vi ** 2)

        return Ii


class Generator:
    @overload
    def __init__(self,
                 gen_data=None,
                 base_mva: Optional[float] = None,
                 bus: Optional[int] = None,
                 id: Optional[str] = None,
                 remote_bus: Optional[int] = None,
                 status: Optional[int] = None,
                 P: Optional[float] = None,
                 Q: Optional[float] = None,
                 Q_max: Optional[float] = None,
                 Q_min: Optional[float] = None,
                 P_max: Optional[float] = None,
                 P_min: Optional[float] = None,
                 fuel: Optional[str] = None,
                 unit_type: Optional[str] = None):
        ...

    def __init__(self, **kwargs):
        gen_data = kwargs.pop("gen_data", None)
        base_mva = kwargs.pop("base_mva", 100)

        self._bus = kwargs.pop("bus", gen_data.i)
        self._id = kwargs.pop("id", gen_data.id)
        self._remote_bus = kwargs.pop("remote_bus", gen_data.ireg)
        self._status = kwargs.pop("status", gen_data.stat)
        self._P = _init_var("P", name=f"Pg_{self._bus}_{self._id}",
                            val=kwargs.pop("P", float(gen_data.pg / base_mva)),
                            lb=kwargs.pop("P_min", float(gen_data.pb / base_mva)),
                            ub=kwargs.pop("P_max", float(gen_data.pt / base_mva))
                            )
        self._Q = _init_var("Q", name=f"Qg_{self._bus}_{self._id}",
                            val=kwargs.pop("Q", float(gen_data.qg / base_mva)),
                            domain="Reals",
                            lb=kwargs.pop("Q_min", float(gen_data.qb / base_mva)),
                            ub=kwargs.pop("Q_max", float(gen_data.qt / base_mva))
                            )
        self._fuel = kwargs.pop("fuel", "Unknown")
        self._unit_type = kwargs.pop("unit_type", "Unknown")

    def __repr__(self):
        msg = (f"Generator(bus={self._bus},id='{self._id}',status={self._status},P={self._P.value}, Q={self._Q.value}, "
               f"fuel={self._fuel})")
        return msg

    def __str__(self):
        msg = (f"{self._unit_type} Gen {self._bus}, '{self._id}' has P={self._P.value}, Q={self._Q.value}, and uses"
               f" {self._fuel} fuel.")
        return msg

    @property
    def bus(self):
        return self._bus

    @property
    def id(self):
        return self._id

    @property
    def remote_bus(self):
        return self._remote_bus

    @remote_bus.setter
    def remote_bus(self, val):
        self._remote_bus = val

    @property
    def status(self):
        return self._status

    @status.setter
    def status(self, val):
        self._status = val

    @property
    def key(self):
        key = (self._bus, self._id)
        return key

    @property
    def P(self):
        return self._P

    @P.setter
    def P(self, val):
        self._P.value = val

    @property
    def Q(self):
        return self._Q

    @Q.setter
    def Q(self, val):
        self._Q.value = val

    @property
    def Q_max(self):
        return self._Q.ub

    @property
    def Q_min(self):
        return self._Q.lb

    @property
    def P_max(self):
        return self._P.ub

    @property
    def P_min(self):
        return self._P.lb

    @property
    def fuel(self):
        return self._fuel

    @fuel.setter
    def fuel(self, val):
        self._fuel = val

    @property
    def unit_type(self):
        return self._unit_type

    @unit_type.setter
    def unit_type(self, val):
        self._unit_type = val

    def calc_real_current(self,
                          Vr: Union[PyomoVar, float],
                          Vi: Union[PyomoVar, float],
                          P: Union[PyomoVar, float],
                          Q: Union[PyomoVar, float]):
        Ir = (P * Vr + Q * Vi) / (Vr ** 2 + Vi ** 2)
        return Ir

    def calc_imag_current(self,
                          Vr: Union[PyomoVar, float],
                          Vi: Union[PyomoVar, float],
                          P: Union[PyomoVar, float],
                          Q: Union[PyomoVar, float]):
        Ii = (P * Vi - Q * Vr) / (Vr ** 2 + Vi ** 2)
        return Ii


class Branch:
    @overload
    def __init__(self,
                 branch_data=None,
                 base_mva: Optional[float] = None,
                 from_bus: Optional[int] = None,
                 to_bus: Optional[int] = None,
                 ckt: Optional[str] = None,
                 status: Optional[int] = None,
                 r: Optional[float] = None,
                 x: Optional[float] = None,
                 b_sh: Optional[float] = None,
                 rate_a: Optional[float] = None,
                 rate_b: Optional[float] = None,
                 rate_c: Optional[float] = None):
        ...

    def __init__(self, **kwargs):
        _branch_data = kwargs.pop("branch_data", None)

        self._from_bus = kwargs.pop("from_bus", _branch_data.i)
        self._to_bus = kwargs.pop("to_bus", _branch_data.j)
        self._ckt = kwargs.pop("ckt", _branch_data.ckt)
        self._status = kwargs.pop("status", _branch_data.st)
        self._r = kwargs.pop("r", _branch_data.r)
        self._x = kwargs.pop("x", _branch_data.x)
        self._b_sh = kwargs.pop("b_sh", _branch_data.b)
        self._g_pu = self._r / (self._r ** 2 + self._x ** 2)
        self._b_pu = -self._x / (self._r ** 2 + self._x ** 2)
        self._i_mag = 0.0
        self._rate_a = kwargs.pop("rate_a", _branch_data.ratea)
        self._rate_b = kwargs.pop("rate_b", _branch_data.rateb)
        self._rate_c = kwargs.pop("rate_c", _branch_data.ratec)
        self._base_mva = kwargs.pop("base_mva", 100)

        self._ratings, self._i_max, self._i_max_ctg = self.calc_ratings(self._rate_a, self._rate_b, self._rate_c,
                                                                        self._base_mva)

    def __repr__(self):
        msg = f"Branch(from_bus={self._from_bus},to_bus={self._to_bus}, ckt='{self._ckt}',status={self._status})"
        return msg

    def __str__(self):
        msg = f"Branch from bus is {self._from_bus}, to bus is {self._to_bus}, and ckt is '{self._ckt}'."
        return msg

    @property
    def from_bus(self):
        return self._from_bus

    @from_bus.setter
    def from_bus(self, val):
        self._from_bus = val

    @property
    def to_bus(self):
        return self._to_bus

    @to_bus.setter
    def to_bus(self, val):
        self._to_bus = val

    @property
    def ckt(self):
        return self._ckt

    @ckt.setter
    def ckt(self, val):
        self._ckt = val

    @property
    def status(self):
        return self._status

    @status.setter
    def status(self, val):
        self._status = val

    @property
    def r(self):
        return self._r

    @r.setter
    def r(self, val):
        self._r = val

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, val):
        self._x = val

    @property
    def b_sh(self):
        return self._b_sh

    @b_sh.setter
    def b_sh(self, val):
        self._b_sh = val

    @property
    def g_pu(self):
        return self._g_pu

    @g_pu.setter
    def g_pu(self, val):
        self._g_pu = val

    @property
    def b_pu(self):
        return self._b_pu

    @b_pu.setter
    def b_pu(self, val):
        self._b_pu = val

    @property
    def i_mag(self):
        return self._i_mag

    @i_mag.setter
    def i_mag(self, val):
        self._i_mag = val

    @property
    def rate_a(self):
        return self._rate_a

    @rate_a.setter
    def rate_a(self, val):
        self._rate_a = val

    @property
    def rate_b(self):
        return self._rate_b

    @rate_b.setter
    def rate_b(self, val):
        self._rate_b = val

    @property
    def rate_c(self):
        return self._rate_c

    @rate_c.setter
    def rate_c(self, val):
        self._rate_c = val

    @property
    def base_mva(self):
        return self._base_mva

    @base_mva.setter
    def base_mva(self, val):
        self._base_mva = val

    @property
    def ratings(self):
        return self._ratings

    @ratings.setter
    def ratings(self, val):
        self._ratings = val

    @property
    def i_max(self):
        return self._i_max

    @i_max.setter
    def i_max(self, val):
        self._i_max = val

    @property
    def i_max_ctg(self):
        return self._i_max_ctg

    @i_max_ctg.setter
    def i_max_ctg(self, val):
        self._i_max_ctg = val

    @staticmethod
    def calc_ratings(rate_A, rate_B, rate_C, base_mva):
        ratings = [rate_A, rate_B, rate_C]
        # # Add ammeters at every non-transformer branch # #
        rating_P = max(rate_A, rate_B)
        rating_P = max(rating_P, rate_C)  # This is in MVA

        # The input should be square of max_current in pu
        max_current = (rating_P / base_mva)
        max_current_ctg = (rate_C / base_mva)

        if max_current == 0:
            max_current = None
        else:
            max_current = round(max_current, 3)

        return ratings, max_current, max_current_ctg

    def calc_real_current(self,
                          Vr_from: Union[PyomoVar, float],
                          Vr_to: Union[PyomoVar, float],
                          Vi_from: Union[PyomoVar, float],
                          Vi_to: Union[PyomoVar, float],
                          bus: float):

        Vr_line = Vr_from - Vr_to
        Vi_line = Vi_from - Vi_to

        Ir_from = (Vr_line * self._g_pu) - (Vi_line * self._b_pu) - (self._b_sh * 0.5 * Vi_from)
        Ir_to = (- Vr_line * self._g_pu) + (Vi_line * self._b_pu) - (self._b_sh * 0.5 * Vi_to)

        Ir = Ir_from if bus == self._from_bus else Ir_to
        return Ir

    def calc_imag_current(self,
                          Vr_from: Union[PyomoVar, float],
                          Vr_to: Union[PyomoVar, float],
                          Vi_from: Union[PyomoVar, float],
                          Vi_to: Union[PyomoVar, float],
                          bus: float):

        Vr_line = Vr_from - Vr_to
        Vi_line = Vi_from - Vi_to

        Ii_from = (Vi_line * self._g_pu) + (Vr_line * self._b_pu) + (self._b_sh * 0.5 * Vr_from)
        Ii_to = (- Vi_line * self._g_pu) - (Vr_line * self._b_pu) + (self._b_sh * 0.5 * Vr_to)

        Ii = Ii_from if bus == self._from_bus else Ii_to
        return Ii


class Transformer:
    @overload
    def __init__(self,
                 transformer_data=None,
                 from_bus: Optional[int] = None,
                 to_bus: Optional[int] = None,
                 ckt: Optional[str] = None,
                 status: Optional[int] = None,
                 impedance_code: Optional[int] = None,
                 nomv1: Optional[float] = None,
                 nomv2: Optional[float] = None,
                 windv1: Optional[float] = None,
                 windv2: Optional[float] = None,
                 kv1: Optional[float] = None,
                 kv2: Optional[float] = None,
                 ang: Optional[float] = None,
                 tr: Optional[float] = None,
                 r: Optional[float] = None,
                 x: Optional[float] = None,
                 g_sh: Optional[float] = None,
                 b_sh: Optional[float] = None,
                 sbase: Optional[float] = None,
                 sbase12: Optional[float] = None,
                 cw: Optional[int] = None,
                 cm: Optional[int] = None,
                 r12: Optional[float] = None,
                 x12: Optional[float] = None,
                 mag1: Optional[float] = None,
                 mag2: Optional[float] = None,
                 b_c: Optional[float] = None,
                 rate_a: Optional[float] = None,
                 rate_b: Optional[float] = None,
                 rate_c: Optional[float] = None,
                 i_max: Optional[float] = None):
        ...

    def __init__(self, **kwargs):
        _transformer_data = kwargs.pop("transformer_data", None)

        self._from_bus = kwargs.pop("from_bus", _transformer_data.i)
        self._to_bus = kwargs.pop("to_bus", _transformer_data.j)
        self._ckt = kwargs.pop("ckt", _transformer_data.ckt)
        self._status = kwargs.pop("status", _transformer_data.stat)
        self._impedance_code = kwargs.pop("impedance_code", _transformer_data.cz)
        self._nomv1 = kwargs.pop("nomv1", _transformer_data.nomv1)
        self._nomv2 = kwargs.pop("nomv2", _transformer_data.nomv2)
        self._windv1 = kwargs.pop("windv1", _transformer_data.windv1)
        self._windv2 = kwargs.pop("windv2", _transformer_data.windv2)
        self._kv1 = kwargs.pop("kv1", 0.0)
        self._kv2 = kwargs.pop("kv2", 0.0)
        _ang = kwargs.pop("ang", _transformer_data.ang1)
        self._ang = _ang if _ang < 90 else _ang - 360
        self._tr = kwargs.pop("tr", None)
        self._r = kwargs.pop("r", None)
        self._x = kwargs.pop("x", None)
        self._g_sh = kwargs.pop("g_sh", None)
        self._b_sh = kwargs.pop("b_sh", None)
        self._sbase = kwargs.pop("sbase", 100)
        self._sbase12 = kwargs.pop("sbase12", _transformer_data.sbase12)
        self._cw = kwargs.pop("winding_data_code", _transformer_data.cw)
        self._cm = kwargs.pop("magnetizing_admittance_code", _transformer_data.cm)
        self._r12 = kwargs.pop("r12", _transformer_data.r12)
        self._x12 = kwargs.pop("x12", _transformer_data.x12)
        self._mag1 = kwargs.pop("mag1", _transformer_data.mag1)
        self._mag2 = kwargs.pop("mag2", _transformer_data.mag2)

        if None in [self._tr, self._r, self._x, self._g_sh, self._b_sh]:
            _tr, _r, _x, _g_sh, _b_sh = self.calc_impedance()
            self._tr = _tr
            self._r = _r
            self._x = _x
            self._g_sh = _g_sh
            self._b_sh = _b_sh

        self._b_c = 0
        self._rate_a = kwargs.pop("rate_a", _transformer_data.rata1)
        self._rate_b = kwargs.pop("rate_b", _transformer_data.ratb1)
        self._rate_c = kwargs.pop("rate_c", _transformer_data.ratc1)

        rating = self.max_current(self._rate_a, self._rate_b, self._r, self._sbase)

        self._i_max = None if rating < 1e-7 else rating
        self._i_mag = 0.0
        self._g_pu = self._r / (self._r ** 2 + self._x ** 2)
        self._b_pu = -self._x / (self._r ** 2 + self._x ** 2)

    def __repr__(self):
        msg = f"Transformer(from_bus={self._from_bus},to_bus={self._to_bus}, ckt='{self._ckt}',status={self._status})"
        return msg

    def __str__(self):
        msg = f"Transformer from bus is {self._from_bus}, to bus is {self._to_bus}, and ckt is '{self._ckt}'."
        return msg

    @property
    def from_bus(self):
        return self._from_bus

    @from_bus.setter
    def from_bus(self, val):
        self._from_bus = val

    @property
    def to_bus(self):
        return self._to_bus

    @to_bus.setter
    def to_bus(self, val):
        self._to_bus = val

    @property
    def ckt(self):
        return self._ckt

    @ckt.setter
    def ckt(self, val):
        self._ckt = val

    @property
    def nomv1(self):
        return self._nomv1

    @nomv1.setter
    def nomv1(self, val):
        self._nomv1 = val

    @property
    def nomv2(self):
        return self._nomv2

    @nomv2.setter
    def nomv2(self, val):
        self._nomv2 = val

    @property
    def windv1(self):
        return self._windv1

    @windv1.setter
    def windv1(self, val):
        self._windv1 = val

    @property
    def windv2(self):
        return self._windv2

    @windv2.setter
    def windv2(self, val):
        self._windv2 = val

    @property
    def kv1(self):
        return self._kv1

    @kv1.setter
    def kv1(self, val):
        self._kv1 = val

    @property
    def kv2(self):
        return self._kv2

    @kv2.setter
    def kv2(self, val):
        self._kv2 = val

    @property
    def ang(self):
        return self._ang

    @ang.setter
    def ang(self, val):
        self._ang = val

    @property
    def tr(self):
        return self._tr

    @tr.setter
    def tr(self, val):
        self._tr = val

    @property
    def r(self):
        return self._r

    @r.setter
    def r(self, val):
        self._r = val

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, val):
        self._x = val

    @property
    def g_sh(self):
        return self._g_sh

    @g_sh.setter
    def g_sh(self, val):
        self._g_sh = val

    @property
    def b_sh(self):
        return self._b_sh

    @b_sh.setter
    def b_sh(self, val):
        self._b_sh = val

    @property
    def sbase(self):
        return self._sbase

    @sbase.setter
    def sbase(self, val):
        self._sbase = val

    @property
    def status(self):
        return self._status

    @status.setter
    def status(self, val):
        self._status = val

    @property
    def i_max(self):
        return self._i_max

    @i_max.setter
    def i_max(self, val):
        self._i_max = val

    @property
    def i_mag(self):
        return self._i_mag

    @i_mag.setter
    def i_mag(self, val):
        self._i_mag = val

    @property
    def g_pu(self):
        return self._g_pu

    @g_pu.setter
    def g_pu(self, val):
        self._g_pu = val

    @property
    def b_pu(self):
        return self._b_pu

    @b_pu.setter
    def b_pu(self, val):
        self._b_pu = val

    def max_current(self,
                    rate_a: Optional[float],
                    rate_b: Optional[float],
                    rate_c: Optional[float],
                    sbase: float):
        rating_A = rate_a / sbase if rate_a is not None else 0.
        rating_B = rate_b / sbase if rate_b is not None else 0.
        rating_C = rate_c / sbase if rate_c is not None else 0.
        max_rating = max(rating_A, rating_B)
        max_rating = round(max(max_rating, rating_C), 3)
        return max_rating

    def calc_impedance(self):
        nomv1 = self._nomv1 if self._nomv1 > 0 else self._kv1
        nomv2 = self._nomv2 if self._nomv2 > 0 else self._kv2

        tr_i_opt = {
            1: self._windv1,
            2: self._windv1 / self._kv1,
            3: self._windv1 * nomv1 / self._kv1
        }

        tr_j_opt = {
            1: self._windv2,
            2: self._windv2 / self._kv2,
            3: self._windv2 * nomv2 / self._kv2
        }

        tr_i = tr_i_opt.get(self._cw)
        tr_j = tr_j_opt.get(self._cw)

        tr = tr_i / tr_j
        tr_j_2 = tr_j * tr_j
        base_impedance = self._sbase / self._sbase12 * tr_j_2
        r_pu = self._r12 / (1e6 * self._sbase12)
        x_pu = self._x12

        r_loss_init = {
            1: self._r12 * tr_j_2,
            2: self._r12 * tr_j_2 * self._sbase / self._sbase12,
            3: r_pu * base_impedance
        }

        x_loss_init = {
            1: self._x12 * tr_j_2,
            2: self._x12 * tr_j_2 * self._sbase / self._sbase12,
            3: sqrt(x_pu * x_pu - r_pu * r_pu) * base_impedance
        }

        r_loss = r_loss_init.get(self._impedance_code)
        x_loss = x_loss_init.get(self._impedance_code)

        s_loss = self._mag2 * self._sbase12 / self._sbase
        mag1 = self._mag1
        mag2 = self._mag2

        g_mag_init = {
            1: mag1,
            2: mag1 / (1e6 * self._sbase) if (mag1 - s_loss) < -1e-6 else 0
        }
        g_mag = g_mag_init.get(self._cm)

        b_mag_init = {
            1: mag2,
            2: sqrt(s_loss * s_loss - g_mag * g_mag) if (mag1 - s_loss) < -1e-6 else 0
        }

        b_mag = b_mag_init.get(self._cm)

        return tr, r_loss, x_loss, g_mag, b_mag

    def calc_real_current(self,
                          Vr_from: Union[PyomoVar, float],
                          Vr_to: Union[PyomoVar, float],
                          Vi_from: Union[PyomoVar, float],
                          Vi_to: Union[PyomoVar, float],
                          bus: float):
        G_cos = self._g_pu * cos(radians(self._ang))
        B_cos = self._b_pu * cos(radians(self._ang))
        G_sin = self._g_pu * sin(radians(self._ang))
        B_sin = self._b_pu * sin(radians(self._ang))

        inv_tr = 1 / self._tr
        inv_tr2 = inv_tr ** 2
        Bt = self._b_pu + self._b_c * 0.5
        Mr_from = (G_cos - B_sin) * inv_tr
        Mi_from = (G_sin + B_cos) * inv_tr
        Mr_to = (G_cos + B_sin) * inv_tr
        Mi_to = (B_cos - G_sin) * inv_tr

        Ir_from = (self._g_pu * inv_tr2 * Vr_from) - Mr_from * Vr_to - (Bt * inv_tr2) * Vi_from + Mi_from * Vi_to
        Ir_to = -Mr_to * Vr_from + self._g_pu * Vr_to + Mi_to * Vi_from - Bt * Vi_to

        Ir = Ir_from if bus == self._from_bus else Ir_to
        return Ir

    def calc_imag_current(self,
                          Vr_from: Union[PyomoVar, float],
                          Vr_to: Union[PyomoVar, float],
                          Vi_from: Union[PyomoVar, float],
                          Vi_to: Union[PyomoVar, float],
                          bus: float):
        G_cos = self._g_pu * cos(radians(self._ang))
        B_cos = self._b_pu * cos(radians(self._ang))
        G_sin = self._g_pu * sin(radians(self._ang))
        B_sin = self._b_pu * sin(radians(self._ang))

        inv_tr = 1 / self._tr
        inv_tr2 = inv_tr ** 2
        Bt = self._b_pu + self._b_c * 0.5
        Mr_from = (G_cos - B_sin) * inv_tr
        Mi_from = (G_sin + B_cos) * inv_tr
        Mr_to = (G_cos + B_sin) * inv_tr
        Mi_to = (B_cos - G_sin) * inv_tr

        Ii_from = (Bt * inv_tr2) * Vr_from - Mi_from * Vr_to + (self._g_pu * inv_tr2 * Vi_from) - Mr_from * Vi_to
        Ii_to = -Mi_to * Vr_from + Bt * Vr_to - Mr_to * Vi_from + self._g_pu * Vi_to

        Ii = Ii_from if bus == self._from_bus else Ii_to
        return Ii


class Shunt:

    @overload
    def __init__(self,
                 shunt_data=None,
                 bus: Optional[int] = None,
                 id: Optional[str] = None,
                 status: Optional[int] = None,
                 g_init: Optional[float] = None,
                 b_init: Optional[float] = None,
                 g: Optional[float] = None,
                 b: Optional[float] = None,
                 base_mva: Optional[float] = None):
        ...

    def __init__(self, **kwargs):
        _shunt_data = kwargs.pop("shunt_data", None)
        self._bus = kwargs.pop("bus", _shunt_data.i)
        self._id = kwargs.pop("id", _shunt_data.id)
        self._status = kwargs.pop("status", _shunt_data.status)
        self._base_mva = kwargs.pop("base_mva", 100)
        _g = kwargs.pop("g_init", 0)
        _b = kwargs.pop("b_init", 1)
        self._g = kwargs.pop("g", float(_g / self._base_mva))
        self._b = kwargs.pop("b", float(_b / self._base_mva))

    def __repr__(self):
        msg = f"Shunt(bus={self._bus},id='{self._id}',status={self._status}, G={self._g}, B={self._b})"
        return msg

    def __str__(self):
        msg = f"Shunt {self._bus}, '{self._id}'  G={self._g}, and B={self._b}."
        return msg

    @property
    def bus(self):
        return self._bus

    @bus.setter
    def bus(self, val):
        self._bus = val

    @property
    def id(self):
        return self._id

    @id.setter
    def id(self, val):
        self._id = val

    @property
    def status(self):
        return self._status

    @status.setter
    def status(self, val):
        self._status = val

    @property
    def base_mva(self):
        return self._base_mva

    @base_mva.setter
    def base_mva(self, val):
        self._base_mva = val

    @property
    def g(self):
        return self._g

    @g.setter
    def g(self, val):
        self._g = val

    @property
    def b(self):
        return self._b

    @b.setter
    def b(self, val):
        self._b = val

    def calc_real_current(self,
                          Vr: Union[PyomoVar, float],
                          Vi: Union[PyomoVar, float]):
        Ir = self._g * Vr - self._b * Vi
        return Ir

    def calc_imag_current(self,
                          Vr: Union[PyomoVar, float],
                          Vi: Union[PyomoVar, float]):
        Ii = self._g * Vi + self._b * Vr
        return Ii


class SwitchedShunt:

    @overload
    def __init__(self,
                 shunt_data=None,
                 bus: Optional[int] = None,
                 id: Optional[str] = None,
                 status: Optional[int] = None,
                 g_init: Optional[float] = None,
                 b_init: Optional[float] = None,
                 g: Optional[float] = None,
                 b: Optional[float] = None,
                 base_mva: Optional[float] = None,
                 vswhi: Optional[float] = None,
                 vswlo: Optional[float] = None,
                 n1: Optional[int] = None,
                 n2: Optional[int] = None,
                 n3: Optional[int] = None,
                 n4: Optional[int] = None,
                 n5: Optional[int] = None,
                 n6: Optional[int] = None,
                 n7: Optional[int] = None,
                 n8: Optional[int] = None,
                 b1: Optional[float] = None,
                 b2: Optional[float] = None,
                 b3: Optional[float] = None,
                 b4: Optional[float] = None,
                 b5: Optional[float] = None,
                 b6: Optional[float] = None,
                 b7: Optional[float] = None,
                 b8: Optional[float] = None
                 ):
        ...

    def __init__(self, **kwargs):
        _shunt_data = kwargs.pop("shunt_data", None)
        self._bus = kwargs.pop("bus", _shunt_data.i)
        self._id = kwargs.pop("id", _shunt_data.id)
        self._status = kwargs.pop("status", _shunt_data.status)
        self._base_mva = kwargs.pop("base_mva", 100)
        _g_init = kwargs.pop("g_init", 0)
        _b_init = kwargs.pop("b_init", 1)
        self._g = kwargs.pop("g", float(_g_init / self._base_mva))
        self._b = kwargs.pop("b", float(_b_init / self._base_mva))
        self._v_swhi = kwargs.pop("vswhi", _shunt_data.vswhi)
        self._v_swlo = kwargs.pop("vswlo", _shunt_data.vswlo)
        self._n1 = kwargs.pop("n1", _shunt_data.n1)
        self._n2 = kwargs.pop("n2", _shunt_data.n2)
        self._n3 = kwargs.pop("n3", _shunt_data.n3)
        self._n4 = kwargs.pop("n4", _shunt_data.n4)
        self._n5 = kwargs.pop("n5", _shunt_data.n5)
        self._n6 = kwargs.pop("n6", _shunt_data.n6)
        self._n7 = kwargs.pop("n7", _shunt_data.n7)
        self._n8 = kwargs.pop("n8", _shunt_data.n8)
        self._b1 = kwargs.pop("b1", _shunt_data.b1)
        self._b2 = kwargs.pop("b2", _shunt_data.b2)
        self._b3 = kwargs.pop("b3", _shunt_data.b3)
        self._b4 = kwargs.pop("b4", _shunt_data.b4)
        self._b5 = kwargs.pop("b5", _shunt_data.b5)
        self._b6 = kwargs.pop("b6", _shunt_data.b6)
        self._b7 = kwargs.pop("b7", _shunt_data.b7)
        self._b8 = kwargs.pop("b8", _shunt_data.b8)

        _q_min, _q_max = self.compute_qmin_qmax(self._n1, self._n2, self._n3, self._n4, self._n5, self._n6, self._n7,
                                                self._n8, self._b1, self._b2, self._b3, self._b4, self._b5, self._b6,
                                                self._b7, self._b8)
        _q_init = _b_init * (self._v_swhi ** 2)

        self._Q = _init_var("Q", name=f"Qsh_{self._bus}_{self._id}",
                            val=kwargs.pop("Q", float(_q_init / self._base_mva)),
                            domain="Reals",
                            lb=kwargs.pop("Q_min", float(_q_min / self._base_mva)),
                            ub=kwargs.pop("Q_max", float(_q_max / self._base_mva))
                            )

    def __repr__(self):
        msg = f"SwitchedShunt(bus={self._bus},id='{self._id}',status={self._status}, Q={self._Q.value})"
        return msg

    def __str__(self):
        msg = f"Switched Shunt {self._bus}, '{self._id}' has Q={self._Q.value}."
        return msg

    @property
    def bus(self):
        return self._bus

    @bus.setter
    def bus(self, val):
        self._bus = val

    @property
    def id(self):
        return self._id

    @id.setter
    def id(self, val):
        self._id = val

    @property
    def status(self):
        return self._status

    @status.setter
    def status(self, val):
        self._status = val

    @property
    def base_mva(self):
        return self._base_mva

    @base_mva.setter
    def base_mva(self, val):
        self._base_mva = val

    @property
    def g(self):
        return self._g

    @g.setter
    def g(self, val):
        self._g = val

    @property
    def b(self):
        return self._b

    @b.setter
    def b(self, val):
        self._b = val

    @property
    def Q(self):
        return self._Q

    @Q.setter
    def Q(self, val):
        self._Q.value = val

    @property
    def Q_max(self):
        return self._Q.ub

    @property
    def Q_min(self):
        return self._Q.lb

    def compute_qmin_qmax(self,
                          n1: float,
                          n2: float,
                          n3: float,
                          n4: float,
                          n5: float,
                          n6: float,
                          n7: float,
                          n8: float,
                          b1: float,
                          b2: float,
                          b3: float,
                          b4: float,
                          b5: float,
                          b6: float,
                          b7: float,
                          b8: float):
        b_min = 0.0
        b_max = 0.0
        b1 = n1 * b1
        b2 = n2 * b2
        b3 = n3 * b3
        b4 = n4 * b4
        b5 = n5 * b5
        b6 = n6 * b6
        b7 = n7 * b7
        b8 = n8 * b8
        for b in [b1, b2, b3, b4, b5, b6, b7, b8]:
            if b > 0.0:
                b_max += b
            elif b < 0.0:
                b_min += b
            else:
                break
        q_max = b_max * (self._v_swhi ** 2)
        q_min = b_min * (self._v_swhi ** 2)
        return q_min, q_max

    def calc_real_current(self,
                          Vr: Union[PyomoVar, float],
                          Vi: Union[PyomoVar, float],
                          Q: Union[PyomoVar, float]):
        Ir = (Q * Vi) / (Vr ** 2 + Vi ** 2)
        return Ir

    def calc_imag_current(self,
                          Vr: Union[PyomoVar, float],
                          Vi: Union[PyomoVar, float],
                          Q: Union[PyomoVar, float]):
        Ii = (- Q * Vr) / (Vr ** 2 + Vi ** 2)
        return Ii
