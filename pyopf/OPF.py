"""Implementation of AC Optimal Power Flow (ACOPF)

  Author(s): Naeem Turner-Bandele

  Created Date: 02-13-2023

  Updated Date: 02-19-2023

  Email: naeem@naeem.engineer

  Status: Development

  Detailed Description:

  Typical usage example:

  opf = OPF()
"""
import os
from math import atan2, degrees, sqrt
from pathlib import Path
from typing import Optional

import pyomo.environ as pe
from pyomo.common.timing import TicTocTimer
from pyomo.opt import SolverFactory, SolverStatus, TerminationCondition
from pyomo.util.infeasible import log_close_to_bounds, log_infeasible_bounds, log_infeasible_constraints

from pyopf.util.Log import Log
from pyopf.util.save_results import save_compressed_results
from pyopf.util.update_grid_data import update_grid_data

__all__ = ["OPF"]


class OPF:
    """Optimization solver for the AC Optimal Power Flow problem using current-voltage formulation.
    """

    def __init__(self):
        self._model = None
        self._scenario = None
        self._objective = None
        self._solved = False
        self._results = None
        self._results_summary = None
        self._logger = None
        self._runtime = None

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, val):
        self._model = val

    @property
    def scenario(self):
        return self._scenario

    @scenario.setter
    def scenario(self, val):
        self._scenario = val

    @property
    def objective(self):
        return self._objective

    @objective.setter
    def objective(self, val):
        self._objective = val

    @property
    def solved(self):
        return self._solved

    @solved.setter
    def solved(self, val):
        self._solved = val

    @property
    def results(self):
        return self._results

    @results.setter
    def results(self, val):
        self._results = val

    @property
    def results_summary(self):
        return self._results_summary

    @results_summary.setter
    def results_summary(self, val):
        self._results_summary = val

    @property
    def logger(self):
        return self._logger

    @logger.setter
    def logger(self, val):
        self._logger = val

    @property
    def runtime(self):
        return self._runtime

    @runtime.setter
    def runtime(self, val):
        self._runtime = val

    def solve(self,
              scenario: str,
              grid_data: dict,
              filepaths: dict,
              objective: Optional[str] = None,
              opf_options: Optional[dict] = None,
              solver_options=None,
              pyomo_options=None):
        """
        Create and solve the uncertain scenario model using OPF
        Args:
            scenario: the name of the scenario being optimized
            grid_data: the grid data for the current scenario being optimized
            filepaths: a dictionary that tracks the filepaths of common directories
            objective: the objective of the optimization
            solver_options: the options for the nonlinear optimization solver being used
            pyomo_options: pyomo specific solver options that provided for details about the solving process
            opf_options: options to make changes to the opf model such as providing custom voltage mag bounds

        Returns:
            None
        """

        logger = Log(filepaths["log"], scenario, "PyOPF")

        self._logger = logger

        self.create_model(scenario, grid_data, objective=objective, opf_options=opf_options)

        logger.info(f"{scenario} Model created successfully")

        timer = TicTocTimer()
        timer.tic()
        self.solve_model(logger, grid_data, pyomo_options, solver_options)
        self._runtime = timer.toc()

    def create_model(self,
                     scenario: str,
                     grid_data: dict,
                     objective: str = None,
                     opf_options: Optional[dict] = None):
        """
        Create an OPF model to solve
        Args:
            scenario: the name of the scenario being optimized
            grid_data: the grid data for the current scenario being optimized
            objective: the objective of the optimization
            opf_options: options to make changes to the opf model such as providing custom voltage mag bounds

        Returns:
            None
        """
        # create concrete optimization model container
        model = pe.ConcreteModel()

        # name the model
        model.name = scenario

        # # == INIT MODEL SETS USED TO INDEX == # #
        model = self.create_model_sets(model, grid_data)

        # # == INIT OPTIMIZATION DECISION VARIABLES == # #
        model = self.create_model_decision_variables(model, grid_data, opf_options)

        # # == INIT OPTIMIZATION PARAMETERS (CONSTANTS) == # #
        model = self.create_model_parameters(model, grid_data)

        # # == DEFINE OPTIMIZATION OBJECTIVE FUNCTION == # #
        model = self.create_model_objective(model, objective)

        # # == DEFINE OPTIMIZATION CONSTRAINTS == # #
        model = self.create_model_constraints(model, grid_data)

        # store the model, scenario name, and model solving status internally

        self._model = model
        self._scenario = scenario
        self._solved = False

    def create_model_sets(self,
                          model: pe.ConcreteModel,
                          grid_data: dict) -> pe.ConcreteModel:
        """
        Create the OPF model sets which are used to index variables and parameters
        Args:
            model: Pyomo model
            grid_data: the grid data for the current scenario being optimized

        Returns:
            Pyomo model with added sets for indexing
        """
        # Buses set
        model.buses_set = pe.Set(
            initialize=[obj.bus for obj in grid_data["buses"].values() if obj.status],
            ordered=True
        )

        # Demand or Loads set
        model.loads_set = pe.Set(
            initialize=[(obj.bus, obj.id) for obj in grid_data["loads"].values() if obj.status],
            ordered=True
        )

        # Generators set
        model.generators_set = pe.Set(
            initialize=[(obj.bus, obj.id) for obj in grid_data["generators"].values() if obj.status],
            ordered=True
        )

        # Transformers set
        model.transformers_set = pe.Set(
            initialize=[(obj.from_bus, obj.to_bus, obj.ckt) for obj in grid_data["transformers"].values() if
                        obj.status],
            ordered=True
        )

        # Lines set
        model.lines_set = pe.Set(
            initialize=[(obj.from_bus, obj.to_bus, obj.ckt) for obj in grid_data["branches"].values() if
                        obj.status],
            ordered=True
        )

        # Switched shunts set
        model.switched_shunts_set = pe.Set(
            initialize=[(obj.bus, obj.id) for obj in grid_data["switched shunts"].values() if obj.status],
            ordered=True
        )

        return model

    def create_model_decision_variables(self,
                                        model: pe.ConcreteModel,
                                        grid_data: dict,
                                        opf_options: Optional[dict] = None) -> pe.ConcreteModel:
        """
        Create the OPF model decision variables
        Args:
            model: Pyomo model
            grid_data: the grid data for the current scenario being optimized
            opf_options: options to make changes to the opf model such as providing custom voltage mag bounds

        Returns:
            The pyomo model with the decision variables added
        """

        # Create generators active and reactive power variables
        def init_generator_active_power(m: pe.ConcreteModel,
                                        gen_bus: int,
                                        gen_id: str) -> float:
            """
            Initialize the generator active power
            Args:
                m: Pyomo model (unused but required)
                gen_bus: the generator bus number
                gen_id: the generator id number

            Returns:
                The value of the active power of the generator indexed at (gen_bus,gen_id)
            """
            P = grid_data["generators"][(gen_bus, gen_id)].P.value
            return P

        def init_generator_active_bounds(m: pe.ConcreteModel,
                                         gen_bus: int,
                                         gen_id: str) -> tuple[float, float]:
            """
            Initialize the generator active power bounds
            Args:
                m: Pyomo model (unused but a required input)
                gen_bus: the generator bus number
                gen_id: the generator id number

            Returns:
                The lower and upper bounds of the generator active power

            """
            generator = grid_data["generators"][(gen_bus, gen_id)]
            return generator.P_min, generator.P_max

        model.Pg = pe.Var(
            model.generators_set,
            domain=pe.Reals,
            initialize=init_generator_active_power,
            bounds=init_generator_active_bounds
        )

        def init_generator_reactive_bounds(m: pe.ConcreteModel,
                                           gen_bus: int,
                                           gen_id: str) -> tuple[float, float]:
            """
            Initialize the generator reactive power bounds
            Args:
                m: Pyomo model (unused but a required input)
                gen_bus: the generator bus number
                gen_id: the generator id number

            Returns:
                The lower and upper bounds of the generator reactive power

            """
            generator = grid_data["generators"][(gen_bus, gen_id)]
            return generator.Q_min, generator.Q_max

        model.Qg = pe.Var(
            model.generators_set,
            domain=pe.Reals,
            initialize=0.,
            bounds=init_generator_reactive_bounds
        )

        # Voltage magnitude (V_mag), real voltage (Vr), and imaginary voltage (Vi) at each bus
        def init_voltage_mag(m: pe.ConcreteModel,
                             bus: int) -> float:
            """
            Find bus at index bus in bus set and return its initial voltage magnitude setpoint
            Args:
                m: Pyomo model (unused but a required input)
                bus: the bus number

            Returns:
                The voltage magnitude setpoint at the bus
            """
            return 1.0

        def init_voltage_mag_bounds(m: pe.ConcreteModel,
                                    bus: int) -> tuple[float, float]:
            """
            Find bus at index bus in bus set and return its voltage limits
            Args:
                m: Pyomo model (unused but a required input)
                bus: the bus number

            Returns:
                The voltage magnitude bounds at the bus
            """
            if opf_options is not None and "voltage bounds" in opf_options:
                v_mag_bounds = opf_options.get("voltage bounds")
                v_mag_lb = v_mag_bounds[0]
                v_mag_ub = v_mag_bounds[1]
            else:
                v_mag_lb = float(grid_data["buses"][bus].v_nlo)
                v_mag_ub = float(grid_data["buses"][bus].v_nhi)
            return v_mag_lb, v_mag_ub

        model.V_mag = pe.Var(
            model.buses_set,
            domain=pe.NonNegativeReals,
            initialize=init_voltage_mag,
            bounds=init_voltage_mag_bounds
        )

        def init_real_voltage(m: pe.ConcreteModel,
                              bus: int) -> float:
            """
            Find bus at index bus in bus set and return its initial real voltage value
            Args:
                m: Pyomo model (unused but a required input)
                bus: the bus number

            Returns:
                The real voltage at the bus
            """
            # return grid_data["buses"][bus].v_r
            return 1.0

        model.Vr = pe.Var(
            model.buses_set,
            domain=pe.Reals,
            initialize=init_real_voltage
        )

        def init_imag_voltage(m: pe.ConcreteModel,
                              bus: int) -> float:
            """
            Find bus at index bus in bus set and return its initial imaginary voltage value
            Args:
                m: Pyomo model (unused but a required input)
                bus: the bus number

            Returns:
                The imaginary voltage at the bus
            """
            # return grid_data["buses"][bus].v_i
            return 0.

        model.Vi = pe.Var(
            model.buses_set,
            domain=pe.Reals,
            initialize=init_imag_voltage
        )

        # fix the slack bus angle
        for obj in grid_data["slacks"].values():
            model.Vi[obj.bus].fix(0.)

        # Line current flows (I_line) and transformer current flows (I_transformer)
        def init_line_current_bounds(m: pe.ConcreteModel,
                                     from_bus: int,
                                     to_bus: int,
                                     ckt: str) -> tuple[None, Optional[float]]:
            """
            Initialize the line current magnitude bounds at the indicated location
            Args:
                m: Pyomo model
                from_bus: the from bus number of the line
                to_bus: the to bus number of the line
                ckt: the circuit of the line

            Returns:
                The line current magnitude bounds
            """
            line = grid_data["branches"][(from_bus, to_bus, ckt)]
            I_mag_bounds = (None, None) if line.i_max is None else (None, line.i_max ** 2)
            return I_mag_bounds

        model.I_line = pe.Var(
            model.lines_set,
            domain=pe.NonNegativeReals,
            initialize=0.,
            bounds=init_line_current_bounds
        )

        def init_transformer_current_bounds(m: pe.ConcreteModel,
                                            from_bus: int,
                                            to_bus: int,
                                            ckt: str) -> tuple[None, Optional[float]]:
            """
            Initialize the transformer current magnitude bounds at the indicated location
            Args:
                m: Pyomo model
                from_bus: the from bus number of the transformer
                to_bus: the to bus number of the transformer
                ckt: the circuit of the transformer

            Returns:
                The transformer current magnitude bounds
            """
            line = grid_data["transformers"][(from_bus, to_bus, ckt)]
            I_mag_bounds = (None, None) if line.i_max is None else (None, line.i_max ** 2)
            return I_mag_bounds

        model.I_transformer = pe.Var(
            model.transformers_set,
            domain=pe.NonNegativeReals,
            initialize=0.,
            bounds=init_transformer_current_bounds
        )

        def init_switched_shunt_power(m: pe.ConcreteModel,
                                      shunt_bus: int,
                                      shunt_id: str) -> float:
            """
            Initialize the generator reactive power
            Args:
                m: Pyomo model (unused but required)
                shunt_bus: the shunt bus number
                shunt_id: the shunt id

            Returns:
                The value of the reactive power of the generator indexed at (gen_bus,gen_id)
            """
            return grid_data["switched shunts"][(shunt_bus, shunt_id)].Q.value

        def init_switched_shunt_bounds(m: pe.ConcreteModel,
                                       shunt_bus: int,
                                       shunt_id: str) -> tuple[float, float]:
            """
            Initialize the generator reactive power bounds
            Args:
                m: Pyomo model (unused but a required input)
                shunt_bus: the shunt bus number
                shunt_id: the shunt id

            Returns:
                The lower and upper bounds of the generator reactive power

            """
            switched_shunt = grid_data["switched shunts"][(shunt_bus, shunt_id)]
            return switched_shunt.Q_min, switched_shunt.Q_max

        model.Qsh = pe.Var(
            model.switched_shunts_set,
            domain=pe.Reals,
            initialize=init_switched_shunt_power,
            bounds=init_switched_shunt_bounds
        )

        return model

    def create_model_parameters(self,
                                model: pe.ConcreteModel,
                                grid_data: dict) -> pe.ConcreteModel:
        """
        Create the OPF model parameters
        Args:
            model: Pyomo model
            grid_data: the grid data for the current scenario being optimized

        Returns:
            The pyomo model with the parameters added
        """
        # create load real and reactive power parameters
        Pd = {obj.key: obj.P.value for obj in grid_data["loads"].values()}
        Qd = {obj.key: obj.Q.value for obj in grid_data["loads"].values()}
        model.Pd = pe.Param(model.loads_set, initialize=Pd, domain=pe.Reals)
        model.Qd = pe.Param(model.loads_set, initialize=Qd, domain=pe.Reals)

        # costs
        model.a_cost = pe.Param(model.generators_set, default=0.)
        model.b_cost = pe.Param(model.generators_set, default=1.0)
        model.c_cost = pe.Param(model.generators_set, default=0.0)

        return model

    def create_model_objective(self,
                               model: pe.ConcreteModel,
                               objective: Optional[str]) -> pe.ConcreteModel:
        """
        Create the objective of the OPF
        Args:
            model: Pyomo model
            objective: the objective function to use

        Returns:
            Pyomo model with added objective function
        """
        if objective is None:
            objective = "min cost"

        def min_cost(m):
            """
            Minimize the cost of generation
            Args:
                m: Pyomo model

            Returns:
                The objective function as a Pyomo expression summation
            """
            return sum(m.a_cost[g] + (m.b_cost[g] * m.Pg[g]) + (m.c_cost[g] * m.Pg[g]) ** 2 for g in
                       m.generators_set)

        self._objective = objective
        obj_options = {
            "min cost": min_cost
        }
        model.obj = pe.Objective(rule=obj_options.get(objective))
        return model

    def create_model_constraints(self,
                                 model: pe.ConcreteModel,
                                 grid_data: dict) -> pe.ConcreteModel:
        """
        Create the OPF model constraints
        Args:
            model: Pyomo model
            grid_data: the grid data for the current scenario being optimized

        Returns:
            The pyomo model with the constraints added
        """

        # Real KCL constraint
        def kcl_real_constraint(m: pe.ConcreteModel,
                                bus: int) -> pe.Expression:
            """
            Constraint that ensures KCL is satisfied for real currents
            Args:
                m: Pyomo model
                bus: bus number

            Returns:
                Constraint as a pyomo expression

            """
            Ir_gen = sum(
                [
                    obj.calc_real_current(
                        m.Vr[obj.bus], m.Vi[obj.bus], m.Pg[obj.key], m.Qg[obj.key]
                    ) for obj in grid_data["generators"].values() if obj.bus == bus and obj.status
                ]
            )
            Ir_load = sum(
                [
                    obj.calc_real_current(
                        m.Vr[obj.bus], m.Vi[obj.bus], m.Pd[obj.key], m.Qd[obj.key]
                    ) for obj in grid_data["loads"].values() if obj.bus == bus and obj.status
                ]
            )

            def calc_Ir_delivery(delivery_objs: dict) -> pe.Expression:
                """
                Calculate the real current from power delivery objects such as transformers and branches
                Args:
                    delivery_objs: transformers or branches

                Returns:
                    Real current from the delivery objects as a pyomo expression
                """
                I = [obj.calc_real_current(
                    m.Vr[obj.from_bus], m.Vr[obj.to_bus], m.Vi[obj.from_bus], m.Vi[obj.to_bus], bus
                ) for obj in delivery_objs.values() if obj.from_bus == bus and obj.status]
                I += [obj.calc_real_current(
                    m.Vr[obj.from_bus], m.Vr[obj.to_bus], m.Vi[obj.from_bus], m.Vi[obj.to_bus], bus
                ) for obj in delivery_objs.values() if obj.to_bus == bus and obj.status]
                return sum(I)

            Ir_line = calc_Ir_delivery(grid_data["branches"])
            Ir_transformer = calc_Ir_delivery(grid_data["transformers"])
            Ir_shunt = sum([
                obj.calc_real_current(m.Vr[obj.bus], m.Vi[obj.bus]) for obj in grid_data["shunts"].values()
                if obj.bus == bus and obj.status
            ])

            Ir_swshunt = sum([
                obj.calc_real_current(m.Vr[obj.bus], m.Vi[obj.bus], m.Qsh[(obj.bus, obj.id)])
                for obj in grid_data["switched shunts"].values()
                if obj.bus == bus and obj.status
            ])

            Ir_constraint = Ir_gen - Ir_load == Ir_line + Ir_transformer + Ir_shunt + Ir_swshunt
            return Ir_constraint

        model.kcl_real_constraint = pe.Constraint(model.buses_set, rule=kcl_real_constraint)

        # Imag KCL constraint
        def kcl_imag_constraint(m: pe.ConcreteModel,
                                bus: int) -> pe.Expression:
            """
            Constraint that ensures KCL is satisfied for imaginary currents
            Args:
                m: Pyomo model
                bus: bus number

            Returns:
                Constraint as a pyomo expresssion
            """
            Ii_gen = sum(
                [
                    obj.calc_imag_current(
                        m.Vr[obj.bus], m.Vi[obj.bus], m.Pg[obj.key], m.Qg[obj.key]
                    ) for obj in
                    grid_data["generators"].values() if obj.bus == bus and obj.status
                ]
            )
            Ii_load = sum(
                [
                    obj.calc_imag_current(
                        m.Vr[obj.bus], m.Vi[obj.bus], m.Pd[obj.key], m.Qd[obj.key]
                    ) for obj in grid_data["loads"].values() if
                    obj.bus == bus and obj.status
                ]
            )

            def calc_Ii_delivery(delivery_objs: dict) -> pe.Expression:
                """
                Calculate the imaginary current from power delivery objects such as transformers and branches
                Args:
                    delivery_objs: transformers or branches

                Returns:
                    Imaginary current from the delivery objects as a pyomo expression
                """
                I = [obj.calc_imag_current(
                    m.Vr[obj.from_bus], m.Vr[obj.to_bus], m.Vi[obj.from_bus], m.Vi[obj.to_bus], bus
                ) for obj in delivery_objs.values() if obj.from_bus == bus and obj.status]
                I += [obj.calc_imag_current(
                    m.Vr[obj.from_bus], m.Vr[obj.to_bus], m.Vi[obj.from_bus], m.Vi[obj.to_bus], bus
                ) for obj in delivery_objs.values() if obj.to_bus == bus and obj.status]
                return sum(I)

            Ii_line = calc_Ii_delivery(grid_data["branches"])
            Ii_transformer = calc_Ii_delivery(grid_data["transformers"])
            Ii_shunt = sum([
                obj.calc_imag_current(m.Vr[obj.bus], m.Vi[obj.bus]) for obj in grid_data["shunts"].values()
                if obj.bus == bus and obj.status
            ])
            Ii_swshunt = sum([
                obj.calc_imag_current(m.Vr[obj.bus], m.Vi[obj.bus], m.Qsh[(obj.bus, obj.id)])
                for obj in grid_data["switched shunts"].values()
                if obj.bus == bus and obj.status
            ])
            Ii_constraint = Ii_gen - Ii_load == Ii_line + Ii_transformer + Ii_shunt + Ii_swshunt
            return Ii_constraint

        model.kcl_imag_constraint = pe.Constraint(model.buses_set, rule=kcl_imag_constraint)

        # Bus Voltage Magnitude constraint
        def v_mag_constraint(m: pe.ConcreteModel,
                             bus: int):
            """
            Constraint that ensures the bus voltage magnitude consists of real and imaginary voltages
            Args:
                m: Pyomo model
                bus: bus number

            Returns:
                Pyomo expression for the voltage magnitude at the bus
            """
            return m.V_mag[bus] == pe.sqrt(m.Vr[bus] ** 2 + m.Vi[bus] ** 2)

        model.v_mag_constraint = pe.Constraint(model.buses_set, rule=v_mag_constraint)

        # Line current flow constraint
        def line_current_constraint(m: pe.ConcreteModel,
                                    from_bus: int,
                                    to_bus: int,
                                    ckt: str) -> pe.Expression:
            """
            Constraint that ensures that the current magnitude at a line consists of real and imaginary current
            Args:
                m: Pyomo model
                from_bus: the line from bus number
                to_bus: the line to bus number
                ckt: the line circuit identifier

            Returns:
                Pyomo expression for the line flow

            """
            line = grid_data["branches"][(from_bus, to_bus, ckt)]
            Ir = line.calc_real_current(m.Vr[from_bus], m.Vr[to_bus], m.Vi[from_bus], m.Vi[to_bus], from_bus)
            Ii = line.calc_imag_current(m.Vr[from_bus], m.Vr[to_bus], m.Vi[from_bus], m.Vi[to_bus], from_bus)
            I_mag = Ir ** 2 + Ii ** 2
            return m.I_line[(from_bus, to_bus, ckt)] == I_mag

        model.i_line_constraint = pe.Constraint(model.lines_set, rule=line_current_constraint)

        # Transformer current flow constraint
        def transformer_current_constraint(m: pe.ConcreteModel,
                                           from_bus: int,
                                           to_bus: int,
                                           ckt: str) -> pe.Expression:
            """
            Constraint that ensures that the current magnitude at a transformer consists of real and imaginary current
            Args:
                m: Pyomo model
                from_bus: the transformer from bus number
                to_bus: the transformer to bus number
                ckt: the transformer circuit identifier

            Returns:
                Pyomo expression for the transformer flow

            """
            transformer = grid_data["transformers"][(from_bus, to_bus, ckt)]
            Ir = transformer.calc_real_current(m.Vr[from_bus], m.Vr[to_bus], m.Vi[from_bus], m.Vi[to_bus], from_bus)
            Ii = transformer.calc_imag_current(m.Vr[from_bus], m.Vr[to_bus], m.Vi[from_bus], m.Vi[to_bus], from_bus)
            I_mag = Ir ** 2 + Ii ** 2
            return m.I_transformer[(from_bus, to_bus, ckt)] == I_mag

        model.i_transformer_constraint = pe.Constraint(model.transformers_set, rule=transformer_current_constraint)

        return model

    def solve_model(self,
                    logger,
                    grid_data,
                    pyomo_options=None,
                    solver_options=None):
        """
        Execute the solving of the  scenario model
        Args:
            logger: A log object to document the solving status
            grid_data: the grid data for the current scenario being optimized
            solver_options: the options for the nonlinear optimization solver being used
            pyomo_options: pyomo specific solver options that provided for details about the solving process

        Returns:
            None, a summary of the results is saved internally
        """

        # check that the model exists
        if self._model is None:
            raise RuntimeError(
                "A OPF model has not been created. Please create the OPF model before continuing."
            )

        (solved, results) = _solve_opf_model(self._model, logger, pyomo_options=pyomo_options,
                                             solver_options=solver_options)

        self._solved = solved["Status"]
        self._results = results

        results_summary = self.summarize_solution(solved, grid_data)

        self._results_summary = results_summary

        logger.info(f"The total cost is {results_summary['Total Cost']:,.2f}")

    def summarize_solution(self,
                           solved: dict,
                           grid_data: dict) -> dict:
        """
        Summarize the results obtained by OPF in a dictionary
        Args:
            solved: provides the solved status and terminating condition
            grid_data: the grid data for the current scenario being optimized

        Returns:
            A summary of the results
        """

        # init model results dict
        results = {
            "Solved": None,
            "Objective": {"name": self._objective, "value": None}
        }

        # check if
        if not self._solved:
            results["Solved"] = solved
            return results

        model = self._model
        obj_value = pe.value(model.obj)

        # # === VMAG AND VMAG DEVIATION === # #
        V_mag = model.V_mag.extract_values()
        V_mag_init = {key: val.v_mag for key, val in grid_data["buses"].items()}
        V_ang_init = {key: val.v_ang for key, val in grid_data["buses"].items()}

        voltage_results = {
            key: {
                "Bus": key,
                "Vmag Initial": V_mag_init[key],
                "Vmag Final": val,
                "Vang Initial": V_ang_init[key],
                "Vang Final": degrees(atan2(model.Vi[key].value, model.Vr[key].value))
            } for key, val in V_mag.items()
        }

        # # === TRANSFORMER LINE FLOWS  === # #
        I_line = model.I_line.extract_values()
        I_line_init = {key: val.i_mag for key, val in grid_data["branches"].items()}

        # normalized I_line
        I_line_capacity = {key: val.ub for key, val in model.I_line.items()}

        I_line_results = {
            key: {
                "From Bus": key[0],
                "To Bus": key[1],
                "Ckt": key[2],
                "Initial": I_line_init[key],
                "Final": sqrt(abs(val)),
                "Capacity": I_line_capacity[key]
            } for key, val in I_line.items()
        }

        # # === TRANSFORMER CURRENT FLOWS === # #
        I_transformer = model.I_transformer.extract_values()
        I_transformer_init = {key: val.i_mag for key, val in grid_data["transformers"].items()}

        I_transformer_capacity = {key: val.ub for key, val in model.I_transformer.items()}

        I_transformer_results = {
            key: {
                "From Bus": key[0],
                "To Bus": key[1],
                "Ckt": key[2],
                "Initial": I_transformer_init[key],
                "Final": sqrt(abs(val)),
                "Capacity": I_transformer_capacity[key]
            } for key, val in I_transformer.items()
        }

        # # === GENERATORS - PG, QG, and DEVIATION Pg, Qg=== # #
        Pg = model.Pg.extract_values()
        Qg = model.Qg.extract_values()

        gen_results = {
            key: {
                "Bus": key[0],
                "ID": key[1],
                "P Initial": val.P.value,
                "P Final": Pg[key],
                "P Max": val.P_max,
                "P Min": val.P_min,
                "Q Initial": val.Q.value,
                "Q Final": Qg[key],
                "Q Max": val.Q_max,
                "Q Min": val.Q_min
            } for key, val in grid_data["generators"].items()
        }

        # add offline generators
        for key, ele in grid_data["offline generators"].items():
            gen_results[key] = {
                "Bus": key[0],
                "ID": key[1],
                "P Initial": ele.P.value,
                "P Final": ele.P.value,
                "P Max": ele.P_max,
                "P Min": ele.P_min,
                "Q Initial": ele.Q.value,
                "Q Final": ele.Q.value,
                "Q Max": ele.Q_max,
                "Q Min": ele.Q_min
            }

        # # === CALCULATE COSTS === # #
        cost_per_gen, total_cost = _cost(model, grid_data["offline generators"])

        # # === STORE RESULTS === # #
        results = {
            "Solved": solved,
            "Objective": {"name": self._objective, "value": obj_value},
            "Runtime": self._runtime,
            "Cost per Gen": cost_per_gen,
            "Total Cost": total_cost * 100,
            "Generators": gen_results,
            "Voltages": voltage_results,
            "Lines": I_line_results,
            "Transformers": I_transformer_results
        }

        return results

    def save_solution(self,
                      case_data_raw: dict,
                      filepaths: dict):
        """
        Save the OPF solution to a RAW file and the results summary as a compressed archive
        Args:
            case_data_raw: the original case data from a RAW file
            filepaths: a dictionary that tracks the filepaths of common directories
        Returns:
            None, the results are saved to a RAW file
        """
        model = self._model
        scenario = self._scenario
        results_summary = self._results_summary
        modified_case_data = update_grid_data(case_data_raw, model)
        solve_status = "solved" if self._solved else "failed"

        raw_file = f"{scenario}_OPF_{solve_status}_.RAW"
        raw_dir = f"{filepaths['case']}/OPF_{solve_status}"
        Path(raw_dir).mkdir(parents=True, exist_ok=True)
        path_to_raw = os.path.normpath(f"{raw_dir}/{raw_file}")
        modified_case_data.raw.write(path_to_raw)
        results_status = "FAILED_" if not self._solved else ""

        save_compressed_results(filepaths, f"{scenario}_OPF_{results_status}results", results_summary)


def _cost(model: pe.ConcreteModel,
          offline_generators: dict):
    cost_per_gen = {
        key: {
            "Bus": key[0],
            "ID": key[1],
            "Cost": model.a_cost[key] + (model.b_cost[key] * model.Pg[key].value) + (model.c_cost[key] * model.Pg[
                key].value) ** 2
        } for key in model.generators_set
    }

    for key, ele in offline_generators.items():
        cost_per_gen[key] = {
            "Bus": key[0],
            "ID": key[1],
            "Cost": 0,
        }

    total_cost = sum(cost_per_gen[key]["Cost"] for key in model.generators_set)
    return cost_per_gen, total_cost


def _solve_opf_model(model: pe.ConcreteModel,
                     logger: Log,
                     solver_name: str = "ipopt",
                     pyomo_options: dict = None,
                     solver_options: dict = None):
    """
    Solve the  model optimization
    Args:
        model: the pyomo optimization model of the uncertain scenario
        logger: A log object to document the solving status
        solver_name: the name of the nonlinear solver being used
        solver_options: the options for the nonlinear optimization solver being used
        pyomo_options: pyomo specific solver options that provided for details about the solving process

    Returns:
        solved: if the optimization was solved or not
        results: the results from the optimization

    """
    # check if there are any pyomo-specific solver options
    if pyomo_options is None:
        # possible options include warm_start=True, execute='filepath', tee=True
        pyomo_options = {"tee": False}

    # check if there are any solver-specific options; otherwise, use ipopt options shown below
    if solver_options is None:
        solver_options = {
            "OF_hessian_approximation": "exact",
            "max_iter": 10000,
            "OF_output_file": os.path.normpath(f"log/{model.name}_OPF_debug.log")

        }

    # create the solver
    opt = SolverFactory(solver_name)

    # solve the optimization
    results = opt.solve(model, options=solver_options, **pyomo_options)

    # access solver status and return the appropriate statement based on current status
    if (results.solver.status == SolverStatus.ok) and \
            (results.solver.termination_condition == TerminationCondition.optimal):
        # solution found
        logger.info("Feasible and optimal solution to {scenario} has been found.".format(scenario=model.name))
        solved = {"Status": True, "Terminating Condition": "Feasible and Optimal"}
    elif results.solver.termination_condition == TerminationCondition.infeasible:
        # scenario is infeasible
        logger.info("{scenario} is infeasible.".format(scenario=model.name))
        log_infeasible_constraints(model, logger=logger, log_expression=True, log_variables=True)
        log_infeasible_bounds(model, logger=logger)
        log_close_to_bounds(model, logger=logger)
        solved = {"Status": False, "Terminating Condition": "Infeasible"}
    elif results.solver.termination_condition == TerminationCondition.maxIterations:
        # exceeded max number of iterations
        logger.info("OPF could not finish solving {scenario}.".format(scenario=model.name))
        logger.info("Solving exceeded the maximum number of iterations.")
        solved = {"Status": False, "Terminating Condition": "Max Iterations Exceeded"}
    else:
        # some other error
        solved = {"Status": False, "Terminating Condition": "Unknown Solver Error"}
        logger.info("{scenario} is not solved due to some unknown error.".format(scenario=model.name))

    return solved, results
