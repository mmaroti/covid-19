#!/usr/bin/env python3
# Copyright (C) 2020, Miklos Maroti
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import enum
import torch

from . import data_population
from . import data_italy


class State(enum.IntEnum):
    """
    Each patient is in (S)usceptible, (E)xposed, (I)nfectious, (R)ecovered
    or (D)eceased state at beginning of each day. The state is updated
    at the end of the day.
    * Susceptible: not infectious, tests negative, no symptoms.
    * Exposed: not infectious, tests positive, no symptoms.
    * Infectious: infectious, tests positive, has symptoms.
    * Recovered: not infectious, tests negative, no symptoms.
    * Deceased: not infectious, tests positive.
    """
    S = 0
    E = 1
    I = 2
    R = 3
    D = 4


class Test(enum.IntEnum):
    """
    Each patient is (U)nknown, (T)ested or (C)onfirmed at the beginning
    of each day, and the test results are revealed at the end of the day.
    * Unknown: all previous tests (if any) are negative, not tested that
      day, not in quarantine (can infect).
    * Tested: all previous tests (if any) are negative, is tested that day,
      not in quarantine (can infect).
    * Confirmed: at least one positive prior test, not tested that day,
      not infectious (either in quarantine or recovered).
    """
    U = 0
    T = 1
    C = 2


class SeirTest():
    """
    Static constraints:
        SC := 0
        all other fields sum up to population

    Transitions:
        SU' + ST' := SU + ST - beta * (SU + ST) * (IU + IT) / population.
        EU' + ET' := EU - alpha * EU + beta * (SU + ST) * (IU + IT) / population.
        EC' := EC + ET - alpha * (EC + ET)
        IU' + IT' := IU - (gamma + delta) * IU + alpha * EU
        IC' := IC + IT - (gamma + delta) * (IC + IT) + alpha * (EC + ET)
        RU' + RT' := RU + RT + gamma * IU
        RC' := RC + gamma * (IC + IT)
        DU' + DT' := DU + delta * IU
        DC' := DC + DT + delta * (IC + IT)
    """

    def __init__(self, device='auto'):
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)

        # average reproduction number is 2, divided by infection period
        # of 3 is around 0.7
        self.alpha_min = 0.4
        self.alpha_max = 1.0
        self.alpha = torch.tensor(0.5 * (self.alpha_min + self.alpha_max),
                                  dtype=torch.float,
                                  requires_grad=True,
                                  device=self.device)

        # average incubation period of 6 days, beta is around 1/6 = 0.17
        self.beta_min = 0.1
        self.beta_max = 0.3
        self.beta = torch.tensor(0.5 * (self.beta_min + self.beta_max),
                                 dtype=torch.float,
                                 requires_grad=True,
                                 device=self.device)

        # average infectious period is 3 days, 75% of that is recovery,
        # gamma is around 1/3 * 0.75 = 0.25
        self.gamma_min = 0.1
        self.gamma_max = 0.4
        self.gamma = torch.tensor(0.5 * (self.gamma_min + self.gamma_max),
                                  dtype=torch.float,
                                  requires_grad=True,
                                  device=self.device)

        # average infectious period is 3 days, 25% of that is death,
        # delta is around 1/3 * 0.25 = 0.08
        self.delta_min = 0.0
        self.delta_max = 0.2
        self.delta = torch.tensor(0.5 * (self.delta_min + self.delta_max),
                                  dtype=torch.float,
                                  requires_grad=True,
                                  device=self.device)

        # optimization parameters, case tables will be added
        self.parameters = {
            'alpha': self.alpha,
            'beta': self.beta,
            'gamma': self.gamma,
            'delta': self.delta,
        }

        self.loss_functions = {}

    def print_parameters(self):
        print("parameter sizes:")
        for name, param in self.parameters.items():
            print("* " + name + ":", list(param.shape))

    def print_losses(self):
        print("losses:")
        for name, func in self.loss_functions.items():
            loss = func().cpu()
            assert list(loss.shape) == []
            print("* " + name + ":", loss.item())

    def optimize(self, steps, learning_rate=200):
        print("optimizing", steps, "steps with", learning_rate, "learning rate")
        optim = torch.optim.Adam(self.parameters.values(), lr=learning_rate)
        for step in range(steps):
            optim.zero_grad()

            loss = torch.zeros([], dtype=torch.float, device=self.device)
            for func in self.loss_functions.values():
                loss = loss + func()

            if step % 1000 == 0:
                print("*", step, "loss:", loss.cpu().item())

            loss.backward()
            optim.step()

        print("*", steps, "loss:", loss.cpu().item())

    def add_case_table(self, name, num_regions, num_days, avgpop=100000):
        """Creates a tensor with optimizable variables describing one case study.
        The returned shape is [num_regions, num_days, len(State), len(Confirmed)]"""
        avgpop = 2 + int(avgpop / (len(State) * len(Test)))
        table = torch.randint(0, avgpop, [num_regions, num_days, len(State), len(Test)],
                              dtype=torch.float, requires_grad=True, device=self.device)
        assert name not in self.parameters
        self.parameters[name] = table
        return table

    def is_case_table(self, table):
        return len(table.shape) == 4 and table.shape[2] == len(State) \
            and table.shape[3] == len(Test)

    @staticmethod
    def _rms_loss(tensor):
        return torch.sqrt(torch.mean(tensor ** 2.0))

    def population(self, table):
        """Returns a tensor of shape [num_regions, num_days]."""
        assert self.is_case_table(table)
        return torch.sum(table, [2, 3], keepdim=False)

    def active_cases(self, table):
        """Returns a tensor of shape [num_regions, num_days]."""
        assert self.is_case_table(table)
        return table[:, :, State.E, Test.C] + table[:, :, State.I, Test.C]

    def closed_recovered(self, table):
        """Returns a tensor of shape [num_regions, num_days]."""
        assert self.is_case_table(table)
        return table[:, :, State.R, Test.C]

    def closed_deaths(self, table):
        """Returns a tensor of shape [num_regions, num_days]."""
        assert self.is_case_table(table)
        return table[:, :, State.D, Test.C]

    def closed_cases(self, table):
        """Returns a tensor of shape [num_regions, num_days]."""
        assert self.is_case_table(table)
        return table[:, :, State.R, Test.C] + table[:, :, State.D, Test.C]

    def new_positive_tests(self, table):
        """Returns a tensor of shape [num_regions, num_days]."""
        assert self.is_case_table(table)
        return table[:, :, State.E, Test.T] + table[:, :, State.I, Test.T] \
            + table[:, :, State.D, Test.T]

    def new_negative_tests(self, table):
        """Returns a tensor of shape [num_regions, num_days]."""
        assert self.is_case_table(table)
        return table[:, :, State.S, Test.T] + table[:, :, State.R, Test.T]

    def new_tests(self, table):
        """Returns a tensor of shape [num_regions, num_days]."""
        assert self.is_case_table(table)
        return torch.sum(table[:, :, :, Test.T], axis=2, keepdim=False)

    def enforce_constant_population(self, name, table, population):
        """Enforces that the population stays constant and equal to the give
        population data. The shape of the population must be num_regions."""
        assert table.is_leaf
        population = torch.tensor(
            population,
            dtype=torch.float32,
            requires_grad=False,
            device=self.device)
        assert len(population.shape) == 1
        assert table.shape[0] == population.shape[0]

        def loss_func():
            target = torch.reshape(population, [-1, 1])
            simulated = torch.sum(table, [2, 3], keepdim=False)
            return self.rms_loss(simulated - target)

        name = name + ' constant population'
        assert name not in self.loss_functions
        self.loss_functions[name] = loss_func

    def enforce_susceptible_not_confirmed(self, name, table):
        """Enforces that susceptible patients are never confirmed."""
        assert table.is_leaf

        def loss_func():
            simulated = table[:, :, State.S, Test.C]
            return self.rms_loss(simulated)

        name = name + ' susceptible not confirmed'
        assert name not in self.loss_functions
        self.loss_functions[name] = loss_func

    def add_italy(self):
        population = data_population.DataPopulation()
        population.load()

        italy = data_italy.DataItaly()
        italy.load()

        table = self.add_case_table('Italy', len(italy.regions), len(italy.dates))
        self.enforce_constant_population(
            'Italy', table, population.by_regions('Italy', italy.regions))
        self.enforce_susceptible_not_confirmed('Italy', table)


def run(args=None):
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--device', default='auto', type=str,
                        help="sets the CUDA device")
    parser.add_argument('--italy', action='store_true',
                        help="use the Italy dataset")
    args = parser.parse_args(args)

    test = SeirTest(device=args.device)

    if args.italy:
        test.add_italy()
        test.print_parameters()
        test.optimize(10000)
        test.print_losses()


if __name__ == '__main__':
    run()
