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
    and (D)eceised according to the development of the virus where
    * Susceptible: not infectious, tests negative, no symptoms
    * Exposed: not infectious, tests positive, no symptoms
    * Infectious: infectious, tests positive, symptoms
    * Recovered: not infectious, tests negative, no symptoms
    * Deceised: not infectious, tests positive
    """
    S = 0
    E = 1
    I = 2
    R = 3
    D = 4


class Confirmed(enum.IntEnum):
    """
    Each patient is either (N)ot confirmed, or (C)onfirmed where
    * Not confirmed: either not tested at all or all of the tests are negative
    * Confirmed: at least one of the tests are positive (may have recovered)
    """
    N = 0
    C = 1


class seirTest():
    """
    All (S) are (N). An (S,N)->(E,N) infection happens, when an (S,N) patient
    meets an (I,N) patient daily with beta probability. All (I,P) patients are
    assumed to be in quarantine, so they are not infecting.

    (E)->(I) transition happens with alpha probability with exponential
    distribution, so the (E) incubation period is 1 / gamma. The (I)->(R)
    recovery happens with gamma probaility, while (I)->(D) happens with delta
    probability with exponential distributions. Thus the (I) infectios period
    is 1 / (gamma + delta).

    The (E,N)->(E,C), (I,N)->(I,C) and (D,N)->(D,C) transitions happens when
    an actual test is made. We assume that all deaths are confirmed, so there
    are only three terminal states: (R,N), (R,C), (D,C).
    """

    def __init__(self, device='auto'):
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)

        self.beta_min = 0.1
        self.beta_max = 0.5
        self.beta = torch.tensor(0.5 * (self.beta_min + self.beta_max),
                                 dtype=torch.float,
                                 requires_grad=True,
                                 device=self.device)

        self.gamma_min = 0.1
        self.gamma_max = 0.5
        self.gamma = torch.tensor(0.5 * (self.gamma_min + self.gamma_max),
                                  dtype=torch.float,
                                  requires_grad=True,
                                  device=self.device)

        self.delta_min = 0.1
        self.delta_max = 0.5
        self.delta = torch.tensor(0.5 * (self.delta_min + self.delta_max),
                                  dtype=torch.float,
                                  requires_grad=True,
                                  device=self.device)

        # optimization parameters, case tables will be added
        self.parameters = {
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
        avgpop = 2 + int(avgpop / (len(State) * len(Confirmed)))
        table = torch.randint(0, avgpop, [num_regions, num_days, len(State), len(Confirmed)],
                              dtype=torch.float, requires_grad=True, device=self.device)
        assert name not in self.parameters
        self.parameters[name] = table
        return table

    @staticmethod
    def rms_loss(tensor):
        return torch.sqrt(torch.mean(tensor ** 2.0))

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
        """Enforces that susceptible patiens are never confirmed."""
        assert table.is_leaf

        def loss_func():
            simulated = table[:, :, State.S, Confirmed.C]
            return self.rms_loss(simulated)

        name = name + ' suspectible not confirmed'
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

    test = seirTest(device=args.device)

    if args.italy:
        test.add_italy()
        test.print_parameters()
        test.optimize(10000)
        test.print_losses()


if __name__ == '__main__':
    run()
