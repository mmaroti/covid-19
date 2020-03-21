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

from . import data_italy


class State(enum.Enum):
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


class Tested(enum.Enum):
    """
    Each patient is either tested (N)egative or (P)ositive where
    * Negative: either not tested at all or all of the tests are negative
    * Positive: at least one of the tests are positive (may have recovered)
    """
    N = 0
    P = 1


class SeirdTest():
    """
    All (S) are (N). An (S,N)->(E,N) infection happens, when an (S,N) patient
    meets an (I,N) patient. All (I,P) patients are assumed to be in quarantine,
    so they are not infecting.

    (E)->(I) transition happens with alpha probability with exponential
    distribution, so the (E) incubation period is 1 / gamma. The (I)->(R)
    recovery happens with gamma probaility, while (I)->(D) happens with delta
    probability with exponential distributions. Thus the (I) infectios period
    is 1 / (gamma + delta).

    The (E,N)->(E,P), (I,N)->(I,P) and (D,N)->(D,P) transitions happens when
    an actual test is made. We assume that all deaths are confirmed, so there
    are only three terminal states: (R,N), (R,P), (D,P)
    """

    def __init__(self, device='auto'):
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)

        # generic model
        self.beta = torch.tensor([1.0], dtype=torch.float,
                                 requires_grad=True, device=self.device)
        self.gamma = torch.tensor([1.0], dtype=torch.float,
                                  requires_grad=True, device=self.device)
        self.delta = torch.tensor([1.0], dtype=torch.float,
                                  requires_grad=True, device=self.device)

        # optimization parameters, case tables will be added
        self.parameters = {
            'beta': self.beta,
            'gamma': self.gamma,
            'delta': self.delta,
        }

    def info(self):
        print("parameters", self.parameters.keys())

    def add_case_table(self, name, days):
        table = torch.zeros([len(State), len(Tested), days], dtype=torch.float,
                            requires_grad=True, device=self.device)
        assert name not in self.parameters
        self.parameters[name] = table
        return table

    def add_italy(self):
        italy = data_italy.DataItaly()
        italy.load()

        for idx in range(len(italy.regions)):
            table = self.add_case_table("Italy-" + italy.regions[idx],
                                        len(italy.dates))


def run(args=None):
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--device', default='auto', type=str,
                        help="sets the CUDA device")
    parser.add_argument('--italy', action='store_true',
                        help="use the Italy dataset")
    args = parser.parse_args(args)

    test = SeirdTest(device=args.device)
    if args.italy:
        test.add_italy()
    test.info()


if __name__ == '__main__':
    run()
