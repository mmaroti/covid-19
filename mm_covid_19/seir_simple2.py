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
import math
import matplotlib.pyplot as plt
import numpy as np
import torch


class TensorModel():
    def __init__(self, device, shape, num_days):
        self.device = device
        self.tensor = torch.randn(
            list(shape) + [int(num_days)],
            dtype=torch.float,
            requires_grad=True,
            device=device)

    def calc_table(self):
        return self.tensor * 1e7

    @property
    def params(self):
        return [self.tensor]


class NetworkModel():
    def __init__(self, device, shape, num_days):
        self.device = device
        self.num_funs = np.prod(shape)
        self.num_days = int(num_days)
        self.shape = list(shape) + [num_days]
        self.model = torch.nn.Sequential(
            torch.nn.Linear(1, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, self.num_funs),
        )
        self.days = -0.5 + 1.0 / self.num_days * torch.arange(
            0, self.num_days,
            dtype=torch.float,
            device=device).reshape([-1, 1])

    def calc_table(self):
        return self.model(self.days).T.reshape(self.shape)

    @property
    def params(self):
        return list(self.model.parameters())


class FourierModel():
    def __init__(self, device, shape, num_days):
        self.device = device
        self.num_funs = np.prod(shape)
        self.num_days = int(num_days)
        fft_size = 2 ** int(math.ceil(math.log2(num_days * 1.2)))
        self.shape = list(shape) + [num_days]
        self.tensor = torch.randn(
            [self.num_funs, fft_size, 2],
            dtype=torch.float,
            requires_grad=True,
            device=device)

    def calc_table(self):
        real = torch.fft(self.tensor, signal_ndim=1)[:, :self.num_days, 0]
        return real.reshape(self.shape) * 1e7

    @property
    def params(self):
        return [self.tensor]


class State(enum.IntEnum):
    S = 0  # Susceptible
    E = 1  # Exposed
    I = 2  # Infectious
    R = 3  # Removed


class Optimizer():
    def __init__(self, model, population):
        self.device = model.device
        self.model = model
        self.population = population
        self.num_days = model.calc_table().shape[-1]

        # Average basic reproduction number is around 2. We should divide this
        # by the average infection period of 3 to get the rate of infection
        # between cases in (I) and (S). Thus beta should be around 2/3 = 0.66.
        self.beta_min = 0.6
        self.beta_max = 0.7
        self.beta = torch.tensor(0.5 * (self.beta_min + self.beta_max),
                                 dtype=torch.float,
                                 requires_grad=True,
                                 device=self.device)

        # The average incubation period is around 6 days, which is the time
        # patients spend in state (E). We assume an exponential distribution,
        # and every day we move a beta fraction of cases in (E) to (I).
        # Thus sigma should be around 1/6 = 0.17.
        self.sigma_min = 0.1
        self.sigma_max = 0.2
        self.sigma = torch.tensor(0.5 * (self.sigma_min + self.sigma_max),
                                  dtype=torch.float,
                                  requires_grad=True,
                                  device=self.device)

        # The average infectious period is around 3 days, which is the time
        # patients spend in state (I). We assume an exponential distribution,
        # and every day we move a gamma fraction of cases in (I) to (R).
        # Thus gamma should be around 1/3 = 0.33
        self.gamma_min = 0.2
        self.gamma_max = 0.3
        self.gamma = torch.tensor(0.5 * (self.gamma_min + self.gamma_max),
                                  dtype=torch.float,
                                  requires_grad=True,
                                  device=self.device)

        self.losses = {}

    def plot_table(self, table):
        table = table.detach().cpu().numpy()
        _fig, (ax1, ax2) = plt.subplots(1, 2)

        ax1.stackplot(
            range(self.num_days),
            table,
            labels=["Susceptible", "Exposed", "Infectious", "Removed"])
        ax1.legend()

        ax2.plot(table[State.S, :], label="Susceptible")
        ax2.plot(table[State.E, :], label="Exposed")
        ax2.plot(table[State.I, :], label="Infectious")
        ax2.plot(table[State.R, :], label="Removed")
        ax2.legend()

        plt.show()

    def add_population_losses(self, table):
        self.losses["positivity"] = torch.clamp_max(table, 0)
        self.losses["population"] = (
            table.sum(dim=0, keepdim=False) - self.population)

    def add_limit_losses(self):
        self.losses["beta"] = 1e6 * (
            self.beta - self.beta.clamp(self.beta_min, self.beta_max))
        self.losses["sigma"] = 1e6 * (
            self.sigma - self.sigma.clamp(self.sigma_min, self.sigma_max))
        self.losses["gamma"] = 1e6 * (
            self.gamma - self.gamma.clamp(self.gamma_min, self.gamma_max))

    def add_evolution_losses(self, table):
        ds = table[State.S, 1:] - table[State.S, :-1]
        de = table[State.E, 1:] - table[State.E, :-1]
        di = table[State.I, 1:] - table[State.I, :-1]
        dr = table[State.R, 1:] - table[State.R, :-1]

        s2e = table[State.S, :-1] * table[State.I, :-1] * \
            (self.beta / self.population)
        e2r = table[State.E, :-1] * self.sigma
        i2r = table[State.I, :-1] * self.gamma

        self.losses["delta susceptible"] = ds + s2e
        self.losses["delta exposed"] = de + e2r - s2e
        self.losses["delta infectious"] = di + i2r - e2r
        self.losses["delta removed"] = dr - i2r

    def add_initial_losses(self, table):
        self.losses["initial susceptible"] = (
            self.population - 1000 - table[State.S, 0])
        self.losses["initial exposed"] = 1000 - table[State.E, 0]
        self.losses["initial infected"] = 0 - table[State.I, 0]
        self.losses["initial removed"] = 0 - table[State.R, 0]

    def calc_losses(self, table):
        self.losses.clear()
        self.add_population_losses(table)
        self.add_limit_losses()
        self.add_evolution_losses(table)
        self.add_initial_losses(table)
        return self.losses

    def optimize(self, steps=200000, learning_rate=0.00005):
        print("optimizing", steps, "steps with",
              learning_rate, "learning rate")
        optim = torch.optim.Adam(
            self.model.params + [self.beta, self.sigma, self.gamma],
            lr=learning_rate,
            amsgrad=False)
        for step in range(steps):
            optim.zero_grad()

            table = self.model.calc_table()
            losses = self.calc_losses(table)

            total_loss = 0.0
            for loss in losses.values():
                total_loss += torch.mean(loss ** 2.0)

            total_loss.backward()
            optim.step()

            if step % 1000 == 0 or step == steps - 1:
                print("step", step, "losses:")
                for name, loss in losses.items():
                    loss = torch.mean(loss ** 2.0).item()
                    print("*", name, math.sqrt(loss))
                print("* total loss", math.sqrt(total_loss.item()))
                print("parameters:")
                print("* beta", self.beta.item())
                print("* sigma", self.sigma.item())
                print("* gamma", self.gamma.item())

            if step % 20000 == 0 or step == steps - 1:
                self.plot_table(table)


def run(args=None):
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--device', default='auto', type=str,
                        help="sets the CUDA device")
    parser.add_argument('--model', default='tensor',
                        choices=['tensor', 'network', 'fourier'],
                        help="sets the model type")
    parser.add_argument('--days', default=100, type=int,
                        help="sets the number of days in the simulation")
    args = parser.parse_args(args)

    device = args.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    if args.model == 'network':
        model = NetworkModel(device, [len(State)], args.days)
    elif args.model == 'fourier':
        model = FourierModel(device, [len(State)], args.days)
    else:
        model = TensorModel(device, [len(State)], args.days)

    optimizer = Optimizer(model, 10e6)
    optimizer.optimize()


if __name__ == '__main__':
    run()
