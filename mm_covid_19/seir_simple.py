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
import matplotlib.pyplot as plt
import numpy as np

from . import data_population
from . import data_italy


class State(enum.IntEnum):
    S = 0  # Susceptible
    E = 1  # Exposed
    I = 2  # Infectious
    R = 3  # Removed


class Model():
    def __init__(self, device='auto'):
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)

        # Average basic reproduction number is around 2. We should divide this
        # by the average infection period of 3 to get the rate of infection
        # between cases in (I) and (S). Thus beta should be around 2/3 = 0.66.
        self.beta_min = 0.7
        self.beta_max = 0.7
        self.beta = torch.tensor(0.5 * (self.beta_min + self.beta_max),
                                 dtype=torch.float,
                                 requires_grad=True,
                                 device=self.device)

        # The average incubation period is around 6 days, which is the time
        # patients spend in state (E). We assume an exponential distribution,
        # and every day we move a beta fraction of cases in (E) to (I).
        # Thus sigma should be around 1/6 = 0.17.
        self.sigma_min = 0.2
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
        self.gamma_max = 0.2
        self.gamma = torch.tensor(0.5 * (self.gamma_min + self.gamma_max),
                                  dtype=torch.float,
                                  requires_grad=True,
                                  device=self.device)

        # The average mortality rate is around 2%. Thus mu fraction of the
        # removed cases are deceased.
        self.mu_min = 0.02
        self.mu_max = 0.02
        self.mu = torch.tensor(0.5 * (self.mu_min + self.mu_max),
                               dtype=torch.float,
                               requires_grad=True,
                               device=self.device)

        self.population = data_population.DataPopulation()
        self.population.load()

        self.italy = data_italy.DataItaly()
        self.italy.load()

        self.italy_population = torch.tensor(
            self.population.by_regions('Italy', self.italy.regions),
            dtype=torch.float,
            requires_grad=False,
            device=self.device)

        self.italy_cases = self.create_cases(
            'Italy',
            self.italy_population,
            100)
        # len(self.italy.dates))

        # optimization parameters
        self.parameters = {
            'beta': self.beta,
            'sigma': self.sigma,
            'gamma': self.gamma,
            'mu': self.mu,
            'italy': self.italy_cases,
        }

        # various losses will be put here
        self.losses = {}

    def print_parameter_sizes(self):
        print("parameter sizes:")
        for name, param in self.parameters.items():
            print("* " + name + ":", list(param.shape))

    def print_parameters(self):
        print("parameters:")
        for name, param in self.parameters.items():
            if list(param.shape) == []:
                print("* " + name + ":", param.cpu().item())

    def clamp_parameters(self):
        self.beta.data = self.beta.clamp(self.beta_min, self.beta_max)
        self.sigma.data = self.sigma.clamp(self.sigma_min, self.sigma_max)
        self.gamma.data = self.gamma.clamp(self.gamma_min, self.gamma_max)
        self.mu.data = self.mu.clamp(self.mu_min, self.mu_max)

        # TODO: check why 0.0 gives nans
        self.italy_cases.data = self.italy_cases.clamp(min=0.1)

    def create_cases(self, name, population, num_days):
        """Creates a tensor with entries describing the number of cases in each
        category. The returned shape is [num_regions, num_days, len(State)].
        Initially, all of the population are in (S)."""
        cases = torch.randn(
            [len(population), num_days, len(State)],
            dtype=torch.float,
            requires_grad=True,
            device=self.device)
        population = torch.reshape(population, [-1, 1])
        cases.data[:, :, State.S] = population
        return cases

    @staticmethod
    def rms_loss(tensor):
        return torch.sqrt(torch.mean(tensor ** 2.0))

    def susceptible_cases(self, cases):
        """Returns the total number of susceptible cases as a tensor of shape
        [num_regions, num_days]."""
        return cases[:, :, State.S]

    def exposed_cases(self, cases):
        """Returns the actively exposed cases as a tensor of shape
        [num_regions, num_days]."""
        return cases[:, :, State.E]

    def infectious_cases(self, cases):
        """Returns the actively infectious cases as a tensor of shape
        [num_regions, num_days]."""
        return cases[:, :, State.I]

    def removed_cases(self, cases):
        """Returns the total number of removed cases as a tensor of shape
        [num_regions, num_day]."""
        return cases[:, :, State.R]

    def resistant_cases(self, cases):
        """Returns the total number of resistant cases as a tensor of shape
        [num_regions, num_day]."""
        return cases[:, :, State.R] * (1.0 - self.mu)

    def deceased_cases(self, cases):
        """Returns the total number of deceased cases as a tensor of shape
        [num_regions, num_days]."""
        return cases[:, :, State.R] * self.mu

    def population_alive(self, cases):
        """Returns the population alive (not counting the deceased) as a tensor
        of shape [num_regions, num_days]."""
        return torch.sum(cases[:, :, State.S:State.R], 2, keepdim=False) \
            + cases[:, :, State.R] * (1.0 - self.mu)

    def delta_susceptible(self, cases):
        """Returns the daily change in the number of susceptible cases as a
        tensor of shape [num_regions, num_days - 1]."""
        return cases[:, 1:, State.S] - cases.data[:, :-1, State.S]

    def delta_exposed(self, cases):
        """Returns the daily change in the number of exposed cases as a
        tensor of shape [num_regions, num_days - 1]."""
        return cases[:, 1:, State.E] - cases.data[:, :-1, State.E]

    def delta_infectious(self, cases):
        """Returns the daily change in the number of infectious cases as a
        tensor of shape [num_regions, num_days - 1]."""
        return cases[:, 1:, State.I] - cases.data[:, :-1, State.I]

    def delta_removed(self, cases):
        """Returns the daily change in the number of removed cases as a
        tensor of shape [num_regions, num_days - 1]."""
        return cases[:, 1:, State.R] - cases.data[:, :-1, State.R]

    def susceptible_to_exposed(self, cases, population):
        """Returns the daily number of newly infected cases, those that move
        from (S) to (E), as a tensor of shape [num_regions, num_days - 1]."""
        return cases.data[:, :-1, State.S] * cases.data[:, :-1, State.I] / \
            population.reshape([-1, 1]) * self.beta

    def exposed_to_infectious(self, cases):
        """Returns the daily number of cases that move from (S) to (I), as a
        tensor of shape [num_regions, num_days - 1]."""
        return cases.data[:, :-1, State.E] * self.sigma

    def infectious_to_removed(self, cases):
        """Returns the daily number of cases that move from (I) to (R), as a
        tensor of shape [num_regions, num_days - 1]."""
        return cases.data[:, :-1, State.I] * self.gamma

    def add_loss(self, name, loss):
        assert name not in self.losses
        self.losses[name] = loss

    def add_evolution_losses(self, country, cases, population):
        """Calculates the model evolution losses."""

        s2c = self.susceptible_to_exposed(cases, population)
        self.add_loss(
            country + ' susceptible',
            self.rms_loss(-s2c - self.delta_susceptible(cases)))

        e2i = self.exposed_to_infectious(cases)
        self.add_loss(
            country + ' exposed',
            self.rms_loss(s2c - e2i - self.delta_exposed(cases)))

        i2r = self.infectious_to_removed(cases)
        self.add_loss(
            country + ' infectious',
            self.rms_loss(e2i - i2r - self.delta_infectious(cases)))

        self.add_loss(
            country + ' removed',
            self.rms_loss(i2r - self.delta_removed(cases)))

    def add_initial_italy(self):
        self.add_loss("initial exposed",
                      self.rms_loss(self.exposed_cases(self.italy_cases)[:, 0] - 10.0))

        self.add_loss("initial infectious",
                      self.rms_loss(self.infectious_cases(self.italy_cases)[:, 0] - 0.0))

        self.add_loss("initial removed",
                      self.rms_loss(self.removed_cases(self.italy_cases)[:, 0] - 0.0))

    def plot_cases(self, cases, population):
        _fig, ax1 = plt.subplots(1, 1)

        ax1.plot(
            np.sum(self.susceptible_cases(cases).data.numpy(), axis=0),
            label="susceptible")

        ax1.plot(
            np.sum(self.susceptible_to_exposed(
                cases, population).data.numpy(), axis=0),
            label="susceptible to exposed")

        ax1.plot(
            np.sum(self.exposed_cases(cases).data.numpy(), axis=0),
            label="exposed")

        ax1.plot(
            np.sum(self.exposed_to_infectious(cases).data.numpy(), axis=0),
            label="exposed to infectious")

        ax1.plot(
            np.sum(self.delta_exposed(cases).data.numpy(), axis=0),
            label="delta exposed")

        ax1.plot(
            np.sum(self.delta_infectious(cases).data.numpy(), axis=0),
            label="delta infectious")

        ax1.plot(
            np.sum(self.infectious_cases(cases).data.numpy(), axis=0),
            label="infectious")

        ax1.plot(
            np.sum(self.infectious_to_removed(cases).data.numpy(), axis=0),
            label="infectious to removed")

        ax1.plot(
            np.sum(self.removed_cases(cases).data.numpy(), axis=0),
            label="removed")

        ax1.legend()
        ax1.set_xlabel("days")
        ax1.set_ylabel("cases")

        plt.show()

    def print_losses(self, step=None):
        if step is None:
            print("losses:")
        else:
            print("step", step, "losses:")
        for name, loss in self.losses.items():
            assert list(loss.shape) == []
            print("* " + name + ":", loss.item())

    def optimize(self, steps, learning_rate=1.0):
        print("optimizing", steps, "steps with",
              learning_rate, "learning rate")
        optim = torch.optim.Adam(self.parameters.values(), lr=learning_rate)
        for step in range(steps):
            optim.zero_grad()

            self.losses.clear()
            self.add_evolution_losses(
                'italy', self.italy_cases, self.italy_population)
            self.add_initial_italy()

            total_loss = torch.zeros([], dtype=torch.float, device=self.device)
            for loss in self.losses.values():
                total_loss += loss

            total_loss.backward()
            optim.step()

            self.clamp_parameters()

            if step % 1000 == 0:
                self.print_losses(step)
                self.print_parameters()

            if step % 5000 == 0 or step == steps - 1:
                self.plot_cases(self.italy_cases, self.italy_population)

        print("*", steps, "loss:", loss.cpu().item())


def run(args=None):
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--device', default='auto', type=str,
                        help="sets the CUDA device")
    args = parser.parse_args(args)

    model = Model(device=args.device)
    model.print_parameter_sizes()
    model.optimize(50000, learning_rate=200)
    model.print_losses()


if __name__ == '__main__':
    run()
