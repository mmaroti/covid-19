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

import csv
import os
import numpy as np


class DataPopulation():
    """A loader class that presents population data."""

    def __init__(self, file_name=None):

        self.file_name = file_name
        if not self.file_name:
            self.file_name = os.path.abspath(os.path.join(
                __file__, '..', '..', 'data', 'population.csv'))
            if not os.path.exists(self.file_name):
                raise ValueError('population.csv file not found')

        self.country_map = {}

    def load(self):
        """Actually do the import operation."""

        with open(self.file_name, newline='') as file:
            rows = csv.reader(file, dialect='excel')

            count = 0
            for row in rows:
                count += 1
                if count == 1:
                    if len(row) < 4 or row[0] != 'country' \
                            or row[1] != 'region' \
                            or row[4] != 'population':
                        raise ValueError('unexpected CSV header')
                    continue

                if row[0] not in self.country_map:
                    self.country_map[row[0]] = {}
                region_map = self.country_map[row[0]]

                assert row[1] not in region_map
                region_map[row[1]] = int(row[4])

    def info(self):
        print("countries:", len(self.countries),
              "regions:", sum([len(v) for v in self.country_map.values()]))

    @property
    def countries(self):
        """Returns the list of know countries."""
        return list(self.country_map.keys())

    def regions(self, country):
        """Returns the list of regions of the given country."""
        assert country in self.country_map
        return list(self.country_map[country].keys())

    def population(self, country, region=None):
        """Returns the population of the given country. If the region is
        None, then the all population is returned. If the region is a
        string, then the given population is returned. If the region is
        a list of strings, then a numpy array of populations is returned."""
        assert country in self.country_map
        if region is None:
            return sum(p for p in self.country_map[country].values())
        elif isinstance(region, str):
            assert region in self.country_map[country]
            return self.country_map[country][region]
        else:
            assert isinstance(region, list)
            return np.array([self.country_map[country][r] for r in region],
                            dtype=int)


def run(args=None):
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--file-name', default=None, metavar='FILE', type=str,
                        help="path to population.csv")
    args = parser.parse_args(args)

    population = DataPopulation(args.file_name)
    population.load()
    population.info()


if __name__ == '__main__':
    run()
