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

import argparse
import csv
import os
import numpy as np


class DataItaly():
    def __init__(self, data_path=None):
        self.data_path = data_path
        if not self.data_path:
            self.data_path = os.path.abspath(os.path.join(
                __file__, '..', '..', 'data', 'italy'))
            if not os.path.exists(self.data_path):
                self.data_path = None

        if not self.data_path:
            raise ValueError(
                'https://github.com/pcm-dpc/COVID-19 data path not found')

        self.file_name = os.path.join(
            self.data_path, 'dati-regioni', 'dpc-covid19-ita-regioni.csv')
        if not os.path.exists(self.file_name):
            raise ValueError('dpc-covid19-ita-regioni.csv data file not found')

        self.regions = []
        self.dates = []
        self.severe_cases = np.array([0, 0])

    def load(self):
        with open(self.file_name, newline='') as file:
            rows = csv.reader(file, dialect='excel')

            count = 0
            for row in rows:
                if count == 0:
                    if row[0] != 'data':
                        raise ValueError('unexpected CSV fields')

                print(row)
                count += 1
                if count >= 2:
                    break


def run(args=None):
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data-path', default=None, metavar='DIR', type=str,
                        help="path of the downloaded "
                        "https://github.com/pcm-dpc/COVID-19.git repository")
    args = parser.parse_args(args)

    data = DataItaly(args.data_path)
    data.load()


if __name__ == '__main__':
    run()
