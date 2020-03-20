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
from datetime import datetime
import os
import numpy as np

from . import lemurs


class DataItaly(lemurs.Frame):
    def __init__(self, data_path=None):
        super(DataItaly, self).__init__()

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

        self.add_axis('region', increment=1)
        self.add_axis('date')

        # active and positively tested cases
        self.add_table('active_severe', ['region', 'date'], dtype=int)
        self.add_table('active_critical', ['region', 'date'], dtype=int)
        self.add_table('active_home_conf', ['region', 'date'], dtype=int)

        # closed and cumulative, positively tested cases
        self.add_table('closed_recovered', ['region', 'date'], dtype=int)
        self.add_table('closed_deaths', ['region', 'date'], dtype=int)

        # cumulative total tests performed (positive and negative)
        self.add_table('total_tests', ['region', 'date'], dtype=int)

    def load(self):
        with open(self.file_name, newline='') as file:
            rows = csv.reader(file, dialect='excel')

            count = 0
            for row in rows:
                count += 1
                if count == 1:
                    if len(row) != 16 or row[0] != 'data' \
                            or row[3] != 'denominazione_regione' \
                            or row[6] != 'ricoverati_con_sintomi' \
                            or row[7] != 'terapia_intensiva' \
                            or row[9] != 'isolamento_domiciliare' \
                            or row[12] != 'dimessi_guariti' \
                            or row[13] != 'deceduti' \
                            or row[15] != 'tamponi':
                        raise ValueError('unexpected CSV header')
                    continue

                # we keep only the date, ignore time
                date = datetime.strptime(
                    row[0], "%Y-%m-%d %H:%M:%S").date()
                self['date'].add(date)

                region = row[3]
                self['region'].add(region)

                # entries
                self['active_home_conf'][region, date] = row[9]
                self['active_severe'][region, date] = row[6]
                self['active_critical'][region, date] = row[7]
                self['closed_recovered'][region, date] = row[12]
                self['closed_deaths'][region, date] = row[13]
                self['total_tests'][region, date] = row[14]

        self.trim_data()

    @property
    def dates(self):
        """Returns the list of dates used for all tables."""
        return self['date'].data

    @property
    def regions(self):
        """Returns the list of regions used for all tables."""
        return self['region'].data

    @property
    def active_home_conf(self):
        """Returns the numpy array of confirmed active patients in home
        confinement at the given region and time."""
        return self['active_critical'].data

    @property
    def active_severe(self):
        """Returns the numpy array of confirmed active patients with severe conditions in a hospital at the given region and time."""
        return self['active_severe'].data

    @property
    def active_critical(self):
        """Returns the numpy array of confirmed active patients with critical conditions in a hospital at the given region and time."""
        return self['active_critical'].data

    @property
    def active_cases(self):
        """Returns the numpy array of confirmed active cases at the given region and time."""
        return self.active_home_conf + self.active_severe + self.active_critical

    @property
    def closed_recovered(self):
        """Returns the numpy array of cumulative confirmed and recovered patients at the given region and time."""
        return self['active_critical'].data

    @property
    def closed_deaths(self):
        """Returns the numpy array of cumulative confirmed and deceased patients at the given region and time."""
        return self['active_critical'].data

    @property
    def closed_cases(self):
        """Returns the numpy array of cumulative confirmed closed cases at the given region and time."""
        return self.closed_recovered + self.closed_deaths


def run(args=None):
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data-path', default=None, metavar='DIR', type=str,
                        help="path of the downloaded "
                        "https://github.com/pcm-dpc/COVID-19.git repository")
    args = parser.parse_args(args)

    italy = DataItaly(args.data_path)
    italy.load()
    print(italy.info())
    print("Start date {}, end date {}".format(italy.dates[0], italy.dates[-1]))


if __name__ == '__main__':
    run()
