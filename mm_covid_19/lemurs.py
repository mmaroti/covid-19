#!/usr/bin/env python
# Copyright (C) 2019, Miklos Maroti
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

"""
A pandas alternative, with multi dimensional tables, copied from
MarmotE DARPA SC2 project.
"""

from __future__ import print_function, division

import numpy as np


class Axis(object):
    """
    An object that maps (arbitrary) coordinate values to (zero based and
    incremented) indices.
    """

    def __init__(self, name, increment=10, data=None):
        assert increment >= 1
        self.name = name
        self.increment = increment

        self.data = [] if data is None else list(data)
        self.lookup = {}
        for i, d in enumerate(self.data):
            assert d not in self.lookup
            self.lookup[d] = i

    def __len__(self):
        return len(self.data)

    def __getitem__(self, coord):
        return self.lookup[coord]

    def __contains__(self, coord):
        return coord in self.lookup

    def add(self, coord):
        index = self.lookup.get(coord)
        if index is None:
            index = len(self.data)
            self.data.append(coord)
            self.lookup[coord] = index

    def __repr__(self):
        return "{} of size {}".format(self.name, len(self.data))


class Table(object):
    """
    Maintains a table of values of the same data type indexed by
    a list of axes.
    """

    def __init__(self, name, axes, dtype=float, data=None):
        self.name = name
        self.axes = axes
        if data is None:
            self.data = np.zeros(shape=[len(a) for a in axes], dtype=dtype)
        else:
            assert list(data.shape) == [len(a) for a in axes]
            self.data = data

    def get_indices(self, coords):
        """
        Returns the tuple of indices corresponding to the given coordinate
        values. The underlying data table is enlarged if necessary.
        """

        if not isinstance(coords, tuple):
            coords = (coords,)

        assert len(coords) == len(self.axes)
        indices = [c if c == slice(None) else self.axes[i][c]
                   for i, c in enumerate(coords)]

        for i, j in enumerate(indices):
            if j != slice(None) and self.data.shape[i] <= j:
                widths = [(0, 0)] * len(self.axes)
                widths[i] = (0, j - self.data.shape[i] +
                             self.axes[i].increment)
                self.data = np.pad(self.data, widths, mode='constant')

        return tuple(indices)

    def trim_data(self):
        """Trims the data table to the shape from the axes."""

        new_shape = tuple([len(a) for a in self.axes])
        if new_shape == self.data.shape:
            return

        # pad if necessary
        widths = None
        for i, j in enumerate(new_shape):
            if self.data.shape[i] < j:
                if widths is None:
                    widths = [(0, 0)] * len(self.axes)
                widths[i] = (0, j - self.data.shape[i])
        if widths is not None:
            self.data = np.pad(self.data, widths, mode='constant')

        # trim if necessary
        self.data = self.data[tuple([slice(n) for n in new_shape])].copy()

    def replace_axis(self, old_axis, new_axis):
        """
        Replaces the given axes with the new one. Values from the old
        tables are copied over to the new tables and missing data is filled
        with zeros.
        """

        for pos in range(len(self.axes)):
            if self.axes[pos] != old_axis:
                continue

            new_shape = list(self.data.shape)
            new_shape[pos] = len(new_axis)
            new_data = np.zeros(shape=new_shape, dtype=self.data.dtype)

            old_dim = self.data.shape[pos]
            old_index = [slice(n) for n in new_shape]
            new_index = list(old_index)

            for i, d in enumerate(new_axis.data):
                j = old_axis.lookup.get(d)
                if j is not None and j < old_dim:
                    old_index[pos] = j
                    new_index[pos] = i
                    new_data[tuple(new_index)] = self.data[tuple(old_index)]

            self.axes[pos] = new_axis
            self.data = new_data

    def __getitem__(self, coords):
        indices = self.get_indices(coords)
        return self.data[indices]

    def __setitem__(self, coords, value):
        indices = self.get_indices(coords)
        self.data[indices] = value

    def __repr__(self):
        return "{} of shape [{}] type {}".format(
            self.name,
            ', '.join([a.name for a in self.axes]),
            self.data.dtype)


class Frame(object):
    """A collection of axis and tables that are manipulated together."""

    def __init__(self):
        self.axes = {}
        self.tables = {}

    def __repr__(self):
        return "frame axes [{}] table [{}]".format(
            ', '.join(sorted(self.axes)),
            ', '.join(sorted(self.tables)))

    def info(self):
        def axis_info(axis):
            r = str(axis)
            if len(axis) <= 20:
                r += " = " + str(axis.data)
            return r

        def table_info(table):
            r = str(table)
            if table.data.ndim <= 1 and table.data.size <= 10:
                r += " = " + str(table.data)
            return r

        return "\n".join(
            [axis_info(self.axes[n]) for n in sorted(self.axes)] +
            [table_info(self.tables[n]) for n in sorted(self.tables)])

    def get_axis(self, name):
        return self.axes[name]

    def get_table(self, name):
        return self.tables[name]

    def __getitem__(self, name):
        return self.tables[name] if name in self.tables else self.axes[name]

    def __contains__(self, name):
        return name in self.tables or name in self.axes

    def add_axis(self, name, increment=10, data=None):
        assert name not in self.axes
        axis = Axis(name, increment=increment, data=data)
        self.axes[name] = axis

    def add_table(self, name, axis_names, dtype=float, data=None):
        assert name not in self.tables
        table = Table(
            name, [self.axes[n] for n in axis_names], dtype=dtype, data=data)
        self.tables[name] = table
        return table

    def trim_data(self):
        for s in self.tables.values():
            s.trim_data()

    def replace_axis(self, name, data):
        old_axis = self.axes[name]
        if old_axis.data == data:
            return

        new_axis = Axis(old_axis.name, data=data, increment=old_axis.increment)
        self.axes[name] = new_axis

        for s in self.tables.values():
            s.replace_axis(old_axis, new_axis)

    def extend(self, frame):
        for axis in frame.axes.values():
            assert axis.name not in self.axes
            self.axes[axis.name] = axis

        for table in frame.tables.values():
            assert table.name not in self.tables
            self.tables[table.name] = table


def dump(frame, file):
    frame.trim_data()
    parts = []

    for axis in frame.axes.values():
        parts.append(np.array(
            ['axis', axis.name, str(axis.increment)],
            dtype='U999'))
        parts.append(np.array(axis.data))

    for table in frame.tables.values():
        parts.append(np.array(
            ['table', table.name] + [a.name for a in table.axes],
            dtype='U999'))
        parts.append(table.data)

    np.savez_compressed(file, *parts)


def load(file):
    def getid(name):
        assert name[:4] == 'arr_'
        return int(name[4:])

    scrambled = np.load(file, allow_pickle=False)
    parts = [scrambled[n] for n in sorted(scrambled, key=getid)]

    frame = Frame()
    for idx in range(0, len(parts), 2):
        conf = parts[idx]
        data = parts[idx + 1]
        if conf[0] == 'axis':
            frame.add_axis(conf[1], increment=int(conf[2]), data=list(data))
        elif conf[0] == 'table':
            frame.add_table(conf[1], conf[2:], dtype=data.dtype, data=data)
        else:
            print("Unexpected part", conf)

    return frame


def run(args=None):
    import argparse
    import logging
    import sys

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("filenames", metavar='FILE', nargs='*',
                        help="npz files to query")
    parser.add_argument("--print",
                        help="print this data from the frames")

    args = parser.parse_args(args)
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    np.set_printoptions(edgeitems=20)

    for filename in args.filenames:
        logging.info("Reading %s", filename)
        frame = load(filename)
        print(frame.info())
        if args.print is not None:
            table = frame[args.print]
            if table is not None:
                print(table.data)
            else:
                logging.info("Unknown name %s", args.data)
                args.data = None


if __name__ == "__main__":
    run()
