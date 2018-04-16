#!/usr/bin/env python3
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from itertools import count
from collections import defaultdict as ddict
import numpy as np
import torch as th


def generate_pairs(package_file, sep='.'): 
    mapping = ddict(set) #map from higher order element to the all direct children in the hierarchy
    all_names = set()
    duplicate = set()

    with open(package_file, 'r') as f:
        for line in f:
            package_names = line.strip().split(sep)
            if len(package_names) < 2:
                continue
            package_names = ['ROOT'] + package_names  
            for i in range(len(package_names)-1):
                high, low = package_names[i], package_names[i+1]
                mapping[high].add(low)

    with open('./package/all_package_pairs.tsv', 'w') as fout:
        for k, v_set in mapping.items():
            for v in v_set:
                old_len = len(all_names)
                all_names.add(v)
                new_len = len(all_names)
                if old_len == new_len: #duplicate element
                    duplicate.add(v)
                fout.write(str(v) + '\t' + str(k) + '\n') #more specific package comes first

    with open('./package/duplicate_packages.tsv', 'w') as fdup:
        for i in duplicate:
            fdup.write(str(i) + '\n')


def parse_line(line, length=2, sep='\t'):
    #each line is either (head, tail) or (head, tail, weight). Return tuple of (head, tail, weight)
    d = line.strip().split(sep)
    if len(d) == length:
        w = 1
    elif len(d) == length + 1:
        w = int(d[-1])
        d = d[:-1]
    else:
        raise RuntimeError('Malformed input %s' % line.strip())
    return tuple(d) + (w,)


def iter_line(fname, fparse, length=2, comment='#'):
    with open(fname, 'r') as fin:
        for line in fin:
            if line[0] == comment:
                continue
            tpl = fparse(line, length=length)
            if tpl is not None:
                yield tpl


def intmap_to_list(d):
    arr = [None for _ in range(len(d))]
    for v, i in d.items():
        arr[i] = v
    assert not any(x is None for x in arr)
    return arr


def slurp(fin, fparse=parse_line, symmetrize=False):
    ecount = count()
    enames = ddict(ecount.__next__)

    subs = []
    for i, j, w in iter_line(fin, fparse, length=2):
        print(i, j, w)
        if i == j:
            continue
        subs.append((enames[i], enames[j], w))
        if symmetrize:
            subs.append((enames[j], enames[i], w))
    idx = th.from_numpy(np.array(subs, dtype=np.int)) #array of (i, j, w)

    # freeze defaultdicts after training data and convert to arrays
    objects = intmap_to_list(dict(enames)) #list of all elements
    print('slurp: objects=%d, edges=%d' % (len(objects), len(idx)))
    return idx, objects


if __name__ == '__main__':
    #slurp('test.tsv')
    generate_pairs('./package/functions')