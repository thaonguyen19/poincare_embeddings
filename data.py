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


def generate_train_val(package_file_sorted, sep='.'):
    val_candidates = []
    with open(package_file_sorted, 'r') as f:
        cnt = 0
        prev_line = None
        for line in f:
            if len(line.strip().split(sep)) > 1: #not root
                if prev_line is not None and prev_line in line: #not leaf
                    val_candidates.append(cnt-1)

            prev_line = line.strip()
            cnt += 1

    n_val = int(0.2 * cnt)
    val_lines = list(np.random.choice(val_candidates, n_val))
    with open('./package/functions_train', 'w') as train_f:
        with open('./package/functions_val', 'w') as val_f:
            with open(package_file_sorted, 'r') as f:
                cnt = 0
                for line in f:
                    if cnt in val_lines:
                        val_f.write(line)
                    else:
                        train_f.write(line)
                    cnt += 1


def generate_pairs(package_file, dataset, sep='.'): 
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

    with open('./package/package_'+dataset+'.tsv', 'w') as fout:
        for k, v_set in mapping.items():
            for v in v_set:
                old_len = len(all_names)
                all_names.add(v)
                new_len = len(all_names)
                if old_len == new_len: #duplicate element
                    duplicate.add(v)
                fout.write(str(v) + '\t' + str(k) + '\n') #more specific package comes first

    with open('./package/duplicate_packages_'+dataset+'.tsv', 'w') as fdup:
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
    generate_train_val('./package/functions_sorted') #use command line sort <init file> -o <sorted file> to obtain a sorted file
    generate_pairs('./package/functions_train', 'train')
    #generate_pairs('./package/functions_val', 'val')
