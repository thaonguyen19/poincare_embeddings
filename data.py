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
import itertools
import random
import sys

DEFAULT_WEIGHT = 1

def generate_debug_set(package_file_sorted, n_test=300):
    debug_file_name = package_file_sorted[:-6] + 'debug_sorted'
    with open(package_file_sorted, 'r') as f:
        all_lines = f.readlines()
        test_lines = list(np.random.choice(all_lines, n_test))
        with open(debug_file_name, 'w') as fout:
            print(debug_file_name)
            for l in test_lines:
                fout.write(l)
    return debug_file_name

def generate_train_val2(package_file_sorted, n_val=1000, sep='.'):
    assert('sorted' in package_file_sorted)
    sorted_val_file = package_file_sorted[:-6]+'val'
    val_candidates = []
    with open(package_file_sorted, 'r') as f:
        for line in f:
            if random.uniform(0,1) < 1.0/32:
                val_candidates.append(line)

    with open(sorted_val_file, 'w') as val_f:
        for line in val_candidates:
            val_f.write(line)
    return sorted_val_file


def generate_train_val(package_file_sorted, n_val=1000, sep='.'):
    #package_file_sorted name has format "*_sorted" --> output "*_train", "*_val"
    assert('sorted' in package_file_sorted)
    sorted_val_file = package_file_sorted[:-6]+'val'
    val_candidates = []

    with open(package_file_sorted, 'r') as f:
        cnt = 0
        prev_line = None
        for line in f:
            if len(line.strip().split(sep)) > 1: #not root
                if prev_line is not None and prev_line in line: #not leaf
                    val_candidates.append(cnt)

            prev_line = line.strip()
            cnt += 1

    if len(val_candidates) == 0:
        with open(package_file_sorted, 'r') as f:
            all_lines = f.readlines()
        val_lines = list(np.random.choice(all_lines, n_val))
    else:
        val_lines = list(np.random.choice(val_candidates, n_val))

    with open(sorted_val_file, 'w') as val_f:
        with open(package_file_sorted, 'r') as f:
            cnt = 0
            for line in f:
                if cnt in val_lines:
                    val_f.write(line)
                cnt += 1
    return sorted_val_file


def create_file_wo_duplicate(tsv_package_file): #to check for cycle
    tsv_package_file_wo_duplicate = tsv_package_file[:-4] + '_wo_duplicate.tsv'
    with open(tsv_package_file, 'r') as fin:
        with open(tsv_package_file_wo_duplicate, 'w') as fout:
            for line in fin:
                fout.write(line)


def generate_pairs(package_file, dataset, sep='.'): 
    #package_file name has format "*_sorted"
    mapping = ddict(set) #map from higher order element to the all direct children in the hierarchy
    all_last_tokens = set()
    duplicate = set() #assume the immediate parent package names would be different
    tsv_package_file = package_file[:-6]+dataset+'.tsv'

    with open(tsv_package_file, 'w') as fout:
        with open(package_file, 'r') as f:
            for line in f:
                package_names = line.strip().split(sep)
                if len(package_names) < 2:
                    continue
                package_names = ['ROOT'] + package_names  
                for i in range(len(package_names)-2):
                    high, low = package_names[i], package_names[i+1]
                    #fout.write(low + '\t' + high + '\n') #more specific package comes first
                    mapping[high].add(low)
                
                #process the last pair separately to check for duplicates
                if package_names[-1] in all_last_tokens:
                    duplicate.add(package_names[-1])
                else:
                    all_last_tokens.add(package_names[-1])

        for k, v_set in mapping.items():
            for v in v_set:
                #if v in duplicate: #don't add duplicate elements for now
                #    continue
                fout.write(v + '\t' + k + '\n') #more specific package comes first

    duplicate_file_name = package_file[:-6]+'duplicate_'+dataset
    with open(duplicate_file_name, 'w') as fdup:
        for i in duplicate:
            fdup.write(str(i) + '\n')

    return duplicate, duplicate_file_name, tsv_package_file


def get_duplicate(duplicate_file_name):
    result = []
    with open(duplicate_file_name, 'r') as fdup:
        for line in file:
            result.append(line.strip())
    return result


def process_duplicate(duplicate_set, file_read, file_write, clique_type, sep='.'):
    '''clique_type options: no_clique, full_clique, basic_clique'''
    w = int(DEFAULT_WEIGHT/1)
    duplicate_dict = ddict(set)
    count_renamed_tokens = ddict(int)

    with open(file_write, 'a') as fout:
        with open(file_read, 'r') as fin:
            for line in fin:
                tokens = line.strip().split(sep)
                if tokens[-1] not in duplicate_set:
                    fout.write(tokens[-1] + '\t' + tokens[-2] + '\n')
                else:
                    renamed_token = tokens[-1] + '_' + line.strip()[:(-len(tokens[-1])-1)]
                    count_renamed_tokens[renamed_token] += 1
                    fout.write(renamed_token + '\t' + tokens[-2] + '\n')
                    if clique_type == 'full_clique':
                        duplicate_dict[tokens[-1]].add(renamed_token)
                    elif clique_type == 'basic_clique':
                        duplicate_dict[(tokens[0], tokens[-1])].add(renamed_token)
                    

    if clique_type == 'full_clique' or clique_type == 'basic_clique':
        with open(file_write, 'a') as fout:
            for token, renamed_token_list in duplicate_dict.items():
                print(token, len(renamed_token_list))
                for pair in itertools.combinations(renamed_token_list, 2):
                    fout.write(pair[0] + '\t' + pair[1] + '\t' + str(w) + '\n')
                    if pair[0] != pair[1]:
                        fout.write(pair[1] + '\t' + pair[0] + '\t' + str(w) + '\n')
    
    print("Checking repetition even in renamed tokens:")
    for k, v in count_renamed_tokens.items():
        if v >= 2:
            print(k, v)


def parse_line(line, length=2, sep='\t'):
    #each line is either (head, tail) or (head, tail, weight). Return tuple of (head, tail, weight)
    d = line.strip().split(sep)
    if len(d) == length:
        w = DEFAULT_WEIGHT
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
    return idx, objects, enames


if __name__ == '__main__':
    ### use command line sort <init file> -o <sorted file> to obtain a sorted file
    clique_type = sys.argv[1]
    print("Clique type:", clique_type)
    main_file = ('./package_%s/functions_04182018_sorted' % clique_type)
    #debug_file = generate_debug_set(main_file)
    for package_file_sorted in [main_file]:#, debug_file]:
        #sorted_val_file = generate_train_val2(package_file_sorted) 
        duplicate_set, duplicate_file_name, tsv_package_file = generate_pairs(package_file_sorted, 'train')
        create_file_wo_duplicate(tsv_package_file)
        process_duplicate(duplicate_set, package_file_sorted, tsv_package_file, clique_type=clique_type)
