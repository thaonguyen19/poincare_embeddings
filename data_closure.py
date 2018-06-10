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
                first = package_names[0]
                for i in range(len(package_names)):
                    package_names[i] = package_names[i] + '-' + first
                package_names = ['ROOT'] + package_names  

                for i in range(len(package_names)-1):
                    for j in range(i+1, len(package_names)-1):
                        high, low = package_names[i], package_names[j]
                        mapping[high].add(low)                
                #process the last pair separately to check for duplicates

                if package_names[-1] in all_last_tokens:
                    duplicate.add(package_names[-1])
                else:
                    all_last_tokens.add(package_names[-1])

        for k, v_set in mapping.items():
            for v in v_set:
                #print("High", k, "Low", v)
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
    '''clique_type options: wo_clique, full_clique, basic_clique'''
    w = int(DEFAULT_WEIGHT/1)
    duplicate_dict = ddict(set)
    count_renamed_tokens = ddict(int)

    with open(file_write, 'a') as fout:
        with open(file_read, 'r') as fin:
            for line in fin:
                tokens = line.strip().split(sep)
                first = tokens[0]
                length = len(tokens[-1])
                for i in range(len(tokens)):
                    tokens[i] = tokens[i] + '-' + first

                last_token_name = tokens[-1]    
                if last_token_name in duplicate_set:
                    last_token_name = tokens[-1] + '_' + line.strip()[:(-length-1)]
                    count_renamed_tokens[last_token_name] += 1
                    if clique_type == 'full_clique':
                        duplicate_dict[tokens[-1]].add(last_token_name)
                    elif clique_type == 'basic_clique':
                        duplicate_dict[tokens[-1]].add(last_token_name) 

                for i in range(len(tokens)-1):
                    fout.write(last_token_name + '\t' + tokens[i] + '\n')

    if clique_type == 'full_clique' or clique_type == 'basic_clique':
        with open(file_write, 'a') as fout:
            for token, renamed_token_list in duplicate_dict.items():
                #print(token, len(renamed_token_list))
                for pair in itertools.combinations(renamed_token_list, 2):
                    fout.write(pair[0] + '\t' + pair[1] + '\t' + str(w) + '\n')
                    if pair[0] != pair[1]:
                        fout.write(pair[1] + '\t' + pair[0] + '\t' + str(w) + '\n')
    
    print("Checking repetition even in renamed tokens:")
    for k, v in count_renamed_tokens.items():
        if v >= 2:
            print(k, v)


if __name__ == '__main__':
    ### use command line sort <init file> -o <sorted file> to obtain a sorted file
    clique_type = sys.argv[1]
    print("Clique type:", clique_type)
    main_file = ('./package_closure_%s/functions_04182018_sorted' % clique_type)
    # main_packages = list(find_main_packages(main_file))
    # print(main_packages)
    # with open(main_file, 'r') as f:
    #     all_lines = f.readlines()
    # for i in range(len(main_packages)):
    #     for j in range(i+1, len(main_packages)):
    #         for line in all_lines:
    #             all_tokens = line.strip().split('.')[:-1]
    #             if main_packages[i] in all_tokens and main_packages[j] in all_tokens:
    #                 print(main_packages[i], main_packages[j], line)
    #                 break

    #debug_file = generate_debug_set(main_file)
    for package_file_sorted in [main_file]:#, debug_file]:
        #sorted_val_file = generate_train_val2(package_file_sorted) 
        duplicate_set, duplicate_file_name, tsv_package_file = generate_pairs(package_file_sorted, 'train')
        process_duplicate(duplicate_set, package_file_sorted, tsv_package_file, clique_type=clique_type)
