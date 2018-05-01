#!/usr/bin/env python3
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch as th
import numpy as np
#import logging
import argparse
from torch.autograd import Variable
from collections import defaultdict as ddict
import torch.multiprocessing as mp
import model, train, rsgd
from data import slurp
from rsgd import RiemannianSGD
from sklearn.metrics import average_precision_score
import gc
import sys
import matplotlib.pyplot as plt
import time
from eval_utils import *


def ranking(types, model, distfn): #types here is adjacency matrix
    lt = th.from_numpy(model.embedding())
    embedding = Variable(lt, volatile=True)
    ranks = []
    ap_scores = []
    for s, s_types in types.items():
        s_e = Variable(lt[s].expand_as(embedding), volatile=True)
        _dists = model.dist()(s_e, embedding).data.cpu().numpy().flatten()
        _dists[s] = 1e+12
        _labels = np.zeros(embedding.size(0))
        _dists_masked = _dists.copy()
        _ranks = []
        for o, w in s_types.items():
            _dists_masked[o] = np.Inf
            _labels[o] = w
        ap_scores.append(average_precision_score(_labels, -_dists))
        for o, w in s_types.items():
            d = _dists_masked.copy()
            d[o] = _dists[o]
            r = np.argsort(d)
            _ranks.append(np.where(r == o)[0][0] + 1)
        ranks += _ranks
    return np.mean(ranks), np.mean(ap_scores)


def control(queue, types, data, distfn, processes, model_name, idx_dict, shortest_path_dict, opt):
    out_file = 'nearest_neighbor_results.txt'
    min_rank = (np.Inf, -1)
    max_map = (0, -1)
    while True:
        gc.collect()
        msg = queue.get()
        if msg is None:
            for p in processes:
                p.terminate()
            break
        else:
            epoch, elapsed, loss, model = msg
        if model is not None:
            # save model to fout
            th.save({
                'model': model.state_dict(),
                'objects': data.objects,
                'enames': data.enames,
                'distfn': distfn,
                'dim': opt.dim
            }, model_name+'_epoch_'+str(epoch)+'.pth') 

            # compute embedding quality
            mrank, mAP = ranking(types, model, distfn)
            if mrank < min_rank[0]:
                min_rank = (mrank, epoch)
            if mAP > max_map[0]:
                max_map = (mAP, epoch)
            print("EVAL: epoch %d  elapsed %.2f  loss %.3f  mean_rank %.2f  mAP %.4f  best_rank %.2f  best_mAP %.4f" \
                    % (epoch, elapsed, loss, mrank, mAP, min_rank[0], max_map[0]))

            result_dict = {'epoch': epoch, 'loss': round(loss,3), 'meanrank': round(mrank,2), 'mAP': round(mAP,4), 'bestrank': round(min_rank[0],2), 'bestmAP': round(max_map[0],4)}
            # nearest_neighbor & distance relation evaluation
            find_shortest_path(model, None, idx_dict, shortest_path_dict, result_dict)
        
        else:
            print("json_log: epoch %d  elapsed %.2f  loss %.3f" % (epoch, elapsed, loss))

        if epoch >= opt.epochs - 1:
            print("results: mAP %g  mAP epoch %d  mean rank %g  mean rank epoch %d" % (max_map[0], max_map[1], min_rank[0], min_rank[1]))
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Poincare Embeddings')
    parser.add_argument('-dim', help='Embedding dimension', type=int)
    parser.add_argument('-dset', help='Dataset to embed', type=str)
    parser.add_argument('-fout', help='Filename where to store model', type=str)
    parser.add_argument('-valset', help='Validation Dataset (optional)', type=str, default='')
    parser.add_argument('-dupset', help='Duplicate Data', type=str)
    parser.add_argument('-distfn', help='Distance function', type=str)
    parser.add_argument('-lr', help='Learning rate', type=float)
    parser.add_argument('-epochs', help='Number of epochs', type=int, default=200)
    parser.add_argument('-batchsize', help='Batchsize', type=int, default=50)
    parser.add_argument('-negs', help='Number of negatives', type=int, default=20)
    parser.add_argument('-nproc', help='Number of processes', type=int, default=5)
    parser.add_argument('-ndproc', help='Number of data loading processes', type=int, default=2)
    parser.add_argument('-eval_each', help='Run evaluation each n-th epoch', type=int, default=10)
    parser.add_argument('-burnin', help='Duration of burn in', type=int, default=20)
    #parser.add_argument('-debug', help='Print debug output', action='store_true', default=False)
    opt = parser.parse_args()

    th.set_default_tensor_type('torch.FloatTensor')
    # if opt.debug:
    #     log_level = logging.DEBUG
    # else:
    #     log_level = logging.INFO
    # log = logging.getLogger('poincare-nips17')
    # logging.basicConfig(level=log_level, format='%(message)s', stream=sys.stdout)
    idx, objects, enames_train = slurp(opt.dset)

    # create adjacency list for evaluation
    adjacency = ddict(dict)
    for i in range(len(idx)):
        s, o, w = idx[i]
        adjacency[s][o] = w
    adjacency = dict(adjacency)

    # setup Riemannian gradients for distances
    opt.retraction = rsgd.euclidean_retraction
    if opt.distfn == 'poincare':
        distfn = model.PoincareDistance
        opt.rgrad = rsgd.poincare_grad
    elif opt.distfn == 'euclidean':
        distfn = model.EuclideanDistance
        opt.rgrad = rsgd.euclidean_grad
    elif opt.distfn == 'transe':
        distfn = model.TranseDistance
        opt.rgrad = rsgd.euclidean_grad
    else:
        raise ValueError('Unknown distance function ' + opt.distfn)

    # initialize model and data
    model, data, model_name, conf = model.SNGraphDataset.initialize(distfn, opt, idx, objects, enames_train)

    # Build config string for log
    conf = [
            'distfn ' + opt.distfn,
            'dim ' + str(opt.dim),
            'lr ' + str(opt.lr),
            'batchsize ' + str(opt.batchsize),
            'negs ' + str(opt.negs)
            ] + conf
    print("json_conf: " + ', '.join(conf))

    # initialize optimizer
    optimizer = RiemannianSGD(
        model.parameters(),
        rgrad=opt.rgrad,
        retraction=opt.retraction,
        lr=opt.lr,
    )

    #_, enames_inv = build_graph(opt.dset)
    print("Start computing shortest path for file:", opt.valset + '_train.tsv')    
    t1 = time.time()
    G, enames_inv_val = build_graph(opt.valset + '_train.tsv')
    shortest_path_dict = dict(nx.shortest_path_length(G))
    t2 = time.time()
    idx_dict = dict()
    for i_val in shortest_path_dict:
        i_name = enames_inv_val[i_val]
        i_train = enames_train[i_name]
        idx_dict[i_val] = i_train
    print("Time to compute shortest paths for all nodes:", str(t2-t1))

    # if nproc == 0, run single threaded, otherwise run Hogwild
    if opt.nproc == 0:
        train.train(model, data, optimizer, opt, 0)
    else:
        queue = mp.Manager().Queue()
        model.share_memory()
        processes = []
        for rank in range(opt.nproc):
            p = mp.Process(
                target=train.train_mp,
                args=(model, data, optimizer, opt, rank + 1, queue)
            )
            p.start()
            processes.append(p)

        ctrl = mp.Process(
            target=control,
            args=(queue, adjacency, data, distfn, processes, model_name, idx_dict, shortest_path_dict, opt)
        )
        ctrl.start()
        ctrl.join()
