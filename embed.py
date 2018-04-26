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


def eval_sanity_check(val_filename, checkpoint_file, out_file, dim, distfn, train_dset, negs=50, n_top=5): 
    #GOAL: print n_top top ranked entries
    #how to compute dist given a linkage of packages
    #for each import, go through all other imports (starting from sklearn), as long as it exceeds the min_dist, break and move on the next search
    all_val_strs = []
    with open(val_filename, 'r') as f:
        for line in f:
            all_val_strs.append(line.strip())

    checkpoint = th.load(checkpoint_file)
    parser = argparse.ArgumentParser(description='Train Poincare Embeddings')
    opt = parser.parse_args()
    opt.dim = dim
    opt.distfn = distfn
    opt.negs = negs #doesn't matter
    opt.dset = train_dset 
    idx, objects, name_to_idx = slurp(opt.dset)

    model, data, model_name, _ = model.SNGraphDataset.initialize(distfn, opt, idx, objects)
    model.load_state_dict(checkpoint['state_dict'])
    lt = th.from_numpy(model.embedding())

    with open(out_file, 'w') as fout:
        for s in all_val_strs:
            max_sim = None
            n_top_candidates = [''] * n_top
            #start computing dists

        


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


def control(queue, types, data, distfn, nepochs, processes, model_name):
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
                'epoch': epoch,
                'objects': data.objects,
            }, model_name+'.pth') 
            # compute embedding quality
            mrank, mAP = ranking(types, model, distfn)
            if mrank < min_rank[0]:
                min_rank = (mrank, epoch)
            if mAP > max_map[0]:
                max_map = (mAP, epoch)
            print("EVAL: epoch %d  elapsed %.2f  loss %.3f  mean_rank %.2f  mAP %.4f  best_rank %.2f  best_mAP %.4f" \
                    % (epoch, elapsed, loss, mrank, mAP, min_rank[0], max_map[0]))

        else:
            print("json_log: epoch %d  elapsed %.2f  loss %.3f" % (epoch, elapsed, loss))

        if epoch >= nepochs - 1:
            print("results: mAP %g  mAP epoch %d  mean rank %g  mean rank epoch %d" % (max_map[0], max_map[1], min_rank[0], min_rank[1]))
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Poincare Embeddings')
    parser.add_argument('-dim', help='Embedding dimension', type=int)
    parser.add_argument('-dset', help='Dataset to embed', type=str)
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
    idx, objects, _ = slurp(opt.dset)

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
    model, data, model_name, conf = model.SNGraphDataset.initialize(distfn, opt, idx, objects)

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
            args=(queue, adjacency, data, distfn, opt.epochs, processes, model_name)
        )
        ctrl.start()
        ctrl.join()
