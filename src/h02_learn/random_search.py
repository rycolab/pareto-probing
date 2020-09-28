import os
import sys
import re
# import random
import copy
import itertools
import subprocess
# import math
import numpy as np
from tqdm import tqdm

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from h02_learn.dataset import get_data_loaders
from h02_learn.train import get_args
from util import util


def args2list(args):
    return [
        "--data-path", str(args.data_path),
        '--task', str(args.task),
        '--language', str(args.language),
        '--batch-size', str(args.batch_size),
        "--representation", str(args.representation),
        "--model", str(args.model),
        '--eval-batches', str(args.eval_batches),
        '--wait-epochs', str(args.wait_epochs),
        "--checkpoint-path", str(args.checkpoint_path),
        "--seed", str(args.seed),
    ]


def get_hyperparameters(search):
    hyperparameters = {
        '--hidden-size': search[0],
        '--nlayers': search[1],
        '--dropout': search[2],
        '--embedding-size': search[3],
        '--alpha': search[4],
        '--max-rank': search[5],
    }
    return dict2list(hyperparameters)


def get_hyperparameters_search(n_runs, representation, n_classes):
    bert_embedding_size = list([768])
    fast_embedding_size = list([300])
    onehot_embedding_size = list({int(2**x) for x in np.arange(5.6, 8.2, 0.01)})
    hidden_size = list({int(2**x) for x in np.arange(5, 10, 0.01)})
    nlayers = [0, 1, 2, 3, 4, 5]
    dropout = list(np.arange(0.0, 0.51, 0.01))

    alpha = np.array([0] + [2**x for x in np.linspace(start=-10, stop=3, num=(n_runs - 1))])
    max_rank = np.array([
        int(x) for x in np.linspace(start=1, stop=(n_classes + 0.99), num=(n_runs))])

    if representation in ['onehot', 'random']:
        embedding_size = onehot_embedding_size
    elif representation == 'fast':
        embedding_size = fast_embedding_size
    elif representation in ['bert', 'albert', 'roberta']:
        embedding_size = bert_embedding_size
    else:
        raise ValueError('Invalid representation %s' % representation)

    all_hyper = [hidden_size, nlayers, dropout, embedding_size]
    choices = []
    for hyper in all_hyper:
        choices += [np.random.choice(hyper, size=n_runs, replace=True)]
    choices += [alpha]
    choices += [max_rank]

    return list(zip(*choices))


def dict2list(data):
    list2d = [[k, str(x)] for k, x in data.items()]
    return list(itertools.chain.from_iterable(list2d))


def write_done(done_fname):
    with open(done_fname, "w") as f:
        f.write('done training\n')


def append_result(fname, values):
    with open(fname, "a+") as f:
        f.write(','.join(values) + '\n')


def get_results(out, err):
    res_names = ['loss', 'acc', 'norm', 'rank']
    res_pattern_base = r'^Final %s. Train: (\d+.\d+) Dev: (\d+.\d+) Test: (\d+.\d+)$'

    output = out.decode().split('\n')
    results = []

    try:
        for i, res_name in enumerate(res_names[::-1]):
            res_pattern = res_pattern_base % res_name
            m = re.match(res_pattern, output[-2 - i])
            train_res, dev_res, test_res = m.groups()
            results += [test_res, dev_res, train_res]

    except Exception as exc:
        print('Output:', output)
        raise ValueError('Error in subprocess: %s' % err.decode()) from exc

    return results


def run_experiment(run_count, hyper, hyperparameters, args, results_fname,
                   shuffle_labels=False, shuffle_input=False):
    if shuffle_input:
        if args.representation in ['random', 'fast', 'onehot']:
            append_result(
                results_fname,
                [str(run_count)] + [str(x) for x in hyper] +
                [str(shuffle_labels), str(shuffle_input)] + [''] * 12)
            return

        args = copy.copy(args)
        args.representation = args.representation + 'shuffled'
    opt_args = ['--shuffle-labels'] if shuffle_labels else []

    my_env = os.environ.copy()
    cmd = ['python', 'src/h02_learn/train.py'] + args2list(args) + opt_args + hyperparameters

    tqdm.write(str([args.representation] + hyperparameters + opt_args))
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=my_env)
    out, err = process.communicate()

    results = get_results(out, err)
    append_result(results_fname,
                  [str(run_count)] + [str(x) for x in hyper] +
                  [str(shuffle_labels), str(shuffle_input)] + results[::-1])


def main():
    # pylint: disable=too-many-locals

    args = get_args()
    n_runs = 50

    ouput_path = os.path.join(
        args.checkpoint_path, args.task, args.language, args.model, args.representation)
    results_fname = os.path.join(ouput_path, 'all_results.tsv')
    done_fname = os.path.join(ouput_path, 'finished.txt')


    _, _, _, n_classes, _ = \
        get_data_loaders(args.data_path, args.task, args.language,
                         'onehot', 1, 1)

    curr_iter = util.file_len(results_fname) - 1
    skip_first = False
    util.mkdir(ouput_path)

    if curr_iter == -1:
        res_columns = ['run', 'hidden_size', 'nlayers', 'dropout',
                       'embedding_size', 'alpha', 'max_rank',
                       'shuffle_labels', 'shuffle_input',
                       'train_loss', 'dev_loss', 'test_loss',
                       'train_acc', 'dev_acc', 'test_acc',
                       'train_norm', 'dev_norm', 'test_norm',
                       'train_rank', 'dev_rank', 'test_rank']
        append_result(results_fname, res_columns)
        curr_iter = 0

    if args.shuffle_labels:
        skip_first = curr_iter % 3
        curr_iter = int(curr_iter / 3)

    search = get_hyperparameters_search(n_runs, args.representation, n_classes)

    for i, hyper in tqdm(enumerate(search[curr_iter:]), initial=curr_iter, total=n_runs):
        run_count = curr_iter + i
        hyperparameters = get_hyperparameters(hyper)

        if not skip_first:
            run_experiment(run_count, hyper, hyperparameters, args, results_fname)

        if args.shuffle_labels and skip_first <= 1:
            run_experiment(run_count, hyper, hyperparameters, args, results_fname,
                           shuffle_labels=True)

        if args.shuffle_labels:
            run_experiment(run_count, hyper, hyperparameters, args, results_fname,
                           shuffle_labels=True, shuffle_input=True)

        skip_first = 0

    write_done(done_fname)


if __name__ == '__main__':
    main()
