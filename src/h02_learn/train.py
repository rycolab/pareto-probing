import os
import sys
import argparse
from tqdm import tqdm
import torch
import torch.optim as optim

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from h02_learn.dataset import get_data_loaders
from h02_learn.model import MLP, Linear, RankMax, TransparentDataParallel, LinearParser, MLPParser
from h02_learn.train_info import TrainInfo
from util import util
from util import constants


def get_model_name(args):
    fpath = 'nl_%d-es_%d-hs_%d-d_%.4f-a_%.4f' % \
        (args.nlayers, args.embedding_size, args.hidden_size, args.dropout, args.alpha)
    return fpath


def get_args():
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument('--language', type=str, required=True)
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--shuffle-labels', action='store_true')
    parser.add_argument('--batch-size', type=int, default=2048)
    parser.add_argument("--representation", type=str, required=True)
    # Model
    parser.add_argument('--model', type=str, default='linear')
    parser.add_argument('--nlayers', type=int, default=3)
    parser.add_argument('--embedding-size', type=int, default=300)
    parser.add_argument('--hidden-size', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--alpha', type=float, default=.3)
    parser.add_argument('--max-rank', type=int, default=10)
    # Optimization
    parser.add_argument('--eval-batches', type=int, default=100)
    parser.add_argument('--wait-epochs', type=int, default=10)
    # Others
    parser.add_argument("--checkpoint-path", type=str, required=True)
    parser.add_argument("--seed", type=int, default=20)

    args = parser.parse_args()
    args.wait_iterations = args.wait_epochs * args.eval_batches
    args.save_path = os.path.join(
        args.checkpoint_path, args.task, args.language, args.model,
        args.representation, get_model_name(args))

    util.config(args.seed)
    print(args)

    if args.representation in ['bert', 'albert', 'roberta']:
        args.embedding_size = 768
    elif args.representation == 'fast':
        args.embedding_size = 300
    if args.task == 'dep_label':
        args.embedding_size = args.embedding_size * 2

    if args.task == 'parse':
        args.batch_size = 128

    return args


def get_model(n_classes, n_words, args):
    if args.task != 'parse':
        if args.model == 'linear':
            model = Linear(
                args.task, embedding_size=args.embedding_size,
                n_classes=n_classes, alpha=args.alpha,
                dropout=args.dropout, representation=args.representation,
                n_words=n_words)
        elif args.model == 'mlp':
            model = MLP(
                args.task, embedding_size=args.embedding_size,
                n_classes=n_classes, hidden_size=args.hidden_size,
                nlayers=args.nlayers, dropout=args.dropout,
                representation=args.representation, n_words=n_words)
        elif args.model == 'rank-max':
            model = RankMax(
                args.task, embedding_size=args.embedding_size,
                n_classes=n_classes, max_rank=args.max_rank,
                dropout=args.dropout, representation=args.representation,
                n_words=n_words)
    else:
        if args.model == 'linear':
            model = LinearParser(
                embedding_size=args.embedding_size, alpha=args.alpha,
                dropout=args.dropout,
                representation=args.representation, n_words=n_words)
        if args.model == 'mlp':
            model = MLPParser(
                embedding_size=args.embedding_size, hidden_size=args.hidden_size,
                nlayers=args.nlayers, dropout=args.dropout,
                representation=args.representation, n_words=n_words)

    print(model)
    if torch.cuda.device_count() > 1:
        model = TransparentDataParallel(model)
    return model.to(device=constants.device)


def _evaluate(evalloader, model):
    dev_loss, dev_acc = 0, 0
    for x, y in evalloader:
        loss, acc = model.eval_batch(x, y)
        dev_loss += loss
        dev_acc += acc

    norm = model.get_norm()
    rank = model.get_rank()

    n_instances = len(evalloader.dataset)
    return {
        'loss': dev_loss / n_instances,
        'acc': dev_acc / n_instances,
        'norm': norm.item(),
        'rank': rank,
    }


def evaluate(evalloader, model):
    model.eval()
    with torch.no_grad():
        result = _evaluate(evalloader, model)
    model.train()
    return result


def train_epoch(trainloader, devloader, model, optimizer, train_info):
    for x, y in trainloader:
        loss = model.train_batch(x, y, optimizer)
        train_info.new_batch(loss)

        if train_info.eval:
            dev_results = evaluate(devloader, model)

            if train_info.is_best(dev_results):
                model.set_best()
            elif train_info.finish:
                train_info.print_progress(dev_results)
                return

            train_info.print_progress(dev_results)


def train(trainloader, devloader, model, eval_batches, wait_iterations):
    optimizer = optim.Adam(model.parameters())

    with tqdm(total=wait_iterations) as pbar:
        train_info = TrainInfo(pbar, wait_iterations, eval_batches)
        while not train_info.finish:
            train_epoch(trainloader, devloader, model,
                        optimizer, train_info)

    model.recover_best()


def eval_all(model, trainloader, devloader, testloader):
    train_results = evaluate(trainloader, model)
    dev_results = evaluate(devloader, model)
    test_results = evaluate(testloader, model)

    for res_name in ['loss', 'acc', 'norm', 'rank']:
        print('Final %s. Train: %.4f Dev: %.4f Test: %.4f' %
              (res_name, train_results[res_name], dev_results[res_name], test_results[res_name]))

    return train_results, dev_results, test_results


def save_results(model, train_results, dev_results, test_results, results_fname):
    results = [model.print_param_names() +
               ['train_loss', 'dev_loss', 'test_loss',
                'train_acc', 'dev_acc', 'test_acc',
                'train_norm', 'dev_norm', 'test_norm']]
    results += [model.print_params() +
                [train_results['loss'], dev_results['loss'], test_results['loss'],
                 train_results['acc'], dev_results['acc'], test_results['acc'],
                 train_results['norm'], dev_results['norm'], test_results['norm']]]
    util.write_csv(results_fname, results)


def save_checkpoints(model, train_results, dev_results, test_results, save_path):
    util.mkdir(save_path)
    model.save(save_path)
    results_fname = save_path + '/results.csv'
    save_results(model, train_results, dev_results, test_results, results_fname)


def main():
    args = get_args()

    trainloader, devloader, testloader, n_classes, n_words = \
        get_data_loaders(args.data_path, args.task, args.language,
                         args.representation, args.embedding_size, args.batch_size,
                         args.shuffle_labels)
    print('Language: %s Train size: %d Dev size: %d Test size: %d # Classes %d' %
          (args.language, len(trainloader.dataset),
           len(devloader.dataset), len(testloader.dataset), n_classes))

    model = get_model(n_classes, n_words, args)
    train(trainloader, devloader, model, args.eval_batches, args.wait_iterations)

    train_results, dev_results, test_results = eval_all(
        model, trainloader, devloader, testloader)

    save_checkpoints(model, train_results, dev_results, test_results, args.save_path)


if __name__ == '__main__':
    main()
