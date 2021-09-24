"""The training process.

Partially adapted from:
    > https://github.com/chrischute/squad/blob/master/train.py
"""

import json
import torch
import argparse
import math
import utils

import numpy as np
import ujson as json
from tqdm import tqdm
import torch.utils.data as data
import torch.optim as optim
import torch.nn.functional as F

from datasets import SQuAD
from models import QANet
from utils import EMA


def parse_args():
    """Training arguments"""

    parser = argparse.ArgumentParser('Train')

    parser.add_argument('--train_record_file',
                        type=str,
                        default='./data/train.npz')
    parser.add_argument('--dev_record_file',
                        type=str,
                        default='./data/dev.npz')
    parser.add_argument('--test_record_file',
                        type=str,
                        default='./data/test.npz')
    parser.add_argument('--word_emb_file',
                        type=str,
                        default='./data/word_emb.json')
    parser.add_argument('--char_emb_file',
                        type=str,
                        default='./data/char_emb.json')
    parser.add_argument('--train_eval_file',
                        type=str,
                        default='./data/train_eval.json')
    parser.add_argument('--dev_eval_file',
                        type=str,
                        default='./data/dev_eval.json')
    parser.add_argument('--test_eval_file',
                        type=str,
                        default='./data/test_eval.json')

    parser.add_argument('--name',
                        '-n',
                        type=str,
                        required=True,
                        help='Name to identify training or test run.')
    parser.add_argument('--max_ans_len',
                        type=int,
                        default=15,
                        help='Maximum length of a predicted answer.')
    parser.add_argument('--num_workers',
                        type=int,
                        default=4,
                        help='Number of sub-processes to use per data loader.')
    parser.add_argument('--save_dir',
                        type=str,
                        default='./save/',
                        help='Base directory for saving information.')
    parser.add_argument('--batch_size',
                        type=int,
                        default=64,
                        help='Batch size per GPU. Scales automatically when \
                                  multiple GPUs are available.')
    parser.add_argument('--char_embed_size',
                        type=int,
                        default=64,
                        help='Dimension of character embedding vector.')
    parser.add_argument('--word_embed_size',
                        type=int,
                        default=300,
                        help='Dimension of word embedding vector.')
    parser.add_argument('--hidden_size',
                        type=int,
                        default=96,
                        help='Number of features in encoder hidden layers.')
    parser.add_argument('--num_visuals',
                        type=int,
                        default=10,
                        help='Number of examples to visualize in TensorBoard.')
    parser.add_argument('--load_path',
                        type=str,
                        default=None,
                        help='Path to load as a model checkpoint.')

    parser.add_argument('--eval_steps',
                        type=int,
                        default=50000,
                        help='Number of steps between successive evaluations.')
    parser.add_argument('--base_lr',
                        type=float,
                        default=1.0,
                        help='Base learning rate.')
    parser.add_argument('--lr',
                        type=float,
                        default=0.001,
                        help='Learning rate.')
    parser.add_argument('--num_warmup_steps',
                        type=int,
                        default=1000,
                        help='Number of warming up steps.')
    parser.add_argument('--l2_wd',
                        type=float,
                        default=5e-8,
                        help='L2 weight decay.')
    parser.add_argument('--num_epochs',
                        type=int,
                        default=30,
                        help='Number of epochs for which to train. Negative means forever.')
    parser.add_argument('--drop_prob',
                        type=float,
                        default=0.2,
                        help='Probability of zeroing an activation in dropout layers.')
    parser.add_argument('--metric_name',
                        type=str,
                        default='F1',
                        choices=('loss', 'EM', 'F1'),
                        help='Name of dev metric to determine best checkpoint.')
    parser.add_argument('--max_checkpoints',
                        type=int,
                        default=5,
                        help='Maximum number of checkpoints to keep on disk.')
    parser.add_argument('--max_grad_norm',
                        type=float,
                        default=5.0,
                        help='Maximum gradient norm for gradient clipping.')
    parser.add_argument('--seed',
                        type=int,
                        default=224,
                        help='Random seed for reproducibility.')
    parser.add_argument('--ema_decay',
                        type=float,
                        default=0.999,
                        help='Decay rate for exponential moving average of parameters.')
    parser.add_argument('--notebook',
                        default=False,
                        action="store_true",
                        help='Indicate that the training will happen in a notebook.')

    args = parser.parse_args()

    if args.metric_name == 'loss':
        args.maximize_metric = False
    elif args.metric_name in ('EM', 'F1'):
        args.maximize_metric = True

    return args


def collate_fn(samples):
    """Merge a list of samples to form a mini-batch of Tensor(s).

    Adapted from:
        https://github.com/yunjey/seq2seq-dataloader
    """

    def merge_0d(scalars, dtype=torch.int64):
        return torch.tensor(scalars, dtype=dtype)

    def merge_1d(arrays, dtype=torch.int64, pad_value=0):
        lengths = [(a != pad_value).sum() for a in arrays]
        padded = torch.zeros(len(arrays), max(lengths), dtype=dtype)
        for i, seq in enumerate(arrays):
            end = lengths[i]
            padded[i, :end] = seq[:end]
        return padded

    def merge_2d(matrices, dtype=torch.int64, pad_value=0):
        heights = [(m.sum(1) != pad_value).sum() for m in matrices]
        widths = [(m.sum(0) != pad_value).sum() for m in matrices]
        padded = torch.zeros(len(matrices), max(heights), max(widths), dtype=dtype)
        for i, seq in enumerate(matrices):
            height, width = heights[i], widths[i]
            padded[i, :height, :width] = seq[:height, :width]
        return padded

    # Group by tensor type
    context_idxs, context_char_idxs, \
    question_idxs, question_char_idxs, \
    y1s, y2s, ids = zip(*samples)

    # Merge into batch tensors
    context_idxs = merge_1d(context_idxs)
    context_char_idxs = merge_2d(context_char_idxs)
    question_idxs = merge_1d(question_idxs)
    question_char_idxs = merge_2d(question_char_idxs)
    y1s = merge_0d(y1s)
    y2s = merge_0d(y2s)
    ids = merge_0d(ids)

    return (context_idxs, context_char_idxs,
            question_idxs, question_char_idxs,
            y1s, y2s, ids)


def torch_from_json(path, dtype=torch.float32):
    with open(path, 'r') as f:
        array = np.array(json.load(f))

    tensor = torch.from_numpy(array).type(dtype)

    return tensor


def main(config):
    word_embeddings = torch_from_json(config.word_emb_file)
    char_embeddings = torch_from_json(config.char_emb_file)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = QANet(word_embeddings=word_embeddings,
                  char_embeddings=char_embeddings,
                  word_embed_size=config.word_embed_size,
                  char_embed_size=config.char_embed_size,
                  hidden_size=config.hidden_size,
                  embed_encoder_num_convs=4,
                  embed_encoder_kernel_size=7,
                  embed_encoder_num_heads=4,
                  embed_encoder_num_blocks=1,
                  model_encoder_num_convs=2,
                  model_encoder_kernel_size=5,
                  model_encoder_num_heads=4,
                  model_encoder_num_blocks=7)

    checkpoint_manager = utils.CheckpointManager(config.save_dir,
                                                 max_checkpoints=config.max_checkpoints,
                                                 metric_name=config.metric_name,
                                                 maximize_metric=config.maximize_metric)

    if config.load_path:
        model, step = checkpoint_manager.load_model(model, config.load_path, device)
    else:
        step = 0

    model.to(device)
    model.train()
    ema = EMA(model.parameters(), decay=config.ema_decay, use_num_updates=True)

    optimizer = optim.Adam(model.parameters(), lr=config.base_lr, betas=(0.9, 0.999), eps=1e-08,
                           weight_decay=config.l2_wd)
    cr = config.lr / math.log2(config.num_warmup_steps)
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda s: cr * math.log2(s + 1) if s < config.num_warmup_steps else config.lr
    )

    train_dataset = SQuAD(config.train_record_file)
    train_loader = data.DataLoader(train_dataset,
                                   batch_size=config.batch_size,
                                   shuffle=False,
                                   num_workers=config.num_workers,
                                   collate_fn=collate_fn)

    dev_dataset = SQuAD(config.dev_record_file)
    dev_loader = data.DataLoader(dev_dataset,
                                 batch_size=config.batch_size,
                                 shuffle=False,
                                 num_workers=config.num_workers,
                                 collate_fn=collate_fn)

    steps_till_eval = config.eval_steps
    epoch = step // len(train_dataset) + 1
    print("Training..")
    for epoch in range(epoch, config.num_epochs + 1):

        print(f"Epoch {epoch}/{config.num_epochs}")
        with tqdm(total=len(train_dataset)) as progress:
            for batch_count, (context_idxs, context_char_idxs, question_idxs, question_char_idxs, y1, y2, ids) in \
                    enumerate(train_loader):
                batch_size = context_idxs.size()[0]
                optimizer.zero_grad()

                context_idxs = context_idxs.to(device)
                context_char_idxs = context_char_idxs.to(device)
                question_idxs = question_idxs.to(device)
                question_char_idxs = question_char_idxs.to(device)

                y1, y2 = y1.to(device), y2.to(device)

                log_p1, log_p2 = model(context_idxs, context_char_idxs, question_idxs, question_char_idxs)
                loss = F.nll_loss(log_p1, y1) + F.nll_loss(log_p2, y2)

                loss.backward()
                torch.nn.utils.clip_grad_value_(model.parameters(), config.max_grad_norm)
                optimizer.step()
                scheduler.step()
                ema.update([param for param in model.parameters() if param.requires_grad])

                progress.update(batch_size)
                progress.set_postfix(epoch=epoch, step=step, loss=loss.item())

                step += batch_size
                steps_till_eval -= batch_size
                if steps_till_eval <= 0:
                    steps_till_eval -= batch_size

                    print(f'\nEvaluating at step {step}..')
                    with ema.average_parameters([param for param in model.parameters() if param.requires_grad]):
                        metrics, answer_preds = evaluate(model, dev_loader, config.dev_eval_file, device)
                        checkpoint_manager.save(step, model, metrics[config.metric_name], device)

                    print(f'Step: {step}, loss: {loss:05.2f}, EM: {metrics["EM"]:05.2f}, F1: {metrics["F1"]:05.2f}')


def evaluate(model, data_loader, eval_file, device):
    answer_preds = {}
    losses = []

    with open(eval_file, 'r') as f:
        eval_dict = json.load(f)

    model.eval()
    with torch.no_grad(), tqdm(position=0, total=len(data_loader.dataset)) as progress:
        for step, (context_idxs, context_char_idxs, question_idxs, question_char_idxs, y1, y2, ids) in \
                enumerate(data_loader):
            batch_size = context_idxs.size()[0]

            context_idxs = context_idxs.to(device)
            context_char_idxs = context_char_idxs.to(device)
            question_idxs = question_idxs.to(device)
            question_char_idxs = question_char_idxs.to(device)

            y1, y2 = y1.to(device), y2.to(device)
            log_p1, log_p2 = model(context_idxs, context_char_idxs, question_idxs, question_char_idxs)
            loss = F.nll_loss(log_p1, y1) + F.nll_loss(log_p2, y2)
            losses.append(loss.item())

            p1, p2 = log_p1.exp(), log_p2.exp()
            start_idxs, end_idxs = utils.infer_span(p_start=p1, p_end=p2)

            progress.update(batch_size)
            progress.set_postfix(step=step, loss=loss.item())

            answer_preds_, _ = utils.convert_tokens(eval_dict, ids.tolist(), start_idxs.tolist(), end_idxs.tolist())
            answer_preds.update(answer_preds_)

    model.train()
    metrics = utils.compute_metrics(eval_dict, answer_preds)
    metrics['loss'] = np.mean(losses)

    return metrics, answer_preds


if __name__ == '__main__':
    args = parse_args()
    main(args)
