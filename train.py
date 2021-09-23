import json
import torch
import argparse
import math

import numpy as np
import torch.utils.data as data
import torch.optim as optim
import torch.nn.functional as F

from datasets import SQuAD
from tqdm import tqdm
from models import QANet


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
    parser.add_argument('--use_squad_v2',
                        type=lambda s: s.lower().startswith('t'),
                        default=True,
                        help='Whether to use SQuAD 2.0 (unanswerable) questions.')
    parser.add_argument('--hidden_size',
                        type=int,
                        default=100,
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
                        choices=('NLL', 'EM', 'F1'),
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

    return parser.parse_args()


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
                  hidden_size=96,
                  embed_encoder_num_convs=4,
                  embed_encoder_kernel_size=7,
                  embed_encoder_num_heads=4,
                  embed_encoder_num_blocks=1,
                  model_encoder_num_convs=2,
                  model_encoder_kernel_size=5,
                  model_encoder_num_heads=4,
                  model_encoder_num_blocks=7)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=config.base_lr, betas=(0.9, 0.999), eps=1e-08,
                           weight_decay=config.l2_wd)
    cr = config.lr / math.log2(config.num_warmup_steps)
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda s: cr * math.log2(s + 1) if s < config.num_warmup_steps else config.lr
    )

    train_dataset = SQuAD(config.train_record_file)
    data_loader = data.DataLoader(train_dataset,
                                  batch_size=config.batch_size,
                                  shuffle=False,
                                  num_workers=config.num_workers,
                                  collate_fn=collate_fn)

    print("Training..")
    for epoch in range(1, config.num_epochs + 1):

        print(f"Epoch {epoch}/{config.num_epochs}")
        with tqdm(total=len(data_loader.dataset)) as progress:
            for context_idxs, context_char_idxs, question_idxs, question_char_idxs, y1, y2, ids in data_loader:
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

                progress.update(batch_size)
                progress.set_postfix(epoch=epoch, loss=loss.item())


if __name__ == '__main__':
    args = parse_args()
    main(args)
