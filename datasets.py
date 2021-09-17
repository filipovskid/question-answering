"""Dataset classes

SQuAD dataset adapted from:
    > https://github.com/chrischute/squad/blob/master/util.py

Author:
    Chris Chute (chute@stanford.edu)
    Darko Filipovski (@)
"""

import torch
import numpy as np
import torch.utils.data as data


class SQuAD(data.Dataset):
    """Stanford Question Answering Dataset (SQuAD).

    Parameters:
        data_path: Path to a .npz file containing the data as preprocessed by preprocess.py
    """

    def __init__(self, data_path):
        super(SQuAD, self).__init__()

        dataset = np.load(data_path)
        self.context_idxs = torch.from_numpy(dataset['context_idxs']).long()
        self.context_char_idxs = torch.from_numpy(dataset['context_char_idxs']).long()
        self.question_idxs = torch.from_numpy(dataset['ques_idxs']).long()
        self.question_char_idxs = torch.from_numpy(dataset['ques_char_idxs']).long()
        self.y1s = torch.from_numpy(dataset['y1s']).long()
        self.y2s = torch.from_numpy(dataset['y2s']).long()

        # Use index 0 for no-answer token (token 1 = OOV)
        batch_size, char_len, word_len = self.context_char_idxs.size()
        ones = torch.ones((batch_size, 1), dtype=torch.int64)
        self.context_idxs = torch.cat((ones, self.context_idxs), dim=1)
        self.question_idxs = torch.cat((ones, self.question_idxs), dim=1)

        ones = torch.ones((batch_size, 1, word_len), dtype=torch.int64)
        self.context_char_idxs = torch.cat((ones, self.context_char_idxs), dim=1)
        self.question_char_idxs = torch.cat((ones, self.question_char_idxs), dim=1)

        self.y1s += 1
        self.y2s += 1

        self.ids = torch.from_numpy(dataset['ids']).long()
        self.valid_idxs = [idx for idx in range(len(self.ids))]

    def __getitem__(self, idx):
        idx = self.valid_idxs[idx]
        example = (self.context_idxs[idx],
                   self.context_char_idxs[idx],
                   self.question_idxs[idx],
                   self.question_char_idxs[idx],
                   self.y1s[idx],
                   self.y2s[idx],
                   self.ids[idx])

        return example

    def __len__(self):
        return len(self.valid_idxs)
