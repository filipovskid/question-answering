from __future__ import division
from __future__ import unicode_literals

from typing import Iterable, Optional

import torch
import collections
import weakref
import copy
import contextlib
import string
import re
import os
import shutil
import queue
import pathlib

import torch.nn.functional as F
import ujson as json


class EMA:
    """
    Maintains (exponential) moving average of a set of parameters.

    Code from:
        > https://github.com/fadel/pytorch_ema/blob/master/torch_ema/ema.py

    Args:
        parameters: Iterable of `torch.nn.Parameter` (typically from
            `model.parameters()`).
        decay: The exponential decay.
        use_num_updates: Whether to use number of updates when computing
            averages.
    """

    def __init__(
            self,
            parameters: Iterable[torch.nn.Parameter],
            decay: float,
            use_num_updates: bool = True
    ):
        if decay < 0.0 or decay > 1.0:
            raise ValueError('Decay must be between 0 and 1')
        self.decay = decay
        self.num_updates = 0 if use_num_updates else None
        parameters = list(parameters)
        self.shadow_params = [p.clone().detach()
                              for p in parameters if p.requires_grad]
        self.collected_params = None
        # By maintaining only a weakref to each parameter,
        # we maintain the old GC behaviour of ExponentialMovingAverage:
        # if the model goes out of scope but the ExponentialMovingAverage
        # is kept, no references to the model or its parameters will be
        # maintained, and the model will be cleaned up.
        self._params_refs = [weakref.ref(p) for p in parameters]

    def _get_parameters(
            self,
            parameters: Optional[Iterable[torch.nn.Parameter]]
    ) -> Iterable[torch.nn.Parameter]:
        if parameters is None:
            parameters = [p() for p in self._params_refs]
            if any(p is None for p in parameters):
                raise ValueError(
                    "(One of) the parameters with which this "
                    "ExponentialMovingAverage "
                    "was initialized no longer exists (was garbage collected);"
                    " please either provide `parameters` explicitly or keep "
                    "the model to which they belong from being garbage "
                    "collected."
                )
            return parameters
        else:
            parameters = list(parameters)
            if len(parameters) != len(self.shadow_params):
                raise ValueError(
                    "Number of parameters passed as argument is different "
                    "from number of shadow parameters maintained by this "
                    "ExponentialMovingAverage"
                )
            return parameters

    def update(
            self,
            parameters: Optional[Iterable[torch.nn.Parameter]] = None
    ) -> None:
        """
        Update currently maintained parameters.

        Call this every time the parameters are updated, such as the result of
        the `optimizer.step()` call.

        Args:
            parameters: Iterable of `torch.nn.Parameter`; usually the same set of
                parameters used to initialize this object. If `None`, the
                parameters with which this `ExponentialMovingAverage` was
                initialized will be used.
        """
        parameters = self._get_parameters(parameters)
        decay = self.decay
        if self.num_updates is not None:
            self.num_updates += 1
            decay = min(
                decay,
                (1 + self.num_updates) / (10 + self.num_updates)
            )
        one_minus_decay = 1.0 - decay
        with torch.no_grad():
            parameters = [p for p in parameters if p.requires_grad]
            for s_param, param in zip(self.shadow_params, parameters):
                tmp = (s_param - param)
                # tmp will be a new tensor so we can do in-place
                tmp.mul_(one_minus_decay)
                s_param.sub_(tmp)

    def copy_to(
            self,
            parameters: Optional[Iterable[torch.nn.Parameter]] = None
    ) -> None:
        """
        Copy current averaged parameters into given collection of parameters.

        Args:
            parameters: Iterable of `torch.nn.Parameter`; the parameters to be
                updated with the stored moving averages. If `None`, the
                parameters with which this `ExponentialMovingAverage` was
                initialized will be used.
        """
        parameters = self._get_parameters(parameters)
        for s_param, param in zip(self.shadow_params, parameters):
            if param.requires_grad:
                param.data.copy_(s_param.data)

    def store(
            self,
            parameters: Optional[Iterable[torch.nn.Parameter]] = None
    ) -> None:
        """
        Save the current parameters for restoring later.

        Args:
            parameters: Iterable of `torch.nn.Parameter`; the parameters to be
                temporarily stored. If `None`, the parameters of with which this
                `ExponentialMovingAverage` was initialized will be used.
        """
        parameters = self._get_parameters(parameters)
        self.collected_params = [
            param.clone()
            for param in parameters
            if param.requires_grad
        ]

    def restore(
            self,
            parameters: Optional[Iterable[torch.nn.Parameter]] = None
    ) -> None:
        """
        Restore the parameters stored with the `store` method.
        Useful to validate the model with EMA parameters without affecting the
        original optimization process. Store the parameters before the
        `copy_to` method. After validation (or model saving), use this to
        restore the former parameters.

        Args:
            parameters: Iterable of `torch.nn.Parameter`; the parameters to be
                updated with the stored parameters. If `None`, the
                parameters with which this `ExponentialMovingAverage` was
                initialized will be used.
        """
        if self.collected_params is None:
            raise RuntimeError(
                "This ExponentialMovingAverage has no `store()`ed weights "
                "to `restore()`"
            )
        parameters = self._get_parameters(parameters)
        for c_param, param in zip(self.collected_params, parameters):
            if param.requires_grad:
                param.data.copy_(c_param.data)

    @contextlib.contextmanager
    def average_parameters(
            self,
            parameters: Optional[Iterable[torch.nn.Parameter]] = None
    ):
        r"""
        Context manager for validation/inference with averaged parameters.

        Equivalent to:

            ema.store()
            ema.copy_to()
            try:
                ...
            finally:
                ema.restore()

        Args:
            parameters: Iterable of `torch.nn.Parameter`; the parameters to be
                updated with the stored parameters. If `None`, the
                parameters with which this `ExponentialMovingAverage` was
                initialized will be used.
        """
        parameters = self._get_parameters(parameters)
        self.store(parameters)
        self.copy_to(parameters)
        try:
            yield
        finally:
            self.restore(parameters)

    def to(self, device=None, dtype=None) -> None:
        r"""Move internal buffers of the ExponentialMovingAverage to `device`.

        Args:
            device: like `device` argument to `torch.Tensor.to`
        """
        # .to() on the tensors handles None correctly
        self.shadow_params = [
            p.to(device=device, dtype=dtype)
            if p.is_floating_point()
            else p.to(device=device)
            for p in self.shadow_params
        ]
        if self.collected_params is not None:
            self.collected_params = [
                p.to(device=device, dtype=dtype)
                if p.is_floating_point()
                else p.to(device=device)
                for p in self.collected_params
            ]
        return

    def state_dict(self) -> dict:
        r"""Returns the state of the ExponentialMovingAverage as a dict."""
        # Following PyTorch conventions, references to tensors are returned:
        # "returns a reference to the state and not its copy!" -
        # https://pytorch.org/tutorials/beginner/saving_loading_models.html#what-is-a-state-dict
        return {
            "decay": self.decay,
            "num_updates": self.num_updates,
            "shadow_params": self.shadow_params,
            "collected_params": self.collected_params
        }

    def load_state_dict(self, state_dict: dict) -> None:
        r"""Loads the ExponentialMovingAverage state.

        Args:
            state_dict (dict): EMA state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        # deepcopy, to be consistent with module API
        state_dict = copy.deepcopy(state_dict)
        self.decay = state_dict["decay"]
        if self.decay < 0.0 or self.decay > 1.0:
            raise ValueError('Decay must be between 0 and 1')
        self.num_updates = state_dict["num_updates"]
        assert self.num_updates is None or isinstance(self.num_updates, int), \
            "Invalid num_updates"

        self.shadow_params = state_dict["shadow_params"]
        assert isinstance(self.shadow_params, list), \
            "shadow_params must be a list"
        assert all(
            isinstance(p, torch.Tensor) for p in self.shadow_params
        ), "shadow_params must all be Tensors"

        self.collected_params = state_dict["collected_params"]
        if self.collected_params is not None:
            assert isinstance(self.collected_params, list), \
                "collected_params must be a list"
            assert all(
                isinstance(p, torch.Tensor) for p in self.collected_params
            ), "collected_params must all be Tensors"
            assert len(self.collected_params) == len(self.shadow_params), \
                "collected_params and shadow_params had different lengths"

        if len(self.shadow_params) == len(self._params_refs):
            # Consistant with torch.optim.Optimizer, cast things to consistant
            # device and dtype with the parameters
            params = [p() for p in self._params_refs]
            # If parameters have been garbage collected, just load the state
            # we were given without change.
            if not any(p is None for p in params):
                # ^ parameter references are still good
                for i, p in enumerate(params):
                    self.shadow_params[i] = self.shadow_params[i].to(
                        device=p.device, dtype=p.dtype
                    )
                    if self.collected_params is not None:
                        self.collected_params[i] = self.collected_params[i].to(
                            device=p.device, dtype=p.dtype
                        )
        else:
            raise ValueError(
                "Tried to `load_state_dict()` with the wrong number of "
                "parameters in the saved state."
            )


class CheckpointManager:
    """Class to save and load model checkpoints.
    Save the best checkpoints as measured by a metric value passed into the
    `save` method. Overwrite checkpoints with better checkpoints once
    `max_checkpoints` have been saved.
    Args:
        save_dir (str): Directory to save checkpoints.
        max_checkpoints (int): Maximum number of checkpoints to keep before
            overwriting old ones.
        metric_name (str): Name of metric used to determine best model.
        maximize_metric (bool): If true, best checkpoint is that which maximizes
            the metric value passed in via `save`. Otherwise, best checkpoint
            minimizes the metric.
        log (logging.Logger): Optional logger for printing information.

    Adapted from:
        > https://github.com/chrischute/squad/blob/master/util.py
    """

    def __init__(self, save_dir, max_checkpoints, metric_name, maximize_metric=False, log=None):
        super(CheckpointManager, self).__init__()

        self.save_dir = save_dir
        self.max_checkpoints = max_checkpoints
        self.metric_name = metric_name
        self.maximize_metric = maximize_metric
        self.best_val = None
        self.ckpt_paths = queue.PriorityQueue()
        self.log = log
        self._print(f"Saver will {'max' if maximize_metric else 'min'}imize {metric_name}...")

        save_dir_path = pathlib.Path(self.save_dir)
        save_dir_path.mkdir(parents=True, exist_ok=True)

    def is_best(self, metric_val):
        """Check whether `metric_val` is the best seen so far.
        Args:
            metric_val (float): Metric value to compare to prior checkpoints.
        """
        if metric_val is None:
            # No metric reported
            return False

        if self.best_val is None:
            # No checkpoint saved yet
            return True

        return ((self.maximize_metric and self.best_val < metric_val)
                or (not self.maximize_metric and self.best_val > metric_val))

    def _print(self, message):
        """Print a message if logging is enabled."""
        if self.log is not None:
            self.log.info(message)
        else:
            print(message)

    def save(self, step, model, metric_val, device, predictions=None):
        """Save model parameters to disk.
        Args:
            step (int): Total number of examples seen during training so far.
            model (torch.nn.DataParallel): Model to save.
            metric_val (float): Determines whether checkpoint is best so far.
            device (torch.device): Device where model resides.
            predictions (dict): Question id - predicted answer mappings.
        """
        ckpt_dict = {
            'model_name': model.__class__.__name__,
            'model_state': model.cpu().state_dict(),
            'step': step
        }
        model.to(device)

        checkpoint_path = os.path.join(self.save_dir, f'step_{step}.pth.tar')
        predictions_path = os.path.join(self.save_dir, f'step_{step}_predictions.json')

        torch.save(ckpt_dict, checkpoint_path)

        if predictions:
            with open(predictions_path, "w") as f:
                json.dump(predictions, f)

        self._print(f'Saved checkpoint: {checkpoint_path}')

        if self.is_best(metric_val):
            # Save the best model
            self.best_val = metric_val
            best_path = os.path.join(self.save_dir, 'best.pth.tar')
            shutil.copy(checkpoint_path, best_path)
            self._print(f'New best checkpoint at step {step}...')

        # Add checkpoint path to priority queue (lowest priority removed first)
        if self.maximize_metric:
            priority_order = metric_val
        else:
            priority_order = -metric_val

        self.ckpt_paths.put((priority_order, checkpoint_path))

        # Remove a checkpoint if more than max_checkpoints have been saved
        if self.ckpt_paths.qsize() > self.max_checkpoints:
            _, worst_ckpt = self.ckpt_paths.get()
            try:
                os.remove(worst_ckpt)
                self._print(f'Removed checkpoint: {worst_ckpt}')
            except OSError:
                # Avoid crashing if checkpoint has been removed or protected
                pass

    def load_model(self, model, checkpoint_path, device):
        """Load model from checkpoint"""
        self._print(f'Loading model checkpoint from {checkpoint_path}.')
        ckpt_dict = torch.load(checkpoint_path, map_location=device)

        model.load_state_dict(ckpt_dict['model_state'])
        step = ckpt_dict['step']

        self._print(f'Model loaded, step: {step}.')

        return model, step


def mask_logits(logits, mask):
    mask = mask.type(torch.float32)
    return (1 - mask) * logits + mask * -1e30


def masked_softmax(logits, mask, dim=-1, log_softmax=False):
    masked_logits = mask_logits(logits, mask)
    softmax_fn = F.log_softmax if log_softmax else F.softmax
    probs = softmax_fn(masked_logits, dim)

    return probs


def infer_span(p_start, p_end, max_len=None):
    """Infer the start and end indices of an answer given probabilities of
    the starting and ending positions. The predicted span (i, j) is
    chosen such that:

            p_start[i] * p_end[j] is maximized and i <= j.

    Parameters:
        p_start: torch.Tensor of size (batch_size, context_len). Probabilities for
        each token being a starting position.
        p_end:  torch.Tensor of size (batch_size, context_len). Probabilities for
        each token being an ending position.
        max_len: If the answer is bounded this sets the maximum answer length.
    """
    context_len = p_start.size()[1]

    # Broadcasted size (batch_size, context_len, context_len)
    p_joint = torch.matmul(p_start.unsqueeze(2), p_end.unsqueeze(1))

    # Retain only values where start <= end
    span_mask = torch.triu(torch.ones((context_len, context_len), device=p_start.device))

    if max_len:
        span_mask -= torch.triu(torch.ones((context_len, context_len), device=p_start.device), diagonal=max_len)

    # Joint probability start and end to be the 0th token. This is used as no answer.
    p_no_answer = p_joint[:, 0, 0].clone()
    p_joint[:, 0, :] = 0
    p_joint[:, :, 0] = 0
    p_joint *= span_mask

    # Find (i, j) that maximize p_start[i] * p_end[i]
    row_max, _ = torch.max(p_joint, dim=2)
    col_max, _ = torch.max(p_joint, dim=1)
    start_idxs = torch.argmax(row_max, dim=1)
    end_idxs = torch.argmax(col_max, dim=1)

    # If probability of not having an answer is greater than the joint probability
    # of having an answer, then the query has no answer.
    max_joint_p, _ = torch.max(row_max, dim=1)
    start_idxs[p_no_answer > max_joint_p] = 0
    end_idxs[p_no_answer > max_joint_p] = 0

    return start_idxs, end_idxs


def convert_tokens(eval_dict, qa_id, start_idxs, end_idxs):
    answer_dict = {}
    global_dict = {}

    for qid, start_idx, end_idx in zip(qa_id, start_idxs, end_idxs):
        qid = str(qid)
        context = eval_dict[qid]['context']
        spans = eval_dict[qid]['spans']
        uuid = eval_dict[qid]['uuid']

        if start_idx == 0 or end_idx == 0:
            answer_dict[qid] = ''
            global_dict[uuid] = ''
        else:
            start_idx, end_idx = start_idx - 1, end_idx - 1
            token_start_idx = spans[start_idx][0]
            token_end_idx = spans[end_idx][1]

            answer_dict[qid] = context[token_start_idx:token_end_idx]
            global_dict[uuid] = context[token_start_idx:token_end_idx]

    return answer_dict, global_dict


def compute_metrics(eval_dict, answer_dict):
    """Measure scores over predicted and ground truth answers.
    There are multiple possible ground_truth answers, so the
    maximum score is taken.
    """
    f1_score = 0
    em_score = 0
    total = 0

    def compute_metric_over_ground_truths(metric_fn, pred_answer, ground_truths):
        if not ground_truths:
            return [metric_fn(pred_answer, '')]

        scores = []
        for ground_truth in ground_truths:
            scores.append(metric_fn(pred_answer, ground_truth))

        return scores

    for qid, pred_answer in answer_dict.items():
        ground_truths = eval_dict[qid]['answers']

        f1_score += max(compute_metric_over_ground_truths(compute_f1, pred_answer, ground_truths))
        em_score += max(compute_metric_over_ground_truths(compute_exact, pred_answer, ground_truths))

        total += 1

    return {'EM': 100. * em_score / total,
            'F1': 100. * f1_score / total}


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(s):
    if not s: return []
    return normalize_answer(s).split()


def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1
