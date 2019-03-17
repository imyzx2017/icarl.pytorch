import collections
import time

from tqdm import tqdm
import torch
from torch import nn
from torch.nn import functional as F

import params


def to_onehot(targets, n_classes):
    return torch.eye(n_classes)[targets]


def distillation_loss(logits, old_logits):
    assert len(logits) == len(old_logits)

    return sum(
        q * torch.log(g) + (1 - q) * torch.log(1 - g)
        for q, g in zip(F.sigmoid(old_logits), F.sigmoid(logits))
    )


def compute_loss(logits, targets, old_logits=None, new_idx=0):
    clf_loss = F.binary_cross_entropy_with_logits(logits, targets)

    if new_idx > 0:
        assert old_logits is not None
        distil_loss = F.binary_cross_entropy_with_logits(
            input=logits[..., :new_idx],
            target=old_logits[..., :new_idx]
        )
    else:
        distil_loss = torch.zeros(1, requires_grad=False)

    return clf_loss, distil_loss


def train_task(model, train_loader, task, old_logits=None):
    stats = collections.defaultdict(float)

    optimizer = torch.optim.SGD(model.parameters(), lr=2.)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_scheduler(params.LR))

    for epoch in range(params.EPOCHS_PER_TASK):
        print(f"Epoch {epoch}.")

        lr_scheduler.step()

        cx = 0
        for inputs, targets in tqdm(train_loader):
            inputs, targets = inputs.to(params.DEVICE), targets.to(params.DEVICE)

            onehot_targets = to_onehot(targets, n_classes=task + params.TASK_SIZE)

            logits = model(inputs)
            if old_logits is None:
                previous_logits = None
            else:
                previous_logits = old_logits[cx:cx+inputs.shape[0]]

            clf_loss, distil_loss = compute_loss(
                logits,
                onehot_targets,
                old_logits=previous_logits,
                new_idx=task
            )
            loss = clf_loss + distil_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            stats["classification loss"] += clf_loss.item()
            stats["distillation loss"] += distil_loss.item()
            cx += inputs.shape[0]

    return stats


def _test_task(model, loader):
    acc = 0
    c = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(params.DEVICE), targets.to(params.DEVICE)

        preds = model.classify(inputs)
        acc += (preds == targets.numpy()).sum()
        c += len(preds)

    return acc / c

def test_all_tasks(model, loader, high_range):
    acc_per_task = {}

    model.eval()
    for task in range(0, high_range, params.TASK_SIZE):
        loader.dataset.set_classes_range(task, task + params.TASK_SIZE)
        acc_per_task[task] = _test_task(model, loader)
    model.train()

    return acc_per_task


def compute_logits(model, loader):
    logits = []

    model.eval()
    for inputs, _ in loader:
        inputs = inputs.to(params.DEVICE)
        logits.append(model(inputs))
    model.train()

    return torch.cat(logits).detach()


def timer(func):
    def _timer(*args, **kwargs):
        print("Doing <{}>...".format(func.__name__), end=" ")
        tic = time.time()
        res = func(*args, **kwargs)
        print("Done in {:.2f}".format(time.time() - tic))

        return res
    return _timer


def get_scheduler(scheduling):
    def schedule(epoch):
        chosen_lr = -1.
        for e, lr in scheduling:
            if e < epoch:
                break

            chosen_lr = lr

        return chosen_lr
    return schedule
