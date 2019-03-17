import json

from icarl import ICarl
import params
from utils import *
from data import Dataset


train_loader = torch.utils.data.DataLoader(
    Dataset(train=True),
    num_workers=params.NUM_WORKERS,
    batch_size=params.BATCH_SIZE
)
test_loader = torch.utils.data.DataLoader(
    Dataset(train=False),
    num_workers=params.NUM_WORKERS,
    batch_size=params.BATCH_SIZE
)

model = ICarl(resnet_type="34", n_classes=params.TASK_SIZE, k=params.K)
model = model.to(params.DEVICE)

stats_per_task = {}
old_logits = None

for task in range(0, 100, params.TASK_SIZE):
    print("Task classes {} to {}".format(task, task + params.TASK_SIZE))

    train_loader.dataset.set_classes_range(task, task + params.TASK_SIZE)
    test_loader.dataset.set_classes_range(0, task + params.TASK_SIZE)

    if task > 0:
        old_logits = compute_logits(model, train_loader)

    train_loader.dataset.set_examplars(model.examplars)

    stats_per_task[task] = train_task(
        model,
        train_loader,
        test_loader,
        task,
        old_logits=old_logits
    )
    print(stats_per_task[task])

    model.reduce_examplars()
    model.build_examplars(train_loader, task, task + params.TASK_SIZE)

    acc_per_task = test_all_tasks(model, test_loader, task + params.TASK_SIZE)
    stats_per_task[task]["acc"] = acc_per_task
    print('Accuracy per task', acc_per_task)
    with open("stats.json", "w+") as f:
        json.dump(stats_per_task, f)

    if task != 100 - params.TASK_SIZE:
        print("Adding new classes...")
        model.add_n_classes(params.TASK_SIZE)


torch.save(model.state_dict(), params.SAVE_PATH)
