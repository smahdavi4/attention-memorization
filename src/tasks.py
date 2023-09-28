from torch.nn import CrossEntropyLoss
from torch.utils.data import random_split

from task_datasets import RandomDataset, RandomWikiDataset
import torch.nn.functional as F


class Task:
    def __init__(self):
        pass



class RandomTask(Task):
    def __init__(self, n_dims, seed, n_classes, context_size, dataset_size, is_classification, context_type):
        super().__init__()
        self.n_dims = n_dims
        self.seed = seed
        self.n_classes = n_classes
        self.context_size = context_size
        self.dataset_size = dataset_size
        self.is_classification = is_classification
        self.context_type = context_type

    def get_datasets(self):
        train_dataset = RandomDataset(
            self.n_dims, self.seed, self.n_classes, self.context_size, self.dataset_size,
            is_classification=self.is_classification, context_type=self.context_type
        )
        val_dataset = None
        test_dataset = None
        return {'train': train_dataset, 'val': val_dataset, 'test': test_dataset}

    def compute_loss_acc(self, outputs, labels):
        if self.is_classification:
            logits = outputs
            first_token_logits = logits[:, 0, :]
            loss = F.cross_entropy(first_token_logits, labels)
            acc = (first_token_logits.argmax(dim=-1) == labels).float().mean()
        else:
            assert outputs.shape[-1] == 1
            first_token_outputs = outputs[:, 0, 0]
            loss = F.mse_loss(first_token_outputs, labels)
            acc = loss * 0.0  # Not defined
        return loss, acc

