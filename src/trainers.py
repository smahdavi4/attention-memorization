import os
import logging
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


class TrainerConfig:
    lr = 0.001
    train_steps = 50_000
    validate_every_n_steps = 200
    micro_batch_accumulation = 1
    mixed_precision = False
    checkpoint_path = None
    task = None
    device = 'cuda'
    early_stop = True

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            assert hasattr(self, k)
            self.__setattr__(k, v)
        assert self.checkpoint_path is not None


class Trainer:
    def __init__(self, model: nn.Module, logger, config: TrainerConfig):
        self.model = model
        self.logger = logger
        self.config = config
        self.device = config.device
        self.best_val_loss = None
        self.best_ckpt_file = os.path.join(self.config.checkpoint_path, 'best.ckpt')

    def train(self, train_dataloader: DataLoader, val_dataloader: DataLoader):
        self.model = self.model.to(self.device)
        grad_scaler = torch.cuda.amp.GradScaler(enabled=self.config.mixed_precision)
        optimizer, scheduler = self._get_optimizers_schedulers()
        optimizer.zero_grad()
        for step, batch in tqdm(self._infinite_data_generator(train_dataloader, self.config.train_steps), total=self.config.train_steps):
            self.model.train()
            with torch.cuda.amp.autocast(enabled=self.config.mixed_precision, dtype=torch.float16):
                micro_batch_loss, _ = self._evaluate_batch(*batch)
                backward_loss = micro_batch_loss / self.config.micro_batch_accumulation

            grad_scaler.scale(backward_loss).backward()

            if (step + 1) % self.config.micro_batch_accumulation == 0:
                grad_scaler.step(optimizer)
                grad_scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            log_dict = {}
            if step % 100 == 0:
                log_dict.update(
                    {f'train_loss/{self.model.name}': micro_batch_loss.item()}
                )
            if step % self.config.validate_every_n_steps == 0:
                val_loss, val_acc = self.evaluate_dataloader(val_dataloader)
                log_dict[f'val_loss/{self.model.name}'] = val_loss
                log_dict[f'val_acc/{self.model.name}'] = val_acc
                self._maybe_checkpoint_best_model(val_loss)
                logging.info(f"Step {step}: {log_dict}")
                if self.config.early_stop and abs(val_acc - 1.0) < 1e-6:
                    self.logger.log(
                        log_dict,
                        step=step,
                    )
                    break
            if log_dict:
                self.logger.log(
                    log_dict,
                    step=step,
                )

    @torch.no_grad()
    def evaluate_dataloader(self, dataloader):
        self.model = self.model.to(self.device)
        self.model.eval()
        losses, accs = [], []
        for x, y in dataloader:
            loss, acc = self._evaluate_batch(x, y)
            losses.extend([loss.item()] * len(x))
            accs.extend([acc.item()] * len(x))
        return sum(losses) / len(losses), sum(accs) / len(accs)

    def _evaluate_batch(self, x, y):
        with torch.cuda.amp.autocast(enabled=self.config.mixed_precision, dtype=torch.float16):
            x = x.to(self.device)
            y = y.to(self.device)
            out = self.model(x, y)
            loss, acc = self.config.task.compute_loss_acc(out, y)
        return loss, acc

    def _get_optimizers_schedulers(self):
        mini_batch_steps = self.config.train_steps // self.config.micro_batch_accumulation
        warmup_steps = mini_batch_steps // 20
        optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr)
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer=optimizer,
            schedulers=[
                optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0/100, end_factor=1.0, total_iters=warmup_steps),
                optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=mini_batch_steps - warmup_steps)
            ],
            milestones=[warmup_steps],
        )
        return optimizer, scheduler

    def _infinite_data_generator(self, dataloader, max_steps):
        steps = 0
        iterator = iter(dataloader)
        while True:
            try:
                yield steps, next(iterator)
                steps += 1
                if steps >= max_steps:
                    break
            except StopIteration:
                iterator = iter(dataloader)

    def _maybe_checkpoint_best_model(self, val_loss):
        if self.best_val_loss is None or val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            os.makedirs(self.config.checkpoint_path, exist_ok=True)
            model_state_dict = self.model.state_dict()
            torch.save(
                {'state_dict': model_state_dict},
                self.best_ckpt_file
            )

    def load_best_checkpoint(self):
        ckpt = torch.load(self.best_ckpt_file)
        self.model.load_state_dict(ckpt['state_dict'])
