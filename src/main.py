import json
import logging
import os
import uuid
import torch
import random
import numpy as np
from absl import app
from torch.utils.data import DataLoader
import wandb
from gnn_models import BaselineAttention
from ml_collections.config_dict import ConfigDict
from ml_collections import config_flags

from tasks import RandomTask
from trainers import Trainer, TrainerConfig

os.environ["WANDB_MODE"] = "dryrun"

_CONFIG = config_flags.DEFINE_config_dict(
    'cfg',
    ConfigDict(dict(
        ckpt_path=None,
        data_dir='/tmp',
        log_prefix='test',
        run_id="",
        d_model=64,
        train_size=100_000,
        batch_size=256,
        train_steps=None,
        train_epochs=500,
        min_train_steps=50_000,
        micro_batch_accumulation=1,
        validate_every_n_steps=500,
        mixed_precision=False,
        lr=0.001,
        task_type='random',
        task_args=dict(
            n_dims=64,
            context_size=32,
            n_classes=100,
            is_classification=True,
            context_type='random',  # 'random', 'fixed'
        ),
        model_args=dict(
            num_heads=20,
        ),
        seed=42,
        early_stop=True,
        device='cuda',
        debug=False,
    ))
)


def run(cfg):
    _set_train_epochs(cfg)
    if cfg.run_id == "":
        with cfg.unlocked():
            cfg.run_id = str(uuid.uuid4())[:8]
        _save_config(cfg)
    else:
        logging.info("Resuming run_id=%s", cfg.run_id)
        cfg = _load_config(cfg.data_dir, cfg.run_id)
    _seed_all(cfg.seed)
    logging.info(f"Starting with config {cfg}")

    if cfg.task_type == 'random':
        task = RandomTask(
            n_dims=cfg.task_args.n_dims, context_size=cfg.task_args.context_size, n_classes=cfg.task_args.n_classes,
            seed=cfg.seed, dataset_size=cfg.train_size, is_classification=cfg.task_args.is_classification,
            context_type=cfg.task_args.context_type,
        )
        model = BaselineAttention(
            input_dim=task.n_dims, n_classes=task.n_classes, d_model=cfg.d_model,
            num_heads=cfg.model_args.num_heads,
        )
    else:
        raise NotImplementedError

    datasets = task.get_datasets()
    train_loader = DataLoader(datasets['train'], batch_size=cfg.batch_size, shuffle=True)
    train_loader_test = DataLoader(datasets['train'], batch_size=cfg.batch_size, shuffle=False)
    wandb_run = _config_wandb(cfg)

    ckpt_dir_path = os.path.join(cfg.data_dir, 'checkpoints', cfg.log_prefix, cfg.run_id, model.name)
    trainer = Trainer(
        model=model,
        logger=wandb_run,
        config=TrainerConfig(
            train_steps=cfg.train_steps, checkpoint_path=ckpt_dir_path, device=cfg.device, task=task,
            validate_every_n_steps=cfg.validate_every_n_steps,
            micro_batch_accumulation=cfg.micro_batch_accumulation,
            mixed_precision=cfg.mixed_precision,
            early_stop=cfg.early_stop,
            lr=cfg.lr,
        ),
    )
    trainer.train(train_dataloader=train_loader, val_dataloader=train_loader_test)
    trainer.load_best_checkpoint()
    mem_loss, mem_acc = trainer.evaluate_dataloader(train_loader_test)
    wandb_run.log(
        {f"mem_loss/{model.name}": mem_loss, f"mem_acc/{model.name}": mem_acc}, step=cfg.train_steps
    )
    logging.info(f"Finished run_id={cfg.run_id} with mem_loss={mem_loss:.4f} and mem_acc={mem_acc:.4f}")


def _seed_all(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def _set_train_epochs(cfg):
    # Assert only one of train_steps and train_epochs is set
    assert cfg.train_steps is None or cfg.train_epochs is None, "Only one of train_steps and train_epochs can be set"
    if cfg.train_steps is None:
        with cfg.unlocked():
            cfg.train_steps = max(
                (cfg.train_epochs * cfg.train_size) // cfg.batch_size,
                cfg.min_train_steps,
            )


def _save_config(cfg):
    assert cfg.run_id is not None
    cfg_dir = os.path.join(cfg.data_dir, 'configs')
    os.makedirs(cfg_dir, exist_ok=True)
    with open(os.path.join(cfg_dir, f"{cfg.run_id}.json"), 'w') as f:
        json.dump(cfg.to_dict(), f)


def _load_config(data_dir, run_id):
    cfg_dir = os.path.join(data_dir, 'configs')
    with open(os.path.join(cfg_dir, f"{run_id}.json"), 'r') as f:
        cfg = ConfigDict(json.load(f))
    return cfg


def _config_wandb(cfg):
    log_prefix = cfg.log_prefix
    debug = cfg.debug
    wandb_run = wandb.init(
        project="memorization", entity="wandb-entity", mode="disabled" if debug else None,
        config={}, name=f"{log_prefix}/{cfg.run_id}"
    )
    wandb.config.update(cfg.to_dict())
    wandb.run.save()
    return wandb_run


def main(_):
    cfg = _CONFIG.value
    run(cfg)


if __name__ == '__main__':
    app.run(main)
