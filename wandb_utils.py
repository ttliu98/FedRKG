# -*- coding: UTF-8 -*-
import argparse
import os
from typing import Dict

import wandb


def init_wandb(conf: dict, project_name: str, experiment: str, metrics: Dict[str, str], step_metric: str):
    if metrics is None:
        metrics = {"loss": "min",
                   "accuracy": "max"}
    if 'project_name' in conf and conf.project_name is not None:
        project_name = conf.project_name
    if 'experiment' in conf and conf.experiment is not None:
        experiment = conf.experiment

    wandb.init(project=project_name, name=experiment, config=conf)
    wandb.run.log_code(".")
    wandb.define_metric(step_metric, hidden=True)
    for metric, summary in metrics.items():
        wandb.define_metric(metric, summary=summary, step_metric=step_metric)
