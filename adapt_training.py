from adapt_parameters import get_args
from future.adapt_trainer import AdaptTuner
from future.modules import ptl2classes
from future.hooks import EvaluationRecorder, LearningCurveRecorder

import data_loader.data_configs as data_configs
from future.collocate_fns import task2collocate_fn
from data_loader.wrap_sampler import wrap_sampler
import utils.checkpoint as checkpoint
import utils.logging as logging
import torch
import random
import os
import numpy as np

import sys

from sampled_infos.sampled_data_loader.udpos import SampledUDPOSDataset


task2sampleddataset = {
    "udpos": SampledUDPOSDataset,
}

# early_stop_patience=10
# bert-base-multilingual-cased is from hugging face
config = dict(
    ptl="bert",
    model="bert-base-multilingual-cased",
    dataset_name="udpos",
    experiment="debug",
    adapt_trn_languages="english",
    adapt_epochs=50,
    adapt_batch_size=32,
    adapt_lr=3e-5,
    adapt_num_shots=4,
    group_index=4,
    inference_batch_size=512,
    world="0",
    train_fast=True,
    load_ckpt=True,
    manual_seed=42,
    ckpt_path="path-to-en-ckpt",
    early_stop=True,
    early_stop_patience=5,
    train_all_params=True,
    train_classifier=True,
    train_pooler=True,  # NOTE: tagging does not use this layer
    reinit_classifier=False,
)


def init_task(conf):
    raw_dataset = task2sampleddataset[conf.dataset_name](
        num_shots=conf.adapt_num_shots, group_index=conf.group_index
    )
    metric_name = raw_dataset.metrics[0]
    classes = ptl2classes[conf.ptl]
    tokenizer = classes.tokenizer.from_pretrained(conf.model)
    
    model = classes.seqtag.from_pretrained(
        conf.model, out_dim=raw_dataset.num_labels
    )


    if conf.load_ckpt:
        with open(conf.ckpt_path, "rb") as f:
            ckpt = torch.load(f, map_location=lambda storage, loc: storage)
            # bypass (ckpt["best_state_dict"]["bert.embeddings.position_ids"])
            # version mismatch
            model.load_state_dict(ckpt["best_state_dict"], strict=False)

    exp_languages = sorted(list(set(conf.adapt_trn_languages)))
    # init the project with the data_configs.py
    data_iter_cls = data_configs.task2dataiter[conf.dataset_name]
    data_iter = {}
    
    if hasattr(raw_dataset, "contents"):
        for language in exp_languages:
            data_iter[language] = data_iter_cls(
                raw_dataset=raw_dataset.contents[language],
                model=conf.model,
                tokenizer=tokenizer,
                max_seq_len=conf.max_seq_len,
                do_cache=False,
            )
    else:
        data_iter[raw_dataset.language] = data_iter_cls(
            raw_dataset=raw_dataset,
            model=conf.model,
            tokenizer=tokenizer,
            max_seq_len=conf.max_seq_len,
            do_cache=False,
        )
    collocate_batch_fn = task2collocate_fn[conf.dataset_name]
    return (model, tokenizer, data_iter, metric_name, collocate_batch_fn)


def init_hooks(conf, metric_name):
    eval_recorder = EvaluationRecorder(
        where_=os.path.join(conf.checkpoint_root, "state_dicts"),
        which_metric=metric_name,
    )
    learning_curve_recorder = LearningCurveRecorder(
        where_=os.path.join(conf.checkpoint_root, "learning_curves")
    )
    return [eval_recorder, learning_curve_recorder]


def confirm_model(conf, model):
    # reinit classifier if necessary
    if conf.reinit_classifier:
        for name, param in model.named_parameters():
            if "classifier.weight" in name:
                param.data.normal_(mean=0.0, std=0.02)
                print("[INFO] reset classifier weights.")
            if "classifier.bias" in name:
                print("[INFO] reset classifier bias.")
                param.data.zero_()

    # lets turn off the grad for all first
    for name, param in model.named_parameters():
        param.requires_grad = False

    # if train classifier layer
    if conf.train_classifier:
        for name, param in model.named_parameters():
            if "classifier" in name:
                param.requires_grad = True

    # if train pooler layer, only use pooler in MARC in previous project
    if conf.train_pooler:
        for name, param in model.named_parameters():
            if "bert.pooler" in name:
                param.requires_grad = True

    # if train all, turn on everything
    if conf.train_all_params:
        for name, param in model.named_parameters():
            param.requires_grad = True

    for name, param in model.named_parameters():
        print(name, param.requires_grad)

    return model


def main(conf):
    if conf.override:
        for name, value in config.items():
            assert type(getattr(conf, name)) == type(value), f"{name} {value}"
            setattr(conf, name, value)

    init_config(conf)

    # init model
    model, tokenizer, data_iter, metric_name, collocate_batch_fn = init_task(conf)

    # adding uncertainty sampling/query by committe
    model = confirm_model(conf, model)
    adapt_loaders = {}

    for language, language_dataset in data_iter.items():
        print("language dataset: ", language_dataset)

        adapt_loaders[language] = wrap_sampler(
        trn_batch_size=conf.adapt_batch_size,
        infer_batch_size=conf.inference_batch_size,
        language=language,
        language_dataset=language_dataset,
        )
    

    hooks = init_hooks(conf, metric_name)

    conf.logger.log("Initialized tasks, recorders, and initing the trainer.")

    trainer = AdaptTuner(
        conf, collocate_batch_fn=collocate_batch_fn, logger=conf.logger
    )

    conf.logger.log("Starting training/validation.")
    trainer.train(
        model,
        tokenizer=tokenizer,
        data_iter=data_iter,
        metric_name=metric_name,
        adapt_loaders=adapt_loaders,
        hooks=hooks,
    )

    # update the status.
    conf.logger.log("Finishing training/validation.")
    conf.is_finished = True
    logging.save_arguments(conf)


def init_config(conf):
    conf.is_finished = False
    conf.task = conf.dataset_name

    # device
    assert conf.world is not None, "Please specify the gpu ids."
    conf.world = (
        [int(x) for x in conf.world.split(",")]
        if "," in conf.world
        else [int(conf.world)]
    )
    conf.n_sub_process = len(conf.world)

    # re-configure batch_size if sub_process > 1.
    if conf.n_sub_process > 1:
        conf.batch_size = conf.batch_size * conf.n_sub_process

    conf.adapt_trn_languages = (
        [x for x in conf.adapt_trn_languages.split(",")]
        if "," in conf.adapt_trn_languages
        else [conf.adapt_trn_languages]
    )

    random.seed(conf.manual_seed)
    torch.manual_seed(conf.manual_seed)
    torch.cuda.manual_seed(conf.manual_seed)
    np.random.seed(conf.manual_seed)

    # configure cuda related.
    assert torch.cuda.is_available()
    torch.cuda.set_device(conf.world[0])
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    # define checkpoint for logging.
    checkpoint.init_checkpoint_adapt(conf)

    # display the arguments' info.
    logging.display_args(conf)

    # configure logger.
    conf.logger = logging.Logger(conf.checkpoint_root)


if __name__ == "__main__":
    conf = get_args()

# back to line 156
    main(conf)
