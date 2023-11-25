# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import argparse
import pickle
import torch
import json
import sys
import io
import random
import time
import numpy as np
import ipdb

from multiprocessing.pool import ThreadPool

from tqdm import tqdm, trange
from collections import OrderedDict

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

from pytorch_transformers.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_transformers.optimization import WarmupLinearSchedule
from pytorch_transformers.tokenization_bert import BertTokenizer

import blink.candidate_retrieval.utils
from blink.crossencoder.crossencoder import CrossEncoderRanker, load_crossencoder
import logging

import blink.candidate_ranking.utils as utils
import blink.biencoder.data_process as data
from blink.biencoder.zeshel_utils import DOC_PATH, WORLDS, world_to_id
from blink.common.optimizer import get_bert_optimizer
from blink.common.params import BlinkParser


logger = None


def modify(context_input, candidate_input, max_seq_length):
    new_input = []
    context_input = context_input.tolist()
    candidate_input = candidate_input.tolist()

    for i in range(len(context_input)):
        cur_input = context_input[i]
        try:
            true_length = cur_input.index(0)
            cur_input = cur_input[:true_length]
        except:
            pass
        
        cur_candidates = candidate_input[i]
        mod_input = []
        for j in range(len(cur_candidates)):
            # remove [CLS] token from candidate

            cur_candidate = cur_candidates[j][1:]
                        
            sample = cur_input + cur_candidate + list(np.zeros(max_seq_length))
            sample = sample[:max_seq_length]
            mod_input.append(sample)

        new_input.append(mod_input)

    return torch.LongTensor(new_input)


def evaluate(reranker, eval_dataloader, device, logger, context_length, zeshel=False, silent=True):
    reranker.model.eval()
    
    if silent:
        iter_ = eval_dataloader
    else:
        iter_ = tqdm(eval_dataloader, desc="Evaluation")

    results = {}

    eval_accuracy = 0.0
    nb_eval_examples = 0
    nb_eval_steps = 0

    acc = {}
    tot = {}
    world_size = len(WORLDS)
    for i in range(world_size):
        acc[i] = 0.0
        tot[i] = 0.0

    recall_at = [1,5,10,25]
    recalls = np.zeros(len(recall_at))
    cnt = 0

    logits_full = []
    labels_full = []

    for step, batch in enumerate(iter_):
        if zeshel:
            src = batch[2]
            cnt += 1
        batch = tuple(t.to(device) for t in batch)
        context_input = batch[0]
        label_input = batch[1]
        with torch.no_grad():
            eval_loss, logits = reranker(context_input, label_input, context_length)

        logits = logits.detach().cpu().numpy()
        label_ids = label_input.cpu().numpy()

        logits_full.append(logits)
        labels_full.append(label_ids)

        for logit,label in zip(logits,label_ids):
            ordered_candidate_ids = np.argsort(logit)[::-1]

            logit_rec = []
            for rec_n in recall_at:
                logit_rec.append(label in ordered_candidate_ids[:rec_n])

            recalls += np.array(logit_rec)

        tmp_eval_accuracy, eval_result = utils.accuracy(logits, label_ids)

        eval_accuracy += tmp_eval_accuracy

        nb_eval_examples += context_input.size(0)
        if zeshel:
            for i in range(context_input.size(0)):
                src_w = src[i].item()
                acc[src_w] += eval_result[i]
                tot[src_w] += 1
        nb_eval_steps += 1

    normalized_eval_accuracy = -1
    if nb_eval_examples > 0:
        normalized_eval_accuracy = eval_accuracy / nb_eval_examples
        recalls = recalls / nb_eval_examples
    if zeshel:
        macro = 0.0
        num = 0.0 
        for i in range(len(WORLDS)):
            if acc[i] > 0:
                acc[i] /= tot[i]
                macro += acc[i]
                num += 1
        if num > 0:
            logger.info("Macro accuracy: %.5f" % (macro / num))
            logger.info("Micro accuracy: %.5f" % normalized_eval_accuracy)
    else:
        if logger:
            logger.info("Eval accuracy: %.5f" % normalized_eval_accuracy)
            logger.info("Recalls @ N: {}".format(recalls))

    results["normalized_accuracy"] = normalized_eval_accuracy
    results['recall_at_n'] = list(recalls)
    return results, logits_full, labels_full


def get_optimizer(model, params):
    return get_bert_optimizer(
        [model],
        params["type_optimization"],
        params["learning_rate"],
        fp16=params.get("fp16"),
    )


def get_scheduler(params, optimizer, len_train_data, logger):
    batch_size = params["train_batch_size"]
    grad_acc = params["gradient_accumulation_steps"]
    epochs = params["num_train_epochs"]

    num_train_steps = int(len_train_data / batch_size / grad_acc) * epochs
    num_warmup_steps = int(num_train_steps * params["warmup_proportion"])

    scheduler = WarmupLinearSchedule(
        optimizer, warmup_steps=num_warmup_steps, t_total=num_train_steps,
    )
    logger.info(" Num optimization steps = %d" % num_train_steps)
    logger.info(" Num warmup steps = %d", num_warmup_steps)
    return scheduler


def main(params):
    model_output_path = params["output_path"]
    if not os.path.exists(model_output_path):
        os.makedirs(model_output_path)
    logger = utils.get_logger(params["output_path"])

    # Init model
    reranker = CrossEncoderRanker(params)
    tokenizer = reranker.tokenizer
    model = reranker.model

    # utils.save_model(model, tokenizer, model_output_path)

    device = reranker.device
    n_gpu = reranker.n_gpu

    if params["gradient_accumulation_steps"] < 1:
        raise ValueError(
            "Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                params["gradient_accumulation_steps"]
            )
        )

    # An effective batch size of `x`, when we are accumulating the gradient accross `y` batches will be achieved by having a batch size of `z = x / y`
    # args.gradient_accumulation_steps = args.gradient_accumulation_steps // n_gpu
    params["train_batch_size"] = (
        params["train_batch_size"] // params["gradient_accumulation_steps"]
    )
    train_batch_size = params["train_batch_size"]
    eval_batch_size = params["eval_batch_size"]
    grad_acc_steps = params["gradient_accumulation_steps"]

    # Fix the random seeds
    seed = params["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if reranker.n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

    max_seq_length = params["max_seq_length"]
    context_length = params["max_context_length"]
    
    max_n = 100
    
    if params["num_train_epochs"] > 0:
        fname = os.path.join(params["data_path"], "train.t7")
        train_data = torch.load(fname)
        context_input = train_data["context_vecs"]
        candidate_input = train_data["candidate_vecs"]
        label_input = train_data["labels"]
        if params["debug"]:
            context_input = context_input[:max_n]
            candidate_input = candidate_input[:max_n]
            label_input = label_input[:max_n]

        context_input = modify(context_input, candidate_input, max_seq_length)
        if params["zeshel"]:
            src_input = train_data['worlds'][:len(context_input)]
            train_tensor_data = TensorDataset(context_input, label_input, src_input)
        else:
            train_tensor_data = TensorDataset(context_input, label_input)
        train_sampler = RandomSampler(train_tensor_data)

        train_dataloader = DataLoader(
            train_tensor_data, 
            sampler=train_sampler, 
            batch_size=params["train_batch_size"]
        )
        
        optimizer = get_optimizer(model, params)
        scheduler = get_scheduler(params, optimizer, len(train_tensor_data), logger)

        model.train()

    fname = os.path.join(params["data_path"], "valid.t7")
    valid_data = torch.load(fname)
    context_input = valid_data["context_vecs"]
    candidate_input = valid_data["candidate_vecs"]
    label_input = valid_data["labels"]
    if params["debug"]:
        context_input = context_input[:max_n]
        candidate_input = candidate_input[:max_n]
        label_input = label_input[:max_n]

    context_input = modify(context_input, candidate_input, max_seq_length)
    if params["zeshel"]:
        src_input = valid_data["worlds"][:len(context_input)]
        valid_tensor_data = TensorDataset(context_input, label_input, src_input)
    else:
        valid_tensor_data = TensorDataset(context_input, label_input)
    valid_sampler = SequentialSampler(valid_tensor_data)

    valid_dataloader = DataLoader(
        valid_tensor_data, 
        sampler=valid_sampler, 
        batch_size=params["eval_batch_size"]
    )
    
    fname = os.path.join(params["data_path"], "test.t7")
    test_data = torch.load(fname)
    context_input = test_data["context_vecs"]
    candidate_input = test_data["candidate_vecs"]
    label_input = test_data["labels"]
    if params["debug"]:
        context_input = context_input[:max_n]
        candidate_input = candidate_input[:max_n]
        label_input = label_input[:max_n]

    context_input = modify(context_input, candidate_input, max_seq_length)
    if params["zeshel"]:
        src_input = test_data["worlds"][:len(context_input)]
        test_tensor_data = TensorDataset(context_input, label_input, src_input)
    else:
        test_tensor_data = TensorDataset(context_input, label_input)
    test_sampler = SequentialSampler(test_tensor_data)

    test_dataloader = DataLoader(
        test_tensor_data, 
        sampler=test_sampler, 
        batch_size=params["eval_batch_size"]
    )

    logger.info("***** Saving pre-training model *****")
    epoch_output_folder_path = os.path.join(
        model_output_path, "epoch_init"
    )
    utils.save_model(model, tokenizer, epoch_output_folder_path)

    number_of_samples_per_dataset = {}

    time_start = time.time()

    utils.write_to_file(
        os.path.join(model_output_path, "training_params.txt"), str(params)
    )

    logger.info("Starting training")
    logger.info(
        "device: {} n_gpu: {}, distributed training: {}".format(device, n_gpu, False)
    )

    best_epoch_idx = -1
    best_score = -1

    num_train_epochs = params["num_train_epochs"]

    for epoch_idx in trange(int(num_train_epochs), desc="Epoch"):
        tr_loss = 0
        results = None

        if params["silent"]:
            iter_ = train_dataloader
        else:
            iter_ = tqdm(train_dataloader, desc="Batch")

        part = 0
        for step, batch in enumerate(iter_):
            batch = tuple(t.to(device) for t in batch)
            context_input = batch[0] 
            label_input = batch[1]
            loss, _ = reranker(context_input, label_input, context_length)

            # if n_gpu > 1:
            #     loss = loss.mean() # mean() to average on multi-gpu.

            if grad_acc_steps > 1:
                loss = loss / grad_acc_steps

            tr_loss += loss.item()

            if (step + 1) % (params["print_interval"] * grad_acc_steps) == 0:
                logger.info(
                    "Step {} - epoch {} average loss: {}\n".format(
                        step,
                        epoch_idx,
                        tr_loss / (params["print_interval"] * grad_acc_steps),
                    )
                )
                tr_loss = 0

            loss.backward()

            if (step + 1) % grad_acc_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), params["max_grad_norm"]
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if (step + 1) % (params["eval_interval"] * grad_acc_steps) == 0:
                logger.info("Evaluation on the development dataset")
                results, all_logits, all_labels = evaluate(
                    reranker,
                    valid_dataloader,
                    device=device,
                    logger=logger,
                    context_length=context_length,
                    zeshel=params["zeshel"],
                    silent=params["silent"],
                )
                logger.info("***** Saving fine - tuned model *****")
                epoch_output_folder_path = os.path.join(
                    model_output_path, "epoch_{}_{}".format(epoch_idx, part)
                )
                        
                part += 1
                utils.save_model(model, tokenizer, epoch_output_folder_path)
                pickle.dump((all_logits, all_labels), open(os.path.join(epoch_output_folder_path, "dev_results.p"),'wb'))
                
                output_eval_file = os.path.join(epoch_output_folder_path, "dev_results.txt")
                with open(output_eval_file, 'w') as fp:
                    json.dump(results, fp)

                model.train()
                logger.info("\n")

        logger.info("***** Saving fine - tuned model *****")
        epoch_output_folder_path = os.path.join(
            model_output_path, "epoch_{}".format(epoch_idx)
        )
        utils.save_model(model, tokenizer, epoch_output_folder_path)
        # reranker.save(epoch_output_folder_path)

        results, all_logits, all_labels = evaluate(
            reranker,
            valid_dataloader,
            device=device,
            logger=logger,
            context_length=context_length,
            zeshel=params["zeshel"],
            silent=params["silent"],
        )

        output_eval_file = os.path.join(epoch_output_folder_path, "valid_results.txt")
        with open(output_eval_file, 'w') as fp:
            json.dump(results, fp)
            
        pickle.dump((all_logits, all_labels), open(os.path.join(epoch_output_folder_path, "valid_results.p"),'wb'))

        ls = [best_score, results["normalized_accuracy"]]
        li = [best_epoch_idx, epoch_idx]

        best_score = ls[np.argmax(ls)]
        best_epoch_idx = li[np.argmax(ls)]
        logger.info("\n")

    execution_time = (time.time() - time_start) / 60
    utils.write_to_file(
        os.path.join(model_output_path, "training_time.txt"),
        "The training took {} minutes\n".format(execution_time),
    )
    logger.info("The training took {} minutes\n".format(execution_time))

    # save the best model in the parent_dir
    logger.info("Best performance in epoch: {}".format(best_epoch_idx))
    
    if best_score != -1:      
        params["path_to_model"] = os.path.join(
            model_output_path, "epoch_{}/pytorch_model.bin".format(best_epoch_idx)
        )
        
    reranker = CrossEncoderRanker(params)
    tokenizer = reranker.tokenizer
    model = reranker.model

    epoch_output_folder_path = os.path.join(
                model_output_path, "best_epoch"
    )
    
    if not os.path.exists(epoch_output_folder_path):
        os.makedirs(epoch_output_folder_path)


    results, all_logits, all_labels = evaluate(
            reranker,
            valid_dataloader,
            device=device,
            logger=logger,
            context_length=context_length,
            zeshel=params["zeshel"],
            silent=params["silent"],
    )
    

    output_eval_file = os.path.join(epoch_output_folder_path, "valid_results.txt")
    with open(output_eval_file, 'w') as fp:
        json.dump(results, fp)

    pickle.dump((all_logits, all_labels), open(os.path.join(epoch_output_folder_path, "valid_results.p"),'wb'))
    
    results, all_logits, all_labels = evaluate(
        reranker,
        test_dataloader,
        device=device,
        logger=logger,
        context_length=context_length,
        zeshel=params["zeshel"],
        silent=params["silent"],
    )

    output_eval_file = os.path.join(epoch_output_folder_path, "test_results.txt")
    with open(output_eval_file, 'w') as fp:
        json.dump(results, fp)

    pickle.dump((all_logits, all_labels), open(os.path.join(epoch_output_folder_path, "test_results.p"),'wb'))
    

if __name__ == "__main__":
    parser = BlinkParser(add_model_args=True)
    parser.add_training_args()
    parser.add_eval_args()

    # args = argparse.Namespace(**params)
    args = parser.parse_args()
    print(args)

    params = args.__dict__
    main(params)
