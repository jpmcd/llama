# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import Tuple
import os
import sys
import torch
import fire
import time
import json
import socket
import argparse
from pathlib import Path

from torch.utils.data import DataLoader
import datasets
from fairscale.nn.model_parallel.initialize import initialize_model_parallel

from llama import ModelArgs, Transformer, Tokenizer, LLaMA
from llama.data import get_input_ids_fn, get_collater_fn


USER = os.environ["USER"]


def time_elapsed(prev, out=True):
    cur = time.time()
    elapsed = cur - prev
    if out:
        print(f"Elapsed... {elapsed: .2f} seconds")
    return cur


def setup_model_parallel() -> Tuple[int, int, int]:
    if os.environ.get('OMPI_COMM_WORLD_SIZE') is not None:
        rank = int(os.environ.get('OMPI_COMM_WORLD_RANK'))
        local_rank = int(os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK'))
        world_size = int(os.environ.get('OMPI_COMM_WORLD_SIZE'))
    else:
        rank = int(os.environ.get("RANK", -1))
        local_rank = int(os.environ.get("LOCAL_RANK", -1))
        world_size = int(os.environ.get("WORLD_SIZE", -1))

    hostname = socket.gethostname()
    log_rank = "(RANK, LOCAL_RANK, WORLD_SIZE, HOSTNAME) : ({}, {}, {}, {})".format(
        rank,
        local_rank,
        world_size,
        socket.gethostname(),
    )
    print(log_rank)

    # torch.distributed.init_process_group("nccl")
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)
    initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)

    # seed must be the same in all processes
    torch.manual_seed(1)
    return rank, local_rank, world_size


def load(
    ckpt_dir: str,
    tokenizer_path: str,
    local_rank: int,
    world_size: int,
    max_seq_len: int,
    max_batch_size: int,
) -> LLaMA:
    start_time = time.time()
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    assert world_size == len(
        checkpoints
    ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"
    ckpt_path = checkpoints[local_rank]
    print("Loading")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(
        max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params
    )
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = Transformer(model_args)
    torch.set_default_tensor_type(torch.FloatTensor)
    model.load_state_dict(checkpoint, strict=False)

    generator = LLaMA(model, tokenizer)
    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return generator


def main(args):
    start_time = time.time()
    prev_time = start_time
    rank, local_rank, world_size = setup_model_parallel()
    if rank > 0:
        sys.stdout = open(os.devnull, "w")

    print("Loading model...")
    generator = load(
        args.checkpoint_dir, args.tokenizer_path, rank, world_size, args.max_seq_len, args.batch_size,
    )
    tokenizer = generator.tokenizer
    prev = time.time()
    print("Getting datasets...")
    squad = datasets.load_from_disk(args.dataset_path)
    subset = squad['train'].select(range(2 * args.max_eval_samples))
    prev = time_elapsed(prev)
    print("Preprocessing data and loading to gpu...")
    get_input_ids = get_input_ids_fn(tokenizer, make_targets=args.train)
    subset = subset.map(get_input_ids, remove_columns=subset.column_names)
    subset = subset.filter(lambda example: sum([len(example[column]) for column in subset.column_names]) < args.max_seq_len)
    subset = subset.select(range(args.max_eval_samples))
    n_examples = len(subset)
    dataset = list(zip(subset["input_ids"], subset["target_ids"])) if args.train else subset["input_ids"]
    collater = get_collater_fn(
        max_seq_len=args.max_seq_len,
        max_gen_len=args.max_gen_len,
        pad_id=tokenizer.pad_id,
        make_targets=args.train,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collater)
    prev = time_elapsed(prev)
    if args.train:
        print("Finetuning on examples...")
        prev = time_elapsed(prev)
    if args.generate:
        print(f"Generating stream, {n_examples} examples...")
        generator.generate(
            loader, max_gen_len=args.max_gen_len, temperature=args.temperature, top_p=args.top_p, output=False,
        )
        prev = time_elapsed(prev)


if __name__ == "__main__":
    # fire.Fire(main_generate)
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--checkpoint_dir", type=str)
    parser.add_argument("--tokenizer_path", type=str)
    parser.add_argument("--dataset_path", type=str)
    parser.add_argument("--num_epochs", type=int)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_train_samples", type=int)
    parser.add_argument("--max_eval_samples", type=int)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--max_gen_len", type=int, default=256)
    args = parser.parse_args()
    main(args)
