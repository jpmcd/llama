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
from pathlib import Path

from torch.utils.data import DataLoader
import datasets
from fairscale.nn.model_parallel.initialize import initialize_model_parallel

from llama import ModelArgs, Transformer, Tokenizer, LLaMA
from llama.data import get_input_ids_fn, get_collater


USER = os.environ["USER"]


def setup_model_parallel() -> Tuple[int, int]:
    if os.environ.get("OMPI_COMM_WORLD_SIZE") is not None:
        rank = int(os.environ.get("OMPI_COMM_WORLD_RANK"))
        local_rank = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK"))
        world_size = int(os.environ.get("OMPI_COMM_WORLD_SIZE"))
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


def main_generate(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.8,
    top_p: float = 0.95,
    max_seq_len: int = 512,
    max_batch_size: int = 32,
):
    # local_rank, world_size = setup_model_parallel()
    rank, local_rank, world_size = setup_model_parallel()
    if local_rank > 0:
        sys.stdout = open(os.devnull, "w")

    generator = load(
        # ckpt_dir, tokenizer_path, local_rank, world_size, max_seq_len, max_batch_size
        ckpt_dir, tokenizer_path, rank, world_size, max_seq_len, max_batch_size
    )

    prompts = [
        # For these prompts, the expected answer is the natural continuation of the prompt
        "I believe the meaning of life is",
        "Simply put, the theory of relativity states that ",
        "Building a website can be done in 10 simple steps:\n",
        # Few shot prompts: https://huggingface.co/blog/few-shot-learning-gpt-neo-and-inference-api
        """Tweet: "I hate it when my phone battery dies."
Sentiment: Negative
###
Tweet: "My day has been 👍"
Sentiment: Positive
###
Tweet: "This is the link to the article"
Sentiment: Neutral
###
Tweet: "This new music video was incredibile"
Sentiment:""",
        """Translate English to French:

sea otter => loutre de mer

peppermint => menthe poivrée

plush girafe => girafe peluche

cheese =>""",
    ]
    results = generator.generate_from_strings(
        prompts, max_gen_len=256, temperature=temperature, top_p=top_p
    )

    for result in results:
        print(result)
        print("\n==================================\n")


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.8,
    top_p: float = 0.95,
    max_seq_len: int = 512,
    max_gen_len: int = 256,
    batch_size: int = 32,
):
    # local_rank, world_size = setup_model_parallel()
    rank, local_rank, world_size = setup_model_parallel()
    if rank > 0:
        sys.stdout = open(os.devnull, "w")

    print("Loading model...")
    generator = load(
        # ckpt_dir, tokenizer_path, local_rank, world_size, max_seq_len, max_batch_size
        ckpt_dir, tokenizer_path, rank, world_size, max_seq_len, batch_size
    )
    print("Getting datasets...")
    squad = datasets.load_from_disk(f'/home/gridsan/{USER}/languagemodels/datasets/squad')
    subset = squad['train'].select(range(8))  # 128))
    print("Preprocessing data and loading to gpu...")
    get_input_ids = get_input_ids_fn(generator.tokenizer, True)
    subset = subset.map(get_input_ids, remove_columns=subset.column_names)
    collater = get_collater(max_seq_len, max_gen_len=max_gen_len, pad_id=generator.tokenizer.pad_id)
    prompts = DataLoader(subset, batch_size=8, collate_fn=collater)
    print("Generating stream...")
    generator.generate(
        prompts, max_gen_len=max_gen_len, temperature=temperature, top_p=top_p
    )


if __name__ == "__main__":
    fire.Fire(main)
