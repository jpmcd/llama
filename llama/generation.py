# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import List

from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from llama.tokenizer import Tokenizer
from llama.model import Transformer


class LLaMA:
    def __init__(self, model: Transformer, tokenizer: Tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate_from_strings(
        self,
        prompts: List[str],
        max_gen_len: int,
        temperature: float = 0.8,
        top_p: float = 0.95,
    ) -> List[str]:
        bsz = len(prompts)
        params = self.model.params
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

        prompt_tokens = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts]

        min_prompt_size = min([len(t) for t in prompt_tokens])
        max_prompt_size = max([len(t) for t in prompt_tokens])

        total_len = min(params.max_seq_len, max_gen_len + max_prompt_size)

        tokens = torch.full((bsz, total_len), self.tokenizer.pad_id).cuda().long()
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t).long()
        input_text_mask = tokens != self.tokenizer.pad_id
        start_pos = min_prompt_size
        prev_pos = 0
        for cur_pos in range(start_pos, total_len):
            logits = self.model.forward_only(tokens[:, prev_pos:cur_pos], prev_pos)
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits, dim=-1)
            next_token = next_token.reshape(-1)
            # only replace token if prompt has already been generated
            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token
            prev_pos = cur_pos

        decoded = []
        for i, t in enumerate(tokens.tolist()):
            # cut to max gen len
            t = t[: len(prompt_tokens[i]) + max_gen_len]
            # cut to eos tok if any
            try:
                t = t[: t.index(self.tokenizer.eos_id)]
            except ValueError:
                pass
            decoded.append(self.tokenizer.decode(t))
        return decoded

    def generate(self,
        dataloader: DataLoader,
        max_gen_len: int,
        temperature: float = 0.8,
        top_p: float = 0.95,
        output: bool = True,
    ):
        try:
            for batch_ind, (start_pos, tokens, input_mask) in tqdm(enumerate(dataloader)):
                prev_pos = 0
                total_len = tokens.size(dim=1)
                for cur_pos in range(start_pos, total_len):
                    logits = self.model.forward_only(tokens[:, prev_pos:cur_pos], prev_pos)
                    # greedily/randomly choose next token
                    if temperature > 0:
                        probs = torch.softmax(logits / temperature, dim=-1)
                        next_token = sample_top_p(probs, top_p)
                    else:
                        next_token = torch.argmax(logits, dim=-1)
                    next_token = next_token.reshape(-1)
                    # only replace token if prompt has already been generated
                    next_token = torch.where(
                        input_mask[:, cur_pos], tokens[:, cur_pos], next_token
                    )
                    tokens[:, cur_pos] = next_token
                    prev_pos = cur_pos
                # decode batch
                if output:
                    for i, t in enumerate(tokens.tolist()):
                        # cut to eos tok if any
                        try:
                            t = t[: t.index(self.tokenizer.eos_id)]
                        except ValueError:
                            pass
                        out = self.tokenizer.decode(t)
                        print(out)
                        print("\n==================================\n")
        except:
            print("Error: batch index {}".format(batch_ind))
            raise
        return

    def train_step(self, tokens, targets, start):
        self.optimizer.zero_grad()
        logits = self.model.forward(tokens, start)
        loss = F.cross_entropy(logits, targets, ignore_index=-1)
        loss.backward()
        self.optimizer.step()
        

    def train(self,
        dataloader: DataLoader,
    ):
        self.model.train()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=0.5, betas=(0.9, 0.98), eps=1e-9,
        )
        batch_ind = 0
        try:
            for batch_ind, (start, tokens, targets_mask) in tqdm(enumerate(dataloader)):
                prev_pos = 0
                total_len = tokens.size(dim=1)
                # TODO: apply weight to each row
                for cur_pos in range(start, total_len):
                    targets = torch.where(targets_mask[:, cur_pos], tokens[:, cur_pos], -1)
                    self.train_step(tokens[:, prev_pos:cur_pos], targets, prev_pos)
                    prev_pos = cur_pos
        except:
            print("Error: batch index {}".format(batch_ind))
            raise
        return


def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token
