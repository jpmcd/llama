import datasets
import torch
from torch.utils.data import DataLoader

def get_input_ids_fn(tokenizer, make_targets=False):
    def get_input_ids(example):
        text = " ".join([example["context"], example["question"]])
        example["input_ids"] = tokenizer.encode(text, bos=True, eos=False)
        if make_targets:
            example["target_ids"] = tokenizer.encode(example["answers"]["text"][0], bos=False, eos=True)
        return example
    return get_input_ids

def get_collater_fn(max_seq_len=512, max_gen_len=256, pad_id=0, make_targets=False):
    def collater(batch, max_gen_len=max_gen_len):
        max_target_len = 0
        if make_targets:
            # TODO: this line  # assert max_gen_len == 0
            max_gen_len = 0
            input_ids = batch[0]
            target_ids = batch[1]
            max_target_len = max([len(l) for l in target_ids])
        else:
            input_ids = batch
        min_prompt_size = min([len(t) for t in input_ids])
        max_prompt_size = max([len(t) for t in input_ids]) + max_target_len
        total_len = min(max_seq_len, max_gen_len + max_prompt_size)
        tokens = torch.full((len(batch), total_len), pad_id).long()
        targets_mask = torch.full((len(batch), total_len), False, dtype=torch.bool)
        for k, t in enumerate(input_ids):
            tokens[k, :len(t)] = torch.tensor(t)
            if make_targets:
                l = target_ids[k]
                targets_mask[k, len(t):len(t) + len(l)] = True 
                tokens[k, len(t):len(t) + len(l)] = torch.tensor(l)
            else:
                targets_mask[k, :len(t)] = True
        # if not make_targets:
        #     targets_mask = tokens != pad_id
        return min_prompt_size, tokens, targets_mask
    return collater

# Mapper should perform tokenization and ops that should only happen once.
# Collater should create Tensors and ops that can happen on the fly.
