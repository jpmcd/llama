import datasets
from torch.utils.data import DataLoader

def get_input_ids_fn(tokenizer, labels=False):
    def get_input_ids(example):
        text = "\n\n".join([example["context"], example["question"]])
        example["input_ids"] = tokenizer.encode(text, bos=True, eos=False)
        if labels:
            example["labels"] = tokenizer.encode(example["answers"]["text"][0], bos=False, eos=True)
        return example
    return get_input_ids

def get_collater(max_seq_len=512, max_gen_len=0, pad_id=0, make_targets=False)
    def collater(batch):
        input_ids = batch["input_ids"]
        max_target_len = 0
        if make_targets:
            assert max_gen_len == 0
            target_ids = batch["labels"]
            max_target_len = max([len(l) for l in target_ids])
        min_prompt_size = min([len(t) for t in input_ids])
        max_prompt_size = max([len(t) for t in input_ids]) + max_target_len
        total_len = min(max_seq_len, max_gen_len + max_prompt_size)
        tokens = torch.full((len(batch), total_len), pad_id).cuda().long()
        targets = None
        if make_targets:
            targets_mask = torch.full((len(batch), total_len), False, dtype=torch.bool).cuda().long()
        for k, b in enumerate(batch):
            t = b["input_ids"]
            tokens[k, :len(t)] = torch.tensor(t)
            if make_targets:
                l = b["labels"]
                targets_mask[k, len(t):len(t) + len(l)] = True 
                tokens[k, len(t):len(t) + len(l)] = torch.tensor(l)
        return min_prompt_size, tokens, targets_mask
    return collater

# Mapper should perform tokenization and ops that should only happen once.
# Collater should create Tensors and ops that can happen on the fly.
