import numpy as np
import pandas as pd
import torch
import re


def load_demographic_token_ids(vocab_path='output/vocab.csv'):
    """Load Age_ and Sex_ token IDs from vocab CSV. Returns the set of IDs after the +1 offset."""
    vocab = pd.read_csv(vocab_path)
    mask = vocab['name'].str.startswith('Age_') | vocab['name'].str.startswith('Sex_')
    return set((vocab.loc[mask, 'index'] + 1).tolist())


def get_p2i(data):
    """
    Get the patient to index mapping.
    """

    px = data[:, 0].astype('int')
    p2i = []
    j = 0
    q = px[0]
    for i, p in enumerate(px):
        if p != q:
            p2i.append([j, i - j])
            q = p
            j = i
        if i == len(px) - 1:
            # add last participant
            p2i.append([j, i - j + 1])
    return np.array(p2i)


def get_batch(ix, data, p2i, select='center', index='patient', padding='regular',
              block_size=48, device='cpu', lifestyle_augmentations=False,
              no_event_token_rate=5, cut_batch=False, demographic_token_ids=None):
    """
    Get a batch of data from the dataset. This function packs sequences in a batch and also
    inserts "no event" tokens randomly with the average rate of one every five years.

    Args:
        ix: list of indices to get data from
        data: numpy array of the dataset
        p2i: numpy array of the patient to index mapping
        select: 'left', 'right', 'random', 'mint_start', 'mint_random'
        index: 'patient', 'random'
        padding: 'regular', 'random'
        block_size: size of the block to get
        device: 'cpu' or 'cuda'
        lifestyle_augmentations: whether to perform aurmentations of lifestyle token times
        no_event_token_rate: average rate of "no event" tokens in years
        cut_batch: whether to cut the batch to the smallest size possible
        demographic_token_ids: set of Age_/Sex_ token IDs (after +1 offset), required for mint_random

    Returns:
        x: input tokens
        a: input ages
        y: target tokens
        b: target ages
    """

    mask_time = -10000.

    x = torch.tensor(np.array([p2i[int(i)] for i in ix]))
    ix = torch.tensor(np.array(ix))

    gen = torch.Generator(device='cpu')
    gen.manual_seed(ix.sum().item())  # we want some things be random, but also deterministic

    if index == 'patient':
        if select in ('left', 'mint_start', 'mint_random'):
            traj_start_idx = x[:, 0]
        elif select == 'right':
            traj_start_idx = torch.clamp(x[:, 0] + x[:, 1] - block_size - 1, 0, data.shape[0])
        elif select == 'random':
            traj_start_idx = x[:, 0] + (torch.randint(2**63-1, (len(ix),), generator=gen) % torch.clamp(x[:, 1] - block_size, 1))
            traj_start_idx = torch.clamp(traj_start_idx, 0, data.shape[0])
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    if select in ('mint_start', 'mint_random'):
        traj_lengths = x[:, 1].numpy()
        max_len = int(traj_lengths.max())
        traj_start_idx = torch.clamp(traj_start_idx, 0, data.shape[0] - max_len)
        traj_start_idx = traj_start_idx.numpy()
        batch_idx = np.arange(max_len)[None, :] + traj_start_idx[:, None]
        # mask out positions beyond each trajectory's actual length
        length_mask = np.arange(max_len)[None, :] < traj_lengths[:, None]
    else:
        traj_start_idx = torch.clamp(traj_start_idx, 0, data.shape[0] - block_size - 1)
        traj_start_idx = traj_start_idx.numpy()
        batch_idx = np.arange(block_size + 1)[None, :] + traj_start_idx[:, None]

    batch_idx = np.clip(batch_idx, 0, data.shape[0] - 1)

    mask = torch.from_numpy(data[:, 0][batch_idx].astype(np.int64))
    mask = mask == torch.tensor(data[p2i[ix.numpy()][:, 0], 0][:, None].astype(np.int64)).to(mask.dtype)
    if select in ('mint_start', 'mint_random'):
        mask = mask & torch.from_numpy(length_mask)

    tokens = torch.from_numpy(data[:, 2][batch_idx].astype(np.int64))
    ages = torch.from_numpy(data[:, 1][batch_idx].astype(np.float32))

    # augment lifestyle tokens to avoid immortality bias
    if lifestyle_augmentations:
        lifestyle_idx = (tokens >= 3) * (tokens <= 11)
        if lifestyle_idx.sum():
            #TODO maybe use the same shift for all lifestyle tokens in the trajectory?
            ages[lifestyle_idx] += torch.randint(-20*365, 365*40, (lifestyle_idx.sum(),), generator=gen).float()

    tokens = tokens.masked_fill(~mask, -1)
    ages = ages.masked_fill(~mask, mask_time)

    # insert a "no event" token every 5 years on average
    if (padding.lower() == 'none' or
            padding is None or
            no_event_token_rate == 0 or
            no_event_token_rate is None):
        pad = torch.ones(len(ix), 0)
    elif padding == 'regular':
        pad = torch.arange(0, 36525, 365.25 * no_event_token_rate) * torch.ones(len(ix), 1) + 1
    elif padding == 'random':
        pad = torch.randint(1, 36525, (len(ix), int(100 / no_event_token_rate)), generator=gen)
    else:
        raise NotImplementedError
    
    m = ages.max(1, keepdim=True).values

    # stack "no event" tokens with real tokens
    tokens = torch.hstack([tokens, torch.zeros_like(pad, dtype=torch.int)])
    ages = torch.hstack([ages, pad])

    # mask out "no event" tokens that are too far in the future (i.e. after the last real token)
    tokens = tokens.masked_fill(ages > m, -1)
    ages = ages.masked_fill(ages > m, mask_time)

    # sort everything so that things are correctly ordered about stacking
    s = torch.argsort(ages, 1)
    tokens = torch.gather(tokens, 1, s)
    ages = torch.gather(ages, 1, s)

    # a technical detail: the token 0 is reserved for padding, so we shift all tokens by one
    tokens = tokens + 1

    # cut the padded tokens if possible
    if cut_batch:
        cut_margin = torch.min(torch.sum(tokens == 0, 1))
        tokens = tokens[:, cut_margin:]
        ages = ages[:, cut_margin:]

    # cut to maintain the block size
    if tokens.shape[1] > block_size + 1:
        if select == 'mint_start':
            tokens = tokens[:, :block_size + 1]
            ages = ages[:, :block_size + 1]
        elif select == 'mint_random':
            target_len = block_size + 1
            batch_tokens = []
            batch_ages = []
            for i in range(tokens.shape[0]):
                row_tokens = tokens[i]
                row_ages = ages[i]
                # find valid (non-padding) positions
                valid = row_tokens != 0
                valid_idx = valid.nonzero(as_tuple=True)[0]
                if len(valid_idx) <= target_len:
                    batch_tokens.append(row_tokens[:target_len])
                    batch_ages.append(row_ages[:target_len])
                    continue
                # identify demographic token positions among valid tokens
                demo_mask = torch.zeros(len(valid_idx), dtype=torch.bool)
                for j, idx in enumerate(valid_idx):
                    if row_tokens[idx].item() in demographic_token_ids:
                        demo_mask[j] = True
                demo_positions = valid_idx[demo_mask]
                non_demo_positions = valid_idx[~demo_mask]
                n_demo = len(demo_positions)
                n_remaining = target_len - n_demo
                # select a random consecutive window from non-demographic tokens
                if len(non_demo_positions) <= n_remaining:
                    window_positions = non_demo_positions
                else:
                    max_start = len(non_demo_positions) - n_remaining
                    start = torch.randint(max_start + 1, (1,), generator=gen).item()
                    window_positions = non_demo_positions[start:start + n_remaining]
                # combine demographic + window, sort by age
                selected_idx = torch.cat([demo_positions, window_positions])
                selected_idx = selected_idx[torch.argsort(row_ages[selected_idx])]
                batch_tokens.append(row_tokens[selected_idx])
                batch_ages.append(row_ages[selected_idx])
            tokens = torch.stack(batch_tokens)
            ages = torch.stack(batch_ages)
        else:
            cut_margin = tokens.shape[1] - block_size - 1
            tokens = tokens[:, cut_margin:]
            ages = ages[:, cut_margin:]

    # shift by one to generate targets
    x = tokens[:, :-1]
    a = ages[:, :-1]
    y = tokens[:, 1:]
    b = ages[:, 1:]

    # if the first token is a "no event" token, mask it and the corresponding target
    x = x.masked_fill((x == 0) * (y == 1), 0)
    y = y.masked_fill(x == 0, 0)
    b = b.masked_fill(x == 0, mask_time)

    if device == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, a, y, b = [i.pin_memory().to(device, non_blocking=True) for i in [x, a, y, b]]
    else:
        x, a, y, b = x.to(device), a.to(device), y.to(device), b.to(device)
    return x, a, y, b


def shap_custom_tokenizer(s, return_offsets_mapping=True):
    """Custom tokenizers conform to a subset of the transformers API."""
    pos = 0
    offset_ranges = []
    input_ids = []
    for m in re.finditer(r"\W", s):
        start, end = m.span(0)
        offset_ranges.append((pos, start))
        input_ids.append(s[pos:start])
        pos = end
    if pos != len(s):
        offset_ranges.append((pos, len(s)))
        input_ids.append(s[pos:])
    out = {}
    out["input_ids"] = input_ids
    if return_offsets_mapping:
        out["offset_mapping"] = offset_ranges
    return out


def shap_model_creator(model, disease_ids, person_tokens_ids, person_ages, device):
    """
    Creates a pseudo model that returns only logits for specified tokens.
    Needed for SHAP values, otherwise the SHAP visualisation is too huge.
    """
    def f(ps):
        xs = []
        as_ = []

        for p in ps:
            if len(p) == 0:
                print('No tokens found??')
                raise
            p = list(map(int, p))
            new_tokens = []
            new_ages = []
            for num, (masked, value, age) in enumerate(zip(p, person_tokens_ids, person_ages)):
                if num == 0:
                    new_ages.append(age)
                    if masked == 10000:
                        new_tokens.append(2 if value == 3 else 3)
                    else:
                        new_tokens.append(value)
                else:
                    if masked != 10000 or value == 1:
                        new_ages.append(age)
                        new_tokens.append(value)

            x = (torch.tensor(new_tokens, device=device)[None, ...])
            a = (torch.tensor(new_ages, device=device)[None, ...])

            xs.append(x)
            as_.append(a)

        max_length = max([x.shape[-1] for x in xs])

        xs = [torch.nn.functional.pad(x, (max_length - x.shape[-1], 0), value=0) for x in xs]
        as_ = [torch.nn.functional.pad(x, (max_length - x.shape[-1], 0), value=-10000) for x in as_]

        x = torch.cat(xs)
        a = torch.cat(as_)

        with torch.no_grad():
            probs = model(x, a)[0][:, -1, disease_ids].detach().cpu().numpy()
        return probs

    return f
