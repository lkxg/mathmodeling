"""Load exported arrays and pad a batch without treating missing data as padding."""
import numpy as np


def load_sample(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def collate(samples):
    import torch
    if not samples:
        raise ValueError("empty batch")
    lengths = [int(x["length"]) for x in samples]
    batch = {"lengths": torch.tensor(lengths, dtype=torch.long)}
    for key in ("text", "audio", "vision", "timestamps", "valid_mask", "time_valid_mask",
                "modality_mask", "coverage", "word_indices"):
        shape = (len(samples), max(lengths), *samples[0][key].shape[1:])
        fill = -1 if key == "word_indices" else 0
        array = np.full(shape, fill, dtype=samples[0][key].dtype)
        for i, sample in enumerate(samples):
            array[i, :lengths[i]] = sample[key]
        tensor = torch.from_numpy(array)
        batch[key] = tensor.float() if key in ("text", "audio", "vision", "coverage") else tensor
    return batch
