from torch.utils.data import SequentialSampler, DataLoader, RandomSampler

sampler = RandomSampler
DataLoader(egs, sampler=sampler(egs), batch_size=batch_size)

print(sampler)