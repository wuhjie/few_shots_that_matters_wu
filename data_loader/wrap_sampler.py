from torch.utils.data import SequentialSampler, DataLoader, RandomSampler, Sampler

from active_learning.query_by_committee import al_pool
from active_learning.uncertainty_sampling import UncertaintySampler


def wrap_sampler(trn_batch_size, infer_batch_size, language, language_dataset):

    for split_name in ("trn_egs", "val_egs", "tst_egs"):
        egs = getattr(language_dataset, split_name)

        if len(egs) == 0:
            print(f"[WARN] {split_name} of {language} has zero egs")
        if split_name == "trn_egs":  
            # input_idses, tags_ides 
            all_history, all_predictions, lowest_score_index = al_pool(egs)
            print("performance history", all_history)
            print("all predictions: ", all_predictions)

            # TODO: replace the random sampler with uncertainty sampling
            # sampler = RandomSampler
            sampler = Sampler

            batch_size = trn_batch_size
  

        else:
            sampler = SequentialSampler
            batch_size = infer_batch_size
            
        dl = (
            DataLoader(egs, sampler=sampler(egs), batch_size=batch_size)
            if len(egs) > 0
            else None
        )    

        setattr(language_dataset, split_name, dl)

        print("batch size: ", batch_size)
    return language_dataset

