
### training configuration
- in adapt_training.py
```python
config = dict(
    ptl="bert",
    model="bert-base-multilingual-cased",
    dataset_name="marc",
    experiment="debug",
    adapt_trn_languages="german",
    adapt_epochs=50,
    adapt_batch_size=32,
    adapt_lr=3e-5,
    adapt_num_shots=1,
    group_index=4,
    inference_batch_size=512,
    world="0",
    train_fast=True,
    load_ckpt=True,
    manual_seed=42,
    ckpt_path="path-to-en-ckpt",
    early_stop=True,
    early_stop_patience=10,
    train_all_params=True,
    train_classifier=True,
    train_pooler=True,  # NOTE: tagging does not use this layer
    reinit_classifier=False,
)
```
