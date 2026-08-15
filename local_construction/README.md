# To train local SHPP construction policy

### Generating data (Stage 1)

To generate the validation dataset before the 1st-stage curriculum learning:
```bash
python local_construction/generate_data.py
```

### Training (Stage 1)

To train Reviser-100 with multi-distribution SHPP instances:
```bash
python local_construction/run.py --data_distribution scale --graph_size 100 --n_epochs 200
```

To train Reviser-50, 20, 10:
```bash
--graph_size 50
--graph_size 20
--graph_size 10
```

### Generating data and Training (Stage 2)

Once you obtain the revisers that have finished the 2nd-stage curriculum learning, you can use them to generate training dataset for the next reviser.

```bash
# To generate training dataset for Reviser-100:
python local_construction/generate_data_RI.py

# To fine-tune Reviser-100:
python local_construction/run.py --data_distribution scale --RI_train --graph_size 100 --lr_decay 0.99 --RI_path data/RI_train_tsp/500_RI100_seed1235.pt --load_path pretrained/Reviser-stage1/reviser_100/epoch-199.pt --n_epochs 300

# To generate training dataset for Reviser-50 with Reviser-100:
python local_construction/generate_data_RG.py --load_path pretrained/Reviser-stage2/reviser_100/epoch-299.pt --data_path data/RI_train_tsp/500_RI100_seed1235.pt --tgt_size 50 --revision_lens 100 --batch_size 50

# To fine-tune Reviser-50:
python local_construction/run.py --data_distribution scale --RI_train --graph_size 50 --lr_decay 0.99 --RI_path data/RG_train_tsp/RG50.pt --load_path pretrained/Reviser-stage1/reviser_50/epoch-199.pt --n_epochs 300
```

### Integration of data augmentation, adverisal training and new loss functions

Clustered / perturbed dataset can be generated via

```bash
python local_construction/generate_data_RI_w_rbf_soft.py # Liming's
```

Curriculum training on 2 different datasets (the 2 baselines are kept seperately)

```bash
python local_construction/run_curriculum.py \
  --RI_train  --RI_path  data/RI_train_tsp/500_RI100_seed1235.pt \
  --RI_train2 --RI_path2 data/RI_w_rbf_soft_train_tsp/500_RI_w_rbf_soft100_seed1235.pt \
  --n_epochs1 1 --n_epochs2 1 --n_epochs 10 \
  --output_dir outputs/curriculum --run_name ri_then_rbf  \
  --load_path pretrained/Reviser-stage1/reviser_100/epoch-199.pt
```

### Custom per-epoch curriculum sequences

For full control over the dataset schedule, pass `--curriculum_seq` as a string
of digits, one per epoch. `1` selects dataset 1 (`--RI_path`), `2` selects
dataset 2 (`--RI_path2`). Length must equal `--n_epochs`. When set, this flag
overrides the `--n_epochs1`/`--n_epochs2` block behavior.

```bash
python local_construction/run_curriculum.py \
  --RI_train  --RI_path  data/RI_train_tsp/500_RI100_seed1235.pt \
  --RI_train2 --RI_path2 data/RI_w_rbf_soft_train_tsp/500_RI_w_rbf_soft100_seed1235.pt \
  --curriculum_seq 1121121121 --n_epochs 10 \
  --output_dir outputs/curriculum --run_name seq_demo \
  --load_path pretrained/Reviser-stage1/reviser_100/epoch-199.pt
```

For an interleaving like `UUCUUUUCUU` (dataset 1, dataset 1, dataset 2, ...),
use `--curriculum_seq 1121121121` (7×`1` + 3×`2`).