### Reaction Yield Predictor Demo

This directory contains a simple reaction yield prediction demo using PyTorch and RXNFP fingerprints.

#### Conda Environment
```bash
conda env create -f environment.yml
conda activate helsinki-pre-assignment
```

#### Files
- `yield_predict.py`: Model and training loop
- `rxn_data_processing_utils.py`, `rxn_logger.py`: Data utilities and logging
- `test_organic_reactions.csv`: Example dataset

#### Run
```bash
python yield_predict.py
```

Notes
- The script downloads/loads RXNFP model weights on first run.
- Training artifacts are saved to `yield_predictor.pth` every 10 epochs.
- For custom data, adapt column names to those used in `yield_predict.py`.


