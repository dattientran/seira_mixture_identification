# Identifying closely related compounds in mixtures using surface-enhanced infrared absorption spectroscopy

Code for D. Tran et al., *Identifying closely related compounds in mixtures using surface-enhanced infrared absorption
spectroscopy* (2026). The paper uses version 1.0.

## Setup

1. Install Python 3.12 or later, Jupyter and the packages in `requirements.txt`. Use the PyTorch build for your hardware.

   ```bash
   pip install -r requirements.txt
   pip install jupyterlab
   ```

2. Download the data from Zenodo ([DOI]). Put `source_data.xlsx` in `data/` and unzip `spectra.zip`
   into `data/spectra/`.
3. Start Jupyter with `jupyter lab`, open `figures.ipynb` and run all cells. Figures are saved in `figures/`.

| File | Contents |
|---|---|
| `figures.ipynb` | Figures. |
| `train.py` | Synthetic data generation and model fitting. |
| `analyze.py` | Analyses and tables, saved to `results/source_data.xlsx`. |
| `src/` | Shared processing, models, statistics and figure analyses. |
| `models/` | Trained logistic classifiers. Combined is the mean of the 20 leave-one-mixture-out fits. |

To rebuild the workbook from spectra, run `python train.py` (about 1.5 h on GPU or 13 h on CPU), then
`python analyze.py` (about 30 min). Set `SOURCE = "results/source_data.xlsx"` in the notebook and run it again.

## Using the trained classifier

```python
import pandas as pd
from src import models, spectra

theta = pd.read_csv("models/lc_combined.csv", index_col=0).to_numpy()
x = spectra.model_input(wavenumbers, absorbance)
scores = models.lc_scores(theta, [x])[0]
```

`absorbance` holds substrate-subtracted spectra, one per row. The scores follow `spectra.COMPOUNDS`, and a compound is
present at 0.5 or higher.

## License

Code: MIT (see `LICENSE`). Data: CC BY 4.0.
