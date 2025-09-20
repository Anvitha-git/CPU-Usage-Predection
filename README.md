---

## 10. Example Output

After running the pipeline, you can view your metrics in the terminal:

```
type metrics.json
type eval.json
```

Sample output:

**metrics.json** (training metrics)
```json
{
   "mae": 0.008293341766947319,
   "rmse": 0.024439013641369665,
   "r2": 0.8583929309676535
}
```

**eval.json** (evaluation metrics)
```json
{
   "MAE": 0.008631927519482186,
   "RMSE": 0.027657911892549097,
   "R2": 0.7968982110132168
}
```

Your results may vary depending on your data and model configuration.

# CPU Usage Prediction Pipeline

This project predicts CPU usage for Kubernetes workloads using a machine learning pipeline managed by DVC (Data Version Control). It is designed for reproducible data science and model training.

---

## 1. Project Structure & File Roles

**Folders & Files:**

- `src/preprocess.py`: Loads raw data, cleans and selects features, splits into train/test sets, and saves them to `data/processed/`.
- `src/train.py`: Loads processed training data, one-hot encodes categorical features, trains a RandomForestRegressor, saves the trained model (`model.pkl`) and training metrics (`metrics.json`).
- `src/evaluate.py`: Loads processed test data and the trained model, aligns features, predicts CPU usage, computes evaluation metrics (MAE, RMSE, R2), and saves them to `eval.json`.
- `data/`: Contains raw data (`data-*.csv`) and processed data (`processed/train.csv`, `processed/test.csv`).
- `model.pkl`: The trained machine learning model (RandomForestRegressor).
- `metrics.json`: Metrics from training (on train set).
- `eval.json`: Metrics from evaluation (on test set).
- `dvc.yaml`, `dvc.lock`: DVC pipeline configuration and state tracking.
- `requirements.txt`: List of required Python packages.

---

## 2. Step-by-Step Usage Instructions

### Step 1: Install Python & Dependencies

Make sure you have Python 3.8 or newer installed. Then, install all required packages:

```powershell
pip install -r requirements.txt
```

### Step 2: Prepare Data

Place your raw CSV data in the `data/` folder. The default file is `data/data-1754297123597.csv`.

### Step 3: Run the DVC Pipeline

To execute all steps (preprocessing, training, evaluation) in order, run:

```powershell
dvc repro
```

DVC will automatically detect changes and only rerun necessary stages. Outputs are tracked for reproducibility.

### Step 4: Inspect Outputs

- Processed data: `data/processed/train.csv`, `data/processed/test.csv`
- Trained model: `model.pkl`
- Training metrics: `metrics.json`
- Evaluation metrics: `eval.json`

---

## 3. DVC Pipeline Stages Explained

**preprocess**
- Command: `python src/preprocess.py`
- Inputs: Raw CSV data
- Outputs: Train/test CSVs in `data/processed/`

**train**
- Command: `python src/train.py`
- Inputs: Processed training data
- Outputs: Trained model (`model.pkl`), training metrics (`metrics.json`)

**evaluate**
- Command: `python src/evaluate.py`
- Inputs: Trained model, processed test data
- Outputs: Evaluation metrics (`eval.json`)

---

## 4. How Each Script Works

### preprocess.py
- Loads raw data
- Fills missing values in `controller_kind`
- Selects relevant features and target
- Splits into train/test sets (80/20)
- Saves processed CSVs

### train.py
- Loads processed training data
- One-hot encodes `controller_kind`
- Trains RandomForestRegressor
- Saves model and metrics

### evaluate.py
- Loads processed test data
- One-hot encodes `controller_kind` to match training
- Loads trained model
- Aligns test features to model input
- Predicts CPU usage
- Computes MAE, RMSE, R2
- Saves evaluation metrics

---

## 5. Extending the Project

- **Add Visualizations:**
   - Use matplotlib/seaborn in `train.py` or `evaluate.py` to plot feature importance, prediction vs actual, error distributions, etc.
- **Change Model:**
   - Swap out RandomForestRegressor for other models (e.g., XGBoost, LinearRegression) in `train.py`.
- **Add Features:**
   - Modify `FEATURES` lists in scripts to include more columns.
- **Track More Artifacts:**
   - Add new outputs to DVC stages in `dvc.yaml`.

---

## 6. Troubleshooting & Tips

- If you get errors about missing columns, check your raw data format and feature lists.
- If DVC skips stages, it means inputs/outputs haven't changed. Force rerun with `dvc repro --force`.
- To clean up outputs, use `dvc remove` or manually delete files.
- For reproducibility, commit your code and DVC files to git.

---

## 7. Requirements

- Python 3.8+
- pandas
- scikit-learn
- matplotlib
- seaborn
- dvc

Install all with:

```powershell
pip install -r requirements.txt
```

---

## 8. Reproducibility

All data, models, and metrics are tracked by DVC. This ensures you can reproduce results and share your pipeline with others.

---

## 9. Getting Help / Improving

For questions, improvements, or to extend the pipeline, open an issue or ask for help!
