# UI Quality Prediction with Deep Q-Networks

This project predicts the quality of mobile app UIs using deep reinforcement learning (DQN) and computer vision. It leverages app metadata, semantic annotations, and layout features to train a model that scores UI screens.

## Features

- **Data Preprocessing:** Cleans and normalizes app metadata, parses download counts, and computes weighted ratings.
- **Feature Extraction:** Caches image, layout, and semantic features for each UI using parallel processing.
- **Model:** Implements a DuelDQN architecture (ResNet backbone + metadata) for UI score prediction.
- **Training:** Trains the DQN model on cached features with reward based on app rating and downloads.
- **Evaluation:** Correlates predicted scores with ground truth, visualizes results, and supports per-UI prediction.
- **Visualization:** Plots weighted ratings, downloads, and adjusted scores per category.

## File Structure

- `final_project_kumarsat_apurvaba.ipynb` — Main notebook with all code (preprocessing, training, evaluation, plotting).
- `dqn_model_final_1L_apurvaba_kumarsat.pth` — Trained DQN model weights.
- `app_details.csv`, `ui_details.csv` — App and UI metadata.
- `component_legend.json`, `textButton_legend.json`, `icon_legend.json` — Semantic label legends.
- `cache_features/` — Cached feature files for each UI.
- `combined/`, `semantic_annotations/` — UI images and semantic masks.

## Usage

1. **Install dependencies:**  
   - Python 3.9+, PyTorch, torchvision, pandas, numpy, matplotlib, tqdm, Pillow

2. **Prepare data:**  
   - Place CSVs and legends in the project directory.
   - Organize UI images and semantic masks in `combined/` and `semantic_annotations/`.

3. **Run notebook:**  
   - Open `final_project_kumarsat_apurvaba.ipynb` in VS Code or Jupyter.
   - Execute all cells to preprocess, cache features, train, and evaluate the model.

4. **Predict UI Score:**  
   - Use `predict_ui_score(ui_number)` to get the predicted score for a specific UI.

## Example

```python
predict_ui_score(10594)
```

## Results

- Plots and evaluation metrics are saved as PNG files.
- Model checkpoints are saved during training.

## Authors

- Kumar Sat

