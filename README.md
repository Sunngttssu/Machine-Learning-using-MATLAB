# Machine Learning with MATLAB

This repository contains MATLAB scripts for classic machine learning classification tasks using an Artificial Neural Network (ANN) and a Support Vector Machine (SVM) on well-known datasets.

- Scripts are located in `Assignment 2/`.
- Datasets used are either included in this repo or referenced from public sources (see instructions below).

## Contents

1. ANN for Breast Cancer Classification (`assignment_2_1.m`)
2. IRIS Dataset Feature-Based Classification (`assignment_2_2.m`)

> Note: The filenames in this repository are `assignment_2_1.m` and `assignment_2_2.m`, corresponding to the ANN Breast Cancer and IRIS SVM tasks respectively.

---

## 1. ANN for Breast Cancer Classification (`assignment_2_1.m`)

### Objective
Design, train, and evaluate an Artificial Neural Network with three hidden layers (10, 20, and 30 neurons respectively) to classify breast cancer diagnoses as either malignant or benign. The model is trained using the Stochastic Gradient Descent (SGD) algorithm, and its performance is visualized using confusion matrices for the training, validation, test, and overall sets.

### Dataset
- Name: UCI Breast Cancer Wisconsin (Diagnostic) Dataset
- File: `wdbc.data`
- Source: UCI Machine Learning Repository
  - URL: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)

### Requirements
- MATLAB
- Deep Learning Toolbox™

### How to Run
1. Download the `wdbc.data` file from the UCI link above.
2. Place `wdbc.data` under the directory:
   - `Assignment 2/breast+cancer+wisconsin+diagnostic/wdbc.data`
3. Open `assignment_2_1.m` in MATLAB.
4. Update the `filename` variable if your local path differs, or ensure the file path in the script points to the copied `wdbc.data`.
5. Run the script. The MATLAB training GUI will appear, and upon completion, figures showing the confusion matrices will be displayed.

Alternatively, to auto-download the dataset directly to the correct folder, run this in the MATLAB Command Window:

```matlab
% Create destination folder if it doesn't exist
destDir = fullfile(pwd, 'breast+cancer+wisconsin+diagnostic');
if ~exist(destDir, 'dir'); mkdir(destDir); end

% Download wdbc.data from UCI
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data';
outFile = fullfile(destDir, 'wdbc.data');
websave(outFile, url);

fprintf('Saved to: %s\n', outFile);
```

Then proceed with steps 3–5 above.

---

## 2. IRIS Dataset Feature-Based Classification (`assignment_2_2.m`)

### Objective
Build a machine learning framework to classify the three species of the IRIS dataset. Instead of using the raw features directly, the script first extracts four statistical features (mean, minimum, maximum, and standard deviation) from the measurements of each sample. A multi-class Support Vector Machine (SVM) classifier is then trained on this simplified statistical feature space.

### Dataset
- Name: IRIS Flower Dataset
- Source: CSV file included in this repository
  - Path: `Assignment 2/IRIS Flower Dataset/IRIS.csv`

### Requirements
- MATLAB
- Statistics and Machine Learning Toolbox™

### How to Run
1. Ensure `IRIS.csv` is present at `Assignment 2/IRIS Flower Dataset/IRIS.csv` (already included).
2. Open `assignment_2_2.m` in MATLAB.
3. Run the script. The Command Window will display the final classification accuracy, and two figures will be generated:
   - A confusion matrix showing the model's performance
   - A 3D scatter plot visualizing the separability of the classes in the new statistical feature space

---

## Repository Structure
```
Assignment 2/
├─ assignment_2_1.m                 # ANN for Breast Cancer (SGD, 3 hidden layers)
├─ assignment_2_2.m                 # IRIS SVM with statistical features
├─ breast+cancer+wisconsin+diagnostic/
│  └─ wdbc.data                     # (Place here; not committed by default)
└─ IRIS Flower Dataset/
   └─ IRIS.csv                      # Provided IRIS dataset CSV
```

---

## Acknowledgements
This work was supervised by Dr. Biswarup Ganguly, Assistant Professor in the Department of Electrical Engineering at NIT Silchar.

## License
This project is licensed under the MIT License.
