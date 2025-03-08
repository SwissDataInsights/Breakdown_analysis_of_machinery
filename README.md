# Breakdown Analysis of Machinery

This project aims to analyze machine failure risk using machine learning methods and extended feature engineering. The Jupyter notebook (*breakdown analysis of machinery.ipynb*) demonstrates the entire process—from data preparation, through model building and calibration, to result interpretation.

## Table of Contents
1. [Prerequisites](#prerequisites)  
2. [Project Structure](#project-structure)  
3. [Analysis Steps Overview](#analysis-steps-overview)  
4. [Running the Project](#running-the-project)  
5. [Results](#results)  
6. [Next Steps](#next-steps)  

---

## Prerequisites

- **Python** version 3.7+ (3.8 or higher recommended).  
- A set of libraries used in the project, for example:
  - [NumPy](https://numpy.org/)
  - [pandas](https://pandas.pydata.org/)
  - [matplotlib](https://matplotlib.org/)
  - [scikit-learn](https://scikit-learn.org/stable/)
  - [imbalanced-learn (imblearn)](https://imbalanced-learn.org/)
  - [TensorFlow/Keras](https://www.tensorflow.org/)
  - [scipy.stats](https://docs.scipy.org/doc/scipy/reference/stats.html)

The easiest way to install the required libraries is to use a `requirements.txt` file (if included) or install packages manually:

```bash
pip install numpy pandas matplotlib scikit-learn imbalanced-learn tensorflow scipy
```

---

## Project Structure

A sample file structure for the project might look like this:

```
.
├── breakdown analysis of machinery.ipynb  # Main Jupyter notebook with the analysis
├── machine_downtimes_data.csv            # Data file (if using CSV data)                               # Example script for running the analysis
├── README.md                             # This README file
└── requirements.txt                      # (Optional) List of required libraries
```

---

## Analysis Steps Overview

1. **Data Loading**  
   - The notebook demonstrates how to load data from `machine_downtimes_data.csv`.  
   - The dataset includes basic machine features such as `production_time`, `units_produced`, `temperature`, `humidity`, `machine_age`, and a target variable `failure` (indicating whether a failure occurred).

2. **Extended Feature Engineering**  
   - Creation of new features, e.g., interactions between parameters, logarithmic transformations, polynomial features, etc.  
   - The goal is to enhance the signal in the data and help models detect patterns more effectively.

3. **Train/Test Split**  
   - Data is split into a training set (typically ~70–80%) and a test set (20–30%).  
   - Standard scaling (`StandardScaler`) is also applied to unify the scales of the features.

4. **Handling Imbalanced Data**  
   - Oversampling (e.g., ADASYN) is used to balance the number of observations with and without failures.  
   - A weighted loss function (cost-sensitive learning) is additionally employed during neural network training.

5. **Model Construction and Training**  
   - Two neural network models are created (with Dense layers, BatchNormalization, Dropout, etc.).  
   - Training uses callbacks (EarlyStopping, ReduceLROnPlateau) to avoid overfitting and optimize the learning process.

6. **Ensemble and Calibration**  
   - Final predictions are an average (*ensemble*) of both models.  
   - Isotonic Regression calibration and threshold optimization (based on the F1-score) improve classification quality.

7. **Analysis and Visualization of Results**  
   - Various metrics (confusion matrix, classification report, precision-recall curve, average precision) assess model performance.  
   - Additional plots (histograms with fitted normal distributions) help interpret key parameter distributions for machines with the lowest failure risk.

---

## Running the Project

1. **Clone the repository** (or download the project files):
   ```bash
   git clone <repository_url>
   cd <folder_name>
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   *(If `requirements.txt` is not provided, install the libraries manually.)*

3. **Launch Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```
   Then open the file `breakdown analysis of machinery.ipynb` in your browser.


---

## Results

- **Precision-Recall Curve, F1-score**: Metrics for evaluating how effectively the model detects failures (class 1).  
- **Classification Report**: Precision, recall, and F1-score for each class.  
- **Confusion Matrix**: Shows the number of correct and incorrect classifications.  
- **Histograms and Feature Distributions**: Visual interpretation of which parameter ranges (e.g., temperature, humidity, machine age) are most common among the lowest-risk machines.

---

## Next Steps

- **Further hyperparameter tuning** (e.g., the number of layers and neurons in the network, learning rate, oversampling parameters).  
- **Enrich the feature set** with additional data (e.g., maintenance costs, maintenance type).  
- **Use other algorithms** (XGBoost, Random Forest, etc.) and compare them with neural network models.  
- **Cross-validation** for a more reliable model evaluation.

---