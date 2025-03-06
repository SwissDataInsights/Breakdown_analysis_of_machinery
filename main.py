import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_recall_curve, average_precision_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import ADASYN
import tensorflow as tf
from tensorflow.keras.models import Sequential, clone_model
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.isotonic import IsotonicRegression
from collections import Counter
import tensorflow.keras.backend as K

# ---------------------------
# 1. Data Generation and Extended Feature Engineering
# ---------------------------
np.random.seed(42)
n = 100000  # number of observations

# Basic features
production_time = np.random.normal(loc=60, scale=10, size=n)
units_produced = np.random.poisson(lam=50, size=n)

# Additional features
temperature = np.random.normal(loc=75, scale=5, size=n)
humidity = np.random.uniform(low=30, high=70, size=n)
machine_age = np.random.exponential(scale=5, size=n)

# Function to compute failure probability based on features
def calculate_failure_probability(pt, up, temp, hum, age):
    coef_pt = -0.05    # lower production time -> higher risk
    coef_up = -0.05    # fewer produced units -> higher risk
    coef_temp = -0.03  # lower temperature -> higher risk
    coef_hum = 0.03    # higher humidity -> higher risk
    coef_age = 0.1     # older machine -> higher risk
    intercept = 3.0
    logit = intercept + coef_pt * pt + coef_up * up + coef_temp * temp + coef_hum * hum + coef_age * age
    prob = 1 / (1 + np.exp(-logit))
    return prob

failure_prob = calculate_failure_probability(production_time, units_produced, temperature, humidity, machine_age)
failures = (np.random.rand(n) < failure_prob).astype(int)

# Create DataFrame
data = pd.DataFrame({
    'production_time': production_time,
    'units_produced': units_produced,
    'temperature': temperature,
    'humidity': humidity,
    'machine_age': machine_age,
    'failure': failures
})

# Extended feature engineering: create new features (interaction, ratios, squares, and log transforms)
data['prod_units'] = data['production_time'] * data['units_produced']
data['temp_hum_ratio'] = data['temperature'] / data['humidity']
data['prod_time_sq'] = data['production_time'] ** 2
data['machine_age_sq'] = data['machine_age'] ** 2
data['interaction_temp_age'] = data['temperature'] * data['machine_age']
# Logarithmic transformations (adding 1 to avoid log(0))
data['log_prod_time'] = np.log(data['production_time'] + 1)
data['log_units'] = np.log(data['units_produced'] + 1)
data['log_machine_age'] = np.log(data['machine_age'] + 1)

print("Sample data with new features:")
print(data.head())
print("\nFailure distribution in data:")
print(data['failure'].value_counts())

# Save data to CSV
data.to_csv('simulated_data.csv', index=False)
print("\nData saved to 'simulated_data.csv'.")

# ---------------------------
# 2. Splitting, Normalization, and Selective Oversampling for 'Smaller' Features
# ---------------------------
features = data.drop('failure', axis=1)
target = data['failure']

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)
print("\nClass distribution before oversampling:")
print(y_train.value_counts())

# Normalize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define "smaller features" as observations where production_time and units_produced are below their medians.
median_pt = np.median(X_train_scaled[:, 0])  # assuming production_time is the first column
median_units = np.median(X_train_scaled[:, 1])  # assuming units_produced is the second column

mask_small = (X_train_scaled[:, 0] < median_pt) & (X_train_scaled[:, 1] < median_units)
X_train_small = X_train_scaled[mask_small]
y_train_small = y_train[mask_small]

mask_large = ~mask_small
X_train_large = X_train_scaled[mask_large]
y_train_large = y_train[mask_large]

print("\nNumber of observations in 'small features' subset:", X_train_small.shape[0])
print("Number of observations in 'large features' subset:", X_train_large.shape[0])

# Oversample only the 'small features' subset using ADASYN
adasyn = ADASYN(random_state=42)
X_train_small_res, y_train_small_res = adasyn.fit_resample(X_train_small, y_train_small)
print("\nClass distribution in 'small features' subset after oversampling:")
print(pd.Series(y_train_small_res).value_counts())

# Combine the oversampled 'small features' subset with the 'large features' subset
X_train_combined = np.vstack([X_train_large, X_train_small_res])
y_train_combined = np.concatenate([y_train_large, y_train_small_res])
print("\nTotal number of observations after combining:", X_train_combined.shape[0])
print("Combined class distribution:")
print(pd.Series(y_train_combined).value_counts())

# Compute class weights for cost-sensitive learning
counter = Counter(y_train_combined)
class_weight = {0: 1.0, 1: counter[0] / counter[1]}
print("\nClass weights:", class_weight)

# ---------------------------
# 3. Define a Custom Weighted Binary Crossentropy Loss Function
# ---------------------------
def weighted_binary_crossentropy(y_true, y_pred):
    # Weight for class 0 is set to 1.0, and for class 1 we use the computed weight
    w0 = 1.0
    w1 = class_weight[1]
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    loss = - (w1 * y_true * K.log(y_pred) + w0 * (1 - y_true) * K.log(1 - y_pred))
    return K.mean(loss)

# ---------------------------
# 4. Build Ensemble Models
# ---------------------------
def create_model_1(input_dim):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss=weighted_binary_crossentropy, metrics=['accuracy'])
    return model

def create_model_2(input_dim):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss=weighted_binary_crossentropy, metrics=['accuracy'])
    return model

input_dim = X_train_combined.shape[1]
model1 = create_model_1(input_dim)
model2 = create_model_2(input_dim)

model1.summary()
model2.summary()

# Callbacks: EarlyStopping and ReduceLROnPlateau
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)

# Train Model 1
history1 = model1.fit(
    X_train_combined, y_train_combined,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    callbacks=[early_stop, reduce_lr],
    verbose=1,
    class_weight=class_weight
)

# Train Model 2
history2 = model2.fit(
    X_train_combined, y_train_combined,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    callbacks=[early_stop, reduce_lr],
    verbose=1,
    class_weight=class_weight
)

# ---------------------------
# 5. Ensemble – Averaging Predictions from Both Models
# ---------------------------
y_probs1 = model1.predict(X_test_scaled).ravel()
y_probs2 = model2.predict(X_test_scaled).ravel()
y_probs_ensemble = (y_probs1 + y_probs2) / 2

# ---------------------------
# 6. Calibration and Threshold Optimization
# ---------------------------
# Plot Precision-Recall Curve before calibration
prec, rec, _ = precision_recall_curve(y_test, y_probs_ensemble)
ap_score = average_precision_score(y_test, y_probs_ensemble)
plt.figure(figsize=(8, 6))
plt.plot(rec, prec, label=f'PR curve (AP = {ap_score:.3f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve (Ensemble - Uncalibrated)')
plt.legend()
plt.show()

# Calibrate the ensemble predictions using Isotonic Regression
calibrator = IsotonicRegression(out_of_bounds='clip')
calibrator.fit(y_probs_ensemble, y_test)
y_probs_calibrated = calibrator.predict(y_probs_ensemble)

# Plot Precision-Recall Curve after calibration
prec_cal, rec_cal, _ = precision_recall_curve(y_test, y_probs_calibrated)
ap_score_cal = average_precision_score(y_test, y_probs_calibrated)
plt.figure(figsize=(8, 6))
plt.plot(rec_cal, prec_cal, label=f'PR curve (AP = {ap_score_cal:.3f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve (Ensemble - Calibrated)')
plt.legend()
plt.show()

# Optimize decision threshold based on F1-score
thresholds = np.arange(0.1, 1.0, 0.1)
best_threshold = 0.5
best_f1 = 0.0

print("\nEvaluating different thresholds (Calibrated Ensemble):")
for t in thresholds:
    y_pred_t = (y_probs_calibrated >= t).astype(int)
    f1 = f1_score(y_test, y_pred_t, pos_label=1)
    print("Threshold: {:.1f}  F1-score: {:.3f}".format(t, f1))
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = t

print("\nOptimal threshold: {:.1f} with F1-score: {:.3f}".format(best_threshold, best_f1))
y_pred_adjusted = (y_probs_calibrated >= best_threshold).astype(int)

# Evaluate the ensemble model
print("\nClassification Report (Calibrated Ensemble, threshold {:.1f}):".format(best_threshold))
print(classification_report(y_test, y_pred_adjusted))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_adjusted))
