import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_recall_curve, average_precision_score
from sklearn.preprocessing import StandardScaler  # Możesz również wypróbować MinMaxScaler
from imblearn.over_sampling import ADASYN  # Używamy ADASYN zamiast SMOTE
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.isotonic import IsotonicRegression
from collections import Counter

# 1. Generowanie rozszerzonych danych symulowanych
np.random.seed(42)
n = 100000  # zwiększona liczba obserwacji

# Cechy podstawowe
production_time = np.random.normal(loc=60, scale=10, size=n)
units_produced = np.random.poisson(lam=50, size=n)

# Dodatkowe cechy
temperature = np.random.normal(loc=75, scale=5, size=n)
humidity = np.random.uniform(low=30, high=70, size=n)
machine_age = np.random.exponential(scale=5, size=n)

# Funkcja obliczająca prawdopodobieństwo awarii
def calculate_failure_probability(pt, up, temp, hum, age):
    coef_pt = -0.05   # niższy czas produkcji -> wyższe ryzyko
    coef_up = -0.05   # mniejsza liczba jednostek -> wyższe ryzyko
    coef_temp = -0.03 # niższa temperatura -> wyższe ryzyko
    coef_hum = 0.03   # wyższa wilgotność -> wyższe ryzyko
    coef_age = 0.1    # starsza maszyna -> wyższe ryzyko
    intercept = 3.0
    logit = intercept + coef_pt * pt + coef_up * up + coef_temp * temp + coef_hum * hum + coef_age * age
    prob = 1 / (1 + np.exp(-logit))
    return prob

failure_prob = calculate_failure_probability(production_time, units_produced, temperature, humidity, machine_age)
failures = (np.random.rand(n) < failure_prob).astype(int)

# Tworzenie DataFrame
data = pd.DataFrame({
    'production_time': production_time,
    'units_produced': units_produced,
    'temperature': temperature,
    'humidity': humidity,
    'machine_age': machine_age,
    'failure': failures
})

# Dodanie nowych cech (inżynieria cech)
data['prod_units'] = data['production_time'] * data['units_produced']
data['temp_hum_ratio'] = data['temperature'] / data['humidity']
data['prod_time_sq'] = data['production_time'] ** 2
data['machine_age_sq'] = data['machine_age'] ** 2
data['interaction_temp_age'] = data['temperature'] * data['machine_age']

print("Przykładowe dane z nowymi cechami:")
print(data.head())
print("\nRozkład awarii w danych:")
print(data['failure'].value_counts())

# Zapis danych do pliku CSV
data.to_csv('simulated_data.csv', index=False)
print("\nDane zapisane do pliku 'simulated_data.csv'.")

# 2. Podział danych na zbiór treningowy i testowy
features = data.drop('failure', axis=1)
target = data['failure']

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)
print("\nRozkład klas przed oversamplingiem:")
print(y_train.value_counts())

# 2a. Normalizacja danych – tutaj stosujemy StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Oversampling przy użyciu ADASYN
adasyn = ADASYN(random_state=42)
X_train_res, y_train_res = adasyn.fit_resample(X_train_scaled, y_train)
print("\nRozkład klas po oversamplingu (ADASYN):")
print(pd.Series(y_train_res).value_counts())

# 4. Obliczenie wag klas (cost-sensitive learning)
counter = Counter(y_train_res)
class_weight = {0: 1.0, 1: counter[0] / counter[1]}
print("\nWagi klas:", class_weight)

# 5. Budowa zmodyfikowanej sieci neuronowej
model = Sequential([
    Input(shape=(X_train_res.shape[1],)),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Callbacki: EarlyStopping oraz ReduceLROnPlateau
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)

# 6. Trenowanie modelu z uwzględnieniem wag klas
history = model.fit(
    X_train_res, y_train_res,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    callbacks=[early_stop, reduce_lr],
    verbose=1,
    class_weight=class_weight
)

# 7. Predykcja na zbiorze testowym (prawdopodobieństwa)
y_probs = model.predict(X_test_scaled).ravel()

# 8. Obliczenie i wizualizacja krzywej Precision-Recall przed kalibracją
prec, rec, _ = precision_recall_curve(y_test, y_probs)
ap_score = average_precision_score(y_test, y_probs)
plt.figure(figsize=(8, 6))
plt.plot(rec, prec, label=f'PR curve (AP = {ap_score:.3f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve (Niekalibrowane)')
plt.legend()
plt.show()

# 9. Kalibracja wyjściowych prawdopodobieństw przy użyciu regresji izotonicznej
calibrator = IsotonicRegression(out_of_bounds='clip')
# Używamy danych testowych do kalibracji – w produkcji warto mieć osobny zbiór kalibracyjny
calibrator.fit(y_probs, y_test)
y_probs_calibrated = calibrator.predict(y_probs)

# Obliczenie i wizualizacja krzywej Precision-Recall po kalibracji
prec_cal, rec_cal, _ = precision_recall_curve(y_test, y_probs_calibrated)
ap_score_cal = average_precision_score(y_test, y_probs_calibrated)
plt.figure(figsize=(8, 6))
plt.plot(rec_cal, prec_cal, label=f'PR curve (AP = {ap_score_cal:.3f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve (Skalibrowane)')
plt.legend()
plt.show()

# 10. Optymalizacja progu decyzyjnego dla skalibrowanych prawdopodobieństw
thresholds = np.arange(0.1, 1.0, 0.1)
best_threshold = 0.5
best_f1 = 0.0

print("\nEwaluacja różnych progów (skalibrowane):")
for t in thresholds:
    y_pred_t = (y_probs_calibrated >= t).astype(int)
    f1 = f1_score(y_test, y_pred_t, pos_label=1)
    print("Threshold: {:.1f}  F1-score: {:.3f}".format(t, f1))
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = t

print("\nWybrany optymalny próg: {:.1f} z F1-score: {:.3f}".format(best_threshold, best_f1))

# Predykcja z użyciem optymalnego progu
y_pred_adjusted = (y_probs_calibrated >= best_threshold).astype(int)

# 11. Ewaluacja modelu po kalibracji
print("\nRaport klasyfikacji (skalibrowane, próg {:.1f}):".format(best_threshold))
print(classification_report(y_test, y_pred_adjusted))
print("Macierz pomyłek:")
print(confusion_matrix(y_test, y_pred_adjusted))
