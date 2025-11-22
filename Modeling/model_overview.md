<<<<<<< HEAD
## Summary: Flight Delay Prediction Project

### Data Preprocessing and Feature Engineering
Extensive data preparation was performed, including:
*   **Unit Conversions**: Temperature (°C to °F), precipitation/snow (mm to inches), wind speed (km/h to mph), and pressure (hPa to inHg) were converted to Imperial units.
*   **Cyclical Time Transformations**: `CRSDepTime` and `CRSArrTime` were transformed into sine and cosine components to capture their periodic nature.
*   **Missing Value Handling**: Columns with >95% missing values and redundant identifiers were dropped.
*   **Custom Feature Creation**: Engineered disruption indicators (`TotalDisruptionMinutes`, `IsSevereDisruption`, etc.), time-based features (`DayOfYear`, `IsHolidayWindow`), operational metrics (`NumDepartures`, `CongestionRatio`), rolling averages (`RouteDelayMean_7d`, `OriginDelayMean_7d`), and interaction terms (`Dist_x_Wspd`, `TempRange`). A `weatherCancellation` flag was also created.
*   **Target Transformations**: `DepDelayMinutes` was transformed using Log, Box-Cox, and Yeo-Johnson methods to address skewness.

### LightGBM Model (Regression)
*   **Objective**: Predict log-transformed `DepDelayMinutes`.
*   **Data Split**: Time-based split (train: <=2023-12-31, valid: 2024-01-01 to 2024-12-31, test: >2024-12-31).
*   **Hyperparameter Tuning**: Optuna was used to minimize MAE on the *original scale* of `DepDelayMinutes` with early stopping.
*   **Evaluation**: On the test set, achieved MAE of `11.83` minutes, RMSE of `26.24` minutes, and R² of `0.6385` (on original scale). Log space R² was `0.7811`.
*   **Insights**: Top features included wind speed, average temperature, and distance.

### Random Forest Model (Classification)
*   **Objective**: Classify `DepDel15` (flight delay >= 15 minutes).
*   **Class Imbalance**: Handled using `class_weight='balanced'`.
*   **Hyperparameter Tuning**: Optuna maximized ROC-AUC, with training on a sampled dataset for efficiency.
*   **Evaluation**: Performance was assessed using Accuracy, Recall, Precision, and ROC-AUC, supported by Confusion Matrix and ROC curve plots.

### Neural Network Model (Regression)
*   **Architecture**: Keras Sequential model with input layer, three dense hidden layers (128, 64, 32 units, ReLU activation), and a single output unit.
*   **Compilation**: Used `adam` optimizer, `mean_squared_error` loss, and monitored `mse`, `mae`.
*   **Training**: `200` epochs, `batch_size=64`, `validation_split=0.2`, with `EarlyStopping` (patience 10) on validation loss.
*   **Evaluation**: Test MAE of `0.1130` and R-squared of `0.5401` (on log-transformed target). Model saved to `/content/drive/MyDrive/weather_prediction_model.h5`.

### Logistic Regression Model (Classification)
*   **Objective**: Predict `weatherScore > 0` (weather-related delay/cancellation).
*   **Data Strategy**: Sampled 1/10th of `weatherScore = 0` rows combined with all `weatherScore != 0` rows, followed by `MinMaxScaler` for feature scaling and `SMOTE` for oversampling the minority class.
*   **Training**: `LogisticRegression` with `solver='liblinear'` and `max_iter=200`.
*   **Evaluation**: Achieved overall Accuracy of `0.8710`. For the minority class (weather delays), precision was `0.49` and recall `0.54`.

### Reproducibility
To reproduce results, ensure:
1.  **Environment Setup**: All Python libraries are installed (typically via `requirements.txt`).
2.  **Sequential Execution**: Run all notebook cells in order.
3.  **Random Seed Management**: Fixed `random_state` values are used across models and data splits for consistent outcomes.
4.  **Colab Environment**: Designed for Google Colaboratory to minimize environmental discrepancies.
=======
## Summary: Flight Delay Prediction Project

### Data Preprocessing and Feature Engineering
Extensive data preparation was performed, including:
*   **Unit Conversions**: Temperature (°C to °F), precipitation/snow (mm to inches), wind speed (km/h to mph), and pressure (hPa to inHg) were converted to Imperial units.
*   **Cyclical Time Transformations**: `CRSDepTime` and `CRSArrTime` were transformed into sine and cosine components to capture their periodic nature.
*   **Missing Value Handling**: Columns with >95% missing values and redundant identifiers were dropped.
*   **Custom Feature Creation**: Engineered disruption indicators (`TotalDisruptionMinutes`, `IsSevereDisruption`, etc.), time-based features (`DayOfYear`, `IsHolidayWindow`), operational metrics (`NumDepartures`, `CongestionRatio`), rolling averages (`RouteDelayMean_7d`, `OriginDelayMean_7d`), and interaction terms (`Dist_x_Wspd`, `TempRange`). A `weatherCancellation` flag was also created.
*   **Target Transformations**: `DepDelayMinutes` was transformed using Log, Box-Cox, and Yeo-Johnson methods to address skewness.

### LightGBM Model (Regression)
*   **Objective**: Predict log-transformed `DepDelayMinutes`.
*   **Data Split**: Time-based split (train: <=2023-12-31, valid: 2024-01-01 to 2024-12-31, test: >2024-12-31).
*   **Hyperparameter Tuning**: Optuna was used to minimize MAE on the *original scale* of `DepDelayMinutes` with early stopping.
*   **Evaluation**: On the test set, achieved MAE of `12.20` minutes, RMSE of `30.96` minutes, and R² of `-0.0433` (on original scale). Log space R² was `0.1933`.
*   **Insights**: Top features included wind speed, average temperature, and distance.

### Random Forest Model (Classification)
*   **Objective**: Classify `DepDel15` (flight delay >= 15 minutes).
*   **Class Imbalance**: Handled using `class_weight='balanced'`.
*   **Hyperparameter Tuning**: Optuna maximized ROC-AUC, with training on a sampled dataset for efficiency.
*   **Evaluation**: Performance was assessed using Accuracy, Recall, Precision, and ROC-AUC, supported by Confusion Matrix and ROC curve plots. The model achieved an accuracy score of `0.0600`, a recall score of `0.747`, a precision score of `0.319`, and a ROC-AUC score of `0.707`

### Neural Network Model (Regression)
*   **Architecture**: Keras Sequential model with input layer, three dense hidden layers (128, 64, 32 units, ReLU activation), and a single output unit.
*   **Compilation**: Used `adam` optimizer, `mean_squared_error` loss, and monitored `mse`, `mae`.
*   **Training**: `200` epochs, `batch_size=64`, `validation_split=0.2`, with `EarlyStopping` (patience 10) on validation loss.
*   **Evaluation**: Test MAE of `0.1130` and R-squared of `0.5401` (on log-transformed target). Model saved to `/content/drive/MyDrive/weather_prediction_model.h5`.

### Logistic Regression Model (Classification)
*   **Objective**: Predict `weatherScore > 0` (weather-related delay/cancellation).
*   **Data Strategy**: Sampled 1/10th of `weatherScore = 0` rows combined with all `weatherScore != 0` rows, followed by `MinMaxScaler` for feature scaling and `SMOTE` for oversampling the minority class.
*   **Training**: `LogisticRegression` with `solver='liblinear'` and `max_iter=200`.
*   **Evaluation**: Achieved overall Accuracy of `0.8710`. For the minority class (weather delays), precision was `0.49` and recall `0.54`.

### Reproducibility
To reproduce results, ensure:
1.  **Environment Setup**: All Python libraries are installed (typically via `requirements.txt`).
2.  **Sequential Execution**: Run all notebook cells in order.
3.  **Random Seed Management**: Fixed `random_state` values are used across models and data splits for consistent outcomes.
4.  **Colab Environment**: Designed for Google Colaboratory to minimize environmental discrepancies.

>>>>>>> 80db84f33d491d67e6d4ab9e7f24f07d3aa94437
