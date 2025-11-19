Data Sources and Preprocessing
1. Where the full data comes from
The data is loaded from a CSV file named FINAL_DATASET.csv, which is located at /content/FINAL_DATASET.csv within the Colab environment. This dataset presumably contains flight and weather-related information.

2. How you accessed it
The dataset was accessed using the pandas library's read_csv function in Python. The command used was df = pd.read_csv('/content/FINAL_DATASET.csv', on_bad_lines='warn', engine='python').

on_bad_lines='warn' was used to log warnings for any malformed rows without stopping the parsing process.
engine='python' was specified for greater robustness in handling potential irregularities in the CSV file format.
3. Breakdown of how the data is structured
The data is loaded into a pandas DataFrame (df), which is a tabular data structure with rows and columns. It contains 122 columns, as indicated by Original df shape: (1894980, 122). The columns include various flight details (e.g., Year, Month, DayofMonth, DayOfWeek, Reporting_Airline, Origin, Dest, DepTime, ArrTime, Cancelled, CancellationCode, WeatherDelay) and weather-related features (e.g., tavg, tmin, tmax, prcp, snow, wdir, wspd, wpgt, pres, tsun).

4. Any preprocessing steps
The following preprocessing steps were applied to the data:

Creation of weatherCancellations: A new binary column weatherCancellations was created. It is set to 1 if a flight was cancelled (Cancelled == 1) specifically due to weather (CancellationCode == 'B'), and 0 otherwise.

Calculation of weatherDelayScore: This is the target variable for the RNN model and is calculated as follows:

Missing WeatherDelay values were filled with 0.
The weatherDelayScore was initially calculated as (WeatherDelay / max_weather_delay) * 100, normalizing the delay minutes to a 0-100 scale.
For flights with weatherCancellations == 1, the weatherDelayScore was set to 100, indicating a maximum impact.
Filtering for RNN Model: Only rows where weatherDelayScore > 0 were selected to create df_rnn, ensuring the model focuses on actual weather-impacted flights.

Feature Selection: A predefined list of excluded_columns (including IDs, date components, and other non-predictive or redundant features) was removed from the dataset to form the features list.

Categorical Feature Handling: Categorical features were identified, any missing values in them were filled with 'Missing', and then they were converted into numerical format using one-hot encoding (pd.get_dummies).

Numerical Feature Handling and Cleaning:

All features in the DataFrame X (after concatenating numerical and one-hot encoded categorical features) were coerced to numeric type, with non-numeric values converted to NaN.
Missing values (NaN) and infinite values (np.inf, -np.inf) in X were systematically replaced with the mean of their respective columns.
Zero-Variance Column Removal: Columns that had zero variance (standard deviation of 0) or entirely NaN values (after coercion and imputation) were dropped from X. This is crucial to prevent division-by-zero errors during feature scaling.
Data Splitting: The data was split into training (80%) and testing (20%) sets using train_test_split with random_state=42 for reproducibility.

Feature Scaling:

StandardScaler was applied to the numerical features in X_train_processed and X_test_processed.
MinMaxScaler was applied to the target variable y_train and y_test to scale scores into a 0-1 range, which is common for neural network outputs.
Reshaping for RNN Input: The processed feature data (X_train_processed, X_test_processed) was reshaped into 3D arrays (samples, 1, features) to match the input requirements of an LSTM layer, where timesteps was set to 1.
