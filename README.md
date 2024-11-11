# Exp.no: 10   IMPLEMENTATION OF SARIMA MODEL
### Date:11.11.24
### NAME : M.Pranathi
### REG NO : 212222240064

### AIM:
To implement SARIMA model using python.
### ALGORITHM:
1. Explore the dataset
2. Check for stationarity of time series
3. Determine SARIMA models parameters p, q
4. Fit the SARIMA model
5. Make time series predictions and Auto-fit the SARIMA model
6. Evaluate model predictions
### PROGRAM:
```
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
import seaborn as sns

import pandas as pd

# Load the dataset
df = pd.read_csv('supermarketsales.csv')


# Convert 'Date' to datetime format with automatic format inference
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# Set 'Date' as index
df.set_index('Date', inplace=True)

# Check the first few rows to verify that the dates are parsed correctly
print(df.head())



# Ensure all columns except 'Date' are numeric and drop any non-numeric columns
df = df.apply(pd.to_numeric, errors='coerce')

# Remove duplicate dates by averaging values with the same date
df = df.groupby(df.index).mean()


if 'Total' in df.columns:
    df['Total_diff'] = df['Total'].diff()  # Calculate the difference of the 'Total' column
else:
    print("Column 'Total' not found in the dataset.")
    # You can choose to handle this error by creating a different column or skipping the analysis

# Plot the ACF of the differenced 'Total' column
if 'Total_diff' in df.columns:
    plt.figure(figsize=(10, 6))
    plot_acf(df['Total_diff'].dropna(), lags=40)
    plt.title('ACF of Differenced Total Sales')
    plt.show()
else:
    print("The 'Total_diff' column is missing.")

# Step 1: Explore the data by plotting
plt.figure(figsize=(10, 6))
plt.plot(df.index, df['Total'], label='Total Sales')
plt.xlabel('Date')
plt.ylabel('Total Sales')
plt.title('Total Sales over Time')
plt.legend()
plt.show()

# Step 2: Check for stationarity using ADF test
result = adfuller(df['Total'].dropna())  # Drop NaN values for ADF test
print(f"ADF Statistic: {result[0]}")
print(f"p-value: {result[1]}")
print(f"Critical Values: {result[4]}")

# Step 3: If non-stationary, difference the data
if result[1] > 0.05:
    df['Total_diff'] = df['Total'] - df['Total'].shift(1)
    df['Total_diff'].dropna(inplace=True)  # Drop the NaN value created by shifting
    print("Differenced data:")
    print(df['Total_diff'].head())
    
    # Plot the differenced data
    plt.figure(figsize=(10, 6))
    plt.plot(df.index[1:], df['Total_diff'], label='Differenced Total Sales')
    plt.xlabel('Date')
    plt.ylabel('Differenced Total Sales')
    plt.title('Differenced Total Sales over Time')
    plt.legend()
    plt.show()

# Step 4: Plot ACF and PACF
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plt.figure(figsize=(10, 6))
plot_acf(df['Total_diff'].dropna(), lags=40)
plt.title('ACF of Differenced Total Sales')
plt.show()

plt.figure(figsize=(10, 6))
plot_pacf(df['Total_diff'].dropna(), lags=40)
plt.title('PACF of Differenced Total Sales')
plt.show()

# Step 5: Fit ARIMA model
from statsmodels.tsa.arima.model import ARIMA

# Fit ARIMA model (use p, d, q based on the ACF and PACF plots)
model = ARIMA(df['Total'], order=(1, 1, 1))  # Example: p=1, d=1, q=1
model_fit = model.fit()

# Step 6: Make predictions
forecast = model_fit.forecast(steps=10)
print(f"Forecasted Total Sales for next 10 periods: {forecast}")

# Plot the original data and forecast
plt.figure(figsize=(10, 6))
plt.plot(df.index, df['Total'], label='Original Total Sales')
plt.plot(pd.date_range(df.index[-1], periods=11, freq='D')[1:], forecast, label='Forecasted Total Sales', color='red')
plt.xlabel('Date')
plt.ylabel('Total Sales')
plt.title('Total Sales and Forecast')
plt.legend()
plt.show()



```

### OUTPUT:

![image](https://github.com/user-attachments/assets/392ea28a-776a-4e82-8331-4ec1c908a53d)

![Screenshot 2024-11-11 190255](https://github.com/user-attachments/assets/0cbe3dc6-0854-4181-849f-34fb2e0c9cd9)

![Screenshot 2024-11-11 190323](https://github.com/user-attachments/assets/1d4c0d1d-3d86-48a7-b6a6-76c5d7a0da1b)

![Screenshot 2024-11-11 190922](https://github.com/user-attachments/assets/64ad5f71-8181-4e09-ab86-e02f0405feb7)


![Screenshot 2024-11-11 190953](https://github.com/user-attachments/assets/af7402ec-d8ea-416a-b88c-dcef66cf7d75)

![Screenshot 2024-11-11 191006](https://github.com/user-attachments/assets/60d4c7aa-7f66-43f5-a456-fb64b3377c57)






### RESULT:
Thus the program run successfully based on the SARIMA model.
