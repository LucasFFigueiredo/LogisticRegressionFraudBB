import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_curve

data = {
    'resident_city': ['RJ', 'SP', 'SP', 'MG', 'RJ', 'SP', 'SP', 'RJ', 'MG', 'SP','SP', 'RJ', 'MG', 'SP', 'RJ', 'SP', 'SP', 'RJ', 'MG', 'RJ','SP', 'SP', 'RJ', 'MG', 'SP', 'RJ', 'SP', 'SP', 'RJ', 'MG'],
    'purchase_city': ['RJ', 'SP', 'RJ', 'MG', 'SP', 'SP', 'SP', 'SP', 'MG', 'SP','RJ', 'RJ', 'MG', 'SP', 'SP', 'SP', 'RJ', 'RJ', 'MG', 'SP','SP', 'SP', 'RJ', 'MG', 'SP', 'SP', 'RJ', 'SP', 'RJ', 'SP'],
    'purchase_value': [120.0, 50.0, 500.0, 30.0, 1500.0, 75.0, 200.0, 90.0, 60.0, 800.0,30.0, 150.0, 250.0, 45.0, 25.0, 50.0, 60.0, 75.0, 40.0, 20.0,100.0, 150.0, 30.0, 80.0, 70.0, 10.0, 15.0, 2000.0, 90.0, 50.0],
    'average_historical_value': [110.0, 45.0, 55.0, 32.0, 150.0, 80.0, 210.0, 95.0, 50.0, 120.0,35.0, 140.0, 230.0, 40.0, 22.0, 55.0, 58.0, 70.0, 42.0, 25.0,110.0, 130.0, 35.0, 85.0, 65.0, 12.0, 18.0, 150.0, 95.0, 55.0],
    'last_24h_transactions': [1, 1, 5, 1, 8, 1, 1, 2, 1, 4,2, 1, 1, 1, 6, 2, 5, 1, 3, 7,1, 2, 1, 1, 1, 4, 3, 9, 1, 2],
    'purchased_before': [1, 1, 0, 1, 0, 1, 1, 0, 1, 0,1, 1, 1, 1, 0, 1, 0, 1, 1, 0,1, 1, 1, 1, 1, 0, 0, 0, 1, 1],
    'purchase_time': [14, 10, 2, 11, 23, 9, 15, 22, 18, 3,12, 15, 17, 10, 2, 13, 22, 16, 11, 1,15, 14, 10, 18, 11, 2, 23, 1, 12, 13],
    'fraud': [0, 0, 1, 0, 1, 0, 0, 0, 0, 1,0, 0, 0, 0, 1, 0, 1, 0, 0, 1,0, 0, 0, 0, 0, 1, 1, 1, 0, 0]
}
df = pd.DataFrame(data)

df['historical_proportion_value'] = df['purchase_value'] / df['average_historical_value']
df['different_city'] = (df['resident_city'] != df['purchase_city']).astype(int)

df['purchase_time_sin'] = np.sin(2 * np.pi * df['purchase_time'] / 24)
df['purchase_time_cos'] = np.cos(2 * np.pi * df['purchase_time'] / 24)

df = df.drop(columns=['purchase_time'])

df_encoded = pd.get_dummies(df, columns=['resident_city', 'purchase_city'], drop_first=True)
scaler = StandardScaler()
df_encoded['purchase_value_scaled'] = scaler.fit_transform(df_encoded[['purchase_value']])

df_final = df_encoded.drop(['purchase_value', 'average_historical_value'], axis=1)
X = df_final.drop('fraud', axis=1)
y = df_final['fraud']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

log_reg = LogisticRegression(solver='liblinear', random_state=42)
log_reg.fit(X_train, y_train)

best_threshold = 0.75

print("--- Model Trained and Ready ---")
print(f"Ideal cutoff point (threshold): {best_threshold:.2f}")

# --- SYSTEM FOR FORECASTING NEW DATA ---
def predict_fraud(resident_city, purchase_city, value, average_historical_value, last_24h_transactions, purchased_before, purchase_time):
   
    new_transaction_df = pd.DataFrame([{
        'resident_city': resident_city,
        'purchase_city': purchase_city,
        'purchase_value': value,
        'average_historical_value': average_historical_value,
        'last_24h_transactions': last_24h_transactions,
        'purchased_before': purchased_before,
        'purchase_time': purchase_time
    }])

    new_transaction_df['average_historical_value'] = new_transaction_df['purchase_value'] / new_transaction_df['average_historical_value']
    new_transaction_df['different_city'] = (new_transaction_df['resident_city'] != new_transaction_df['purchase_city']).astype(int)

    new_transaction_df['purchase_time_sin'] = np.sin(2 * np.pi * new_transaction_df['purchase_time'] / 24)
    new_transaction_df['purchase_time_cos'] = np.cos(2 * np.pi * new_transaction_df['purchase_time'] / 24)
    new_transaction_df = new_transaction_df.drop(columns=['purchase_time'])


    new_transaction_df['purchase_value_scaled'] = scaler.transform(new_transaction_df[['purchase_value']])

    new_transaction_df_encoded = pd.get_dummies(new_transaction_df, columns=['resident_city', 'purchase_city'], drop_first=True)

    for col in X.columns:
        if col not in new_transaction_df_encoded.columns:
            new_transaction_df_encoded[col] = 0
            
    new_transaction_final = new_transaction_df_encoded[X.columns]

    fraud_probability = log_reg.predict_proba(new_transaction_final)[:, 1]

    if fraud_probability > best_threshold:
        return f"FRAUD ALERT! Probability of fraud: {fraud_probability[0]:.2%}"
    else:
        return f"Normal transaction. Probability of fraud: {fraud_probability[0]:.2%}"

# --- Examples ---
print("\n--- Testing the System with New Data ---")

result1 = predict_fraud(resident_city='SP', purchase_city='RJ', value=1200.0, average_historical_value=100.0, last_24h_transactions=5, purchased_before=0, purchase_time=3)
print(f"Example 1: {result1}")

result2 = predict_fraud(resident_city='RJ', purchase_city='RJ', value=85.0, average_historical_value=90.0, last_24h_transactions=1, purchased_before=1, purchase_time=16)
print(f"Example 2: {result2}")

result3 = predict_fraud(resident_city='SP', purchase_city='SP', value=4500.0, average_historical_value=4000.0, last_24h_transactions=1, purchased_before=1, purchase_time=11)
print(f"Example 3: {result3}")

result4 = predict_fraud(resident_city='MG', purchase_city='RJ', value=65.0, average_historical_value=70.0, last_24h_transactions=4, purchased_before=0, purchase_time=2)
print(f"Example 4: {result4}")

result5 = predict_fraud(resident_city='SP', purchase_city='SP', value=950.0, average_historical_value=800.0, last_24h_transactions=1, purchased_before=1, purchase_time=23)
print(f"Example 5: {result5}")

result6 = predict_fraud(resident_city='RJ', purchase_city='SP', value=50.0, average_historical_value=60.0, last_24h_transactions=1, purchased_before=0, purchase_time=14)
print(f"Example 6: {result6}")

result7 = predict_fraud(resident_city='MG', purchase_city='SP', value=3000.0, average_historical_value=100.0, last_24h_transactions=7, purchased_before=0, purchase_time=1)
print(f"Example 7: {result7}")