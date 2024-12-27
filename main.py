import joblib
import pandas as pd

def check_transaction(amount, time, category):
    # Load the trained model
    model = joblib.load('fraud_model.pkl')

    # Simulate a new transaction
    transaction = pd.DataFrame({
        'Amount': [amount],
        'Time': [time],
        'Category': [category]
    })

    # Predict whether the transaction is fraudulent
    prediction = model.predict(transaction)

    # Output the result
    if prediction[0] == 1:
        print("Fraudulent transaction detected!")
    else:
        print("Transaction is legitimate.")

if __name__ == "__main__":
    # Simulate a fraudulent transaction
    check_transaction(amount=2000, time=5, category=2)  # You can change these values for testing

