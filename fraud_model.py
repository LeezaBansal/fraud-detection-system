import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

def train_model():
    # Creating dummy data to simulate transactions (Amount, Time, Category, Fraud)
    data = pd.DataFrame({
        'Amount': [100, 2000, 150, 5000, 300, 1500, 200, 1000, 2500, 3000],
        'Time': [1, 5, 2, 8, 3, 4, 6, 7, 9, 10],
        'Category': [1, 2, 1, 3, 2, 3, 1, 2, 3, 2],
        'Fraud': [0, 1, 0, 1, 0, 1, 0, 0, 1, 0]  # 1 = Fraudulent, 0 = Legitimate
    })

    # Features and target variable
    X = data[['Amount', 'Time', 'Category']]
    y = data['Fraud']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train a Random Forest Classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save the trained model
    joblib.dump(model, 'fraud_model.pkl')
    print("Model trained and saved as fraud_model.pkl")

if __name__ == "__main__":
    train_model()

