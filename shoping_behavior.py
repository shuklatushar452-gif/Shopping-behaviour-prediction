from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd

try:
    
    a = pd.read_csv("shopping.csv")

    
    gender_enc = LabelEncoder()
    location_enc = LabelEncoder()
    item_enc = LabelEncoder()

    
    a['gender_enc'] = gender_enc.fit_transform(a['Gender'])
    a['location_enc'] = location_enc.fit_transform(a['Location'])
    a['item_enc'] = item_enc.fit_transform(a['Item Purchased'])

    
    X = a[['location_enc', 'gender_enc']]
    y = a['item_enc']

    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    
    pred = model.predict(X_test)

    print("Classification Report:")
    print(classification_report(y_test, pred, target_names=item_enc.classes_))

    
    def predict_item(location, gender):
        loc = location_enc.transform([location])[0]
        gen = gender_enc.transform([gender])[0]
        encoded_pred = model.predict([[loc, gen]])[0]
        return item_enc.inverse_transform([encoded_pred])[0]

    
    print("Predicted item:", predict_item("Wisconsin", "Female"))
except Exception as e:
    print("Error:", e)