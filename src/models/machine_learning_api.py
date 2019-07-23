from flask import Flask, request
import pandas as pd
import numpy as np
import json
import pickle
import os

model_path = os.path.join('C:\\Users\\Kiosk\\Documents\\titanic\\', 'models')
model_filepath = os.path.join(model_path, 'lr_model.pkl')

model = pickle.load(open(model_filepath, 'rb'))

columns = [ 'Age', 'Fare', 'FamilySize', 'isMother', 'isMale', 'Deck_A',
       'Deck_B', 'Deck_C', 'Deck_D', 'Deck_E', 'Deck_F', 'Deck_G', 'Deck_Z',
       'Pclass_1', 'Pclass_2', 'Pclass_3', 'Title_Lady', 'Title_Master',
       'Title_Miss', 'Title_Mr', 'Title_Mrs', 'Title_Officer', 'Title_Sir',
       'Fare_bins_Very_Low', 'Fare_bins_Low', 'Fare_bins_High',
       'Fare_bins_Very_High', 'Embarked_C', 'Embarked_Q', 'Embarked_S',
       'AgeState_Adult', 'AgeState_Child']

app = Flask(__name__)

@app.route('/api', methods=['POST'])
def make_predictions():
    data = json.dumps(request.get_json(force=True))
    df = pd.read_json(data)
    
    passenger_ids = df['PassengerId'].ravel()
    actuals = df['Survived'].ravel()
    
    X = df[columns].as_matrix().astype('float')
    
    predictions = model.predict(X)
    
    df_response = pd.DataFrame({'PassengerId': passenger_ids, 'Predictions': predictions, 'Actuals': actuals})
    
    return df_response.to_json()

if __name__ == '__main__':
    app.run(port=10001, debug=True)
