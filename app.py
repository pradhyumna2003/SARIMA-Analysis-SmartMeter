import firebase_admin
from firebase_admin import credentials, db
from flask import Flask, jsonify, request
from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

dbURL = os.environ.get('DB_URL')
apiJSON = os.environ.get('API_JSON')



# Replace 'path/to/serviceAccountKey.json' with the path to your service account key file
cred = credentials.Certificate(apiJSON)
firebase_admin.initialize_app(cred, {
    'databaseURL': dbURL
})

app = Flask(__name__)

def sarima_model_m(ts_data, order, seasonal_order):
    model = SARIMAX(ts_data, order=order, seasonal_order=seasonal_order)
    results = model.fit(disp=False)
    return results


@app.route('/fetch-data', methods=['GET'])
def fetch_data():
    ref = db.reference('dateDB/') # Reference to the root of your database
    data = ref.get() # Fetches all data at the root
    djson=dict(data)
    tmp_k=[]
    for key, value in djson.items():
        tmp_k.append(value)
    sarima_order_m = (1,1,1)  # Adjust the SARIMA non-seasonal order as needed
    sarima_seasonal_order_m = (1,1, 1, 12)
    sarima_results_m = sarima_model_m(tmp_k, sarima_order_m, sarima_seasonal_order_m)
    tot30=np.array(sarima_results_m.forecast(steps=30))
    re={'re':tot30.tolist()}
    return jsonify(re)




if __name__ == '__main__':
    app.run(host='0.0.0.0',port=8090,debug=True)
