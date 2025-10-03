from flask import Flask, request, jsonify
import numpy as np
import joblib

app = Flask(__name__)

# Загружаем модель
model = joblib.load('model.pkl')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['data']
    X_new = np.array(data).reshape(-1, len(data))
    
    try:
        labels = model.predict(X_new)
        probabilities = model.predict_proba(X_new)
        return jsonify({'labels': labels.tolist(),
                        'probabilities': probabilities.tolist()})
    except Exception as e:
        return jsonify({'error' : str(e)}, 500)           


if __name__ == '__main__':
    app.run(debug=True)