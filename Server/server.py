from flask import Flask, request, jsonify
import utility
app = Flask(__name__)


@app.route('/location_names', methods=['GET'])
def location_names():
    response = jsonify({
        'locations': utility.location_names()
    })

    response.headers.add('Access-Control-Allow-Origin', '*')

    return response

@app.route('/predict_house_price', methods=['POST'])
def predict_house_price():
    total_sqft = float(request.form['total_sqft'])
    location = request.form['location']
    bhk = int(request.form['bhk'])
    bath = int(request.form['bath'])

    response = jsonify({
        'predicted_price': utility.predict_price(location,total_sqft,bhk,bath)
    })
    response.headers.add('Access-Control-Allow-Origin', '*')

    return response

if __name__ == "__main__":
    print('Starting flask for house prediction')
    utility.load_saved_model_data()
    app.run()


