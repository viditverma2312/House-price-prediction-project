import pickle
import json
import numpy as np
import warnings
warnings.filterwarnings('ignore')

__all_locations = None
__data_columns = None
__model = None


def location_names():
    return __all_locations


def predict_price(location, sqft, bhk, bath):
    try:
        loc_index = __data_columns.index(location.lower())
    except:
        loc_index = -1

    x = np.zeros(len(__data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    return round(__model.predict([x])[0], 2)


def load_saved_model_data():
    print('Load saved Model')
    global __data_columns
    global __all_locations

    with open('./model data/columns_of_prediction_model.json', 'r') as f:
        __data_columns = json.load(f)['data_columns']
        __all_locations = __data_columns[3:]


    global __model
    with open('./model data/house_price_prediction_model.pickle', 'rb') as f:
        __model = pickle.load(f)

    print('Loading done')


if __name__ == '__main__':
    load_saved_model_data()
    print(location_names())
    print(predict_price('Whitefield', 1000, 3, 3))
    print(predict_price('Rajaji Nagar', 1000, 2, 2))
