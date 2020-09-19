
from dubao_gia import posts_info
from dubao_gia.utils import convert_price_sell
from datetime import datetime, timedelta
from dateutil import relativedelta
import time
import random
import requests
import numpy as np
from sklearn.model_selection import ShuffleSplit
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from dateutil.relativedelta import relativedelta

#################################################################################
# PREDICT_REALESTATE_PRICE #
#################################################################################
'''
Đây là hàm chính dùng để dự đoán giá cho một vùng nhất định trong khoảng thời gian định sẵn :

       Parameter:
            - realestate_type : int
            - addresss_district : int
            - forecast_period : int

        Output:
            estimated_data: [ 
                {
                    date:
                    price_m2:
                }
            ]
            predicted_data: [
                {
                    date:
                    price_m2:
                }
            ]
            historical_data: [
                {
                    date:
                    price_m2:
                }
            ]

'''


def predict_realestate_price(realestate_type: int, address_district: int, period: int):

    posts = posts_info[(posts_info['realestate_type'] == realestate_type)
                       and (posts_info['address_district'] == address_district)]

    ####################################################
    # Drop unused columns
    ####################################################

    posts.drop(columns=['page_source', 'link', 'title', 'content', 'address_ward', 'surrounding', 'surrounding_name',
                        'surrounding_characteristics', 'interior_room', 'project', 'potential',  'lat', 'long', 'coordinate',
                        'orientation', 'post_id', 'crawled_date', 'created_at', 'price_score', 'price_score_filter', 'geom',
                        'status_price_score', 'price_score_timestamp', 'comment_id', 'comment_filter_id', 'is_suspicious',
                        'created_by', 'contact_phone', 'hash_post_key', 'number_duplicated_post', 'orientation_id', 'price_m2'])

    #####################################################
    # Preprocessing Data
    #####################################################

    posts = pre_processing_data(posts)

    ######################################################
    # Building Model
    ######################################################

    # Input and Output Declaration
    X = [posts['time_since_first_post']]
    y = [posts['price_per_m2']]

    X_train, X_test, y_train, y_test = train_test_split(X=X, y=y, degree=2)

    model = regression_model(X, y)

    ###########################
    # Equation of price line
    ###########################
    intercept = model.intercept_
    coeff = model.coef_
    ###########################

    ###########################
    # Results
    ###########################
    today = datetime.timestamp(datetime.now())
    start_date = min(X_train.)
    end_date_after_period = today + relativedelta(months=period)
    estimated_data = get_dates_and_price(coeff, intercept, start_date, today)
    predicted_data = get_dates_and_price(
        coeff, intercept, today, end_date_after_period)
    historical_data = get_sample_data(X, y)

    return dict(estimated_data=estimated_data, predicted_data=predicted_data, historical_data=historical_data)


def train_test_split(X, y, degree):

    X = np.array(X).reshape(-1, 1)
    y = np.array(y).reshape(-1, 1)

    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)

    rs = ShuffleSplit(n_splits=1, test_size=0.25,
                      train_size=0.75, random_state=42)
    for i, (train, test) in enumerate(rs.split(X, y)):
        X_train = [X[i] for i in train]
        X_test = [X[i]for i in test]
        y_train = [y[i] for i in train]
        y_test = [y[i]for i in test]
    return X_train, X_test, y_train, y_test


def regression_model(X, y):

    # Training set: train the model
    # Testing set : evaluate the accuracy of the model
    X_train, X_test, y_train, y_test = train_test_split(X, y, 2)

    model = LinearRegression()
    model.fit(X_train, y_train)

    return model


def pre_processing_data(posts):
    '''
    In preprocessing, we would eliminate na records or records with 0 or negative values 
    The criteria for elimination consideration are : 
        - area_cal (since it affect price_per_m2 value)
        - price_per_m2
    Some other features may be normalized for future improvements: 
        - address_district ===> address_district_normalized:
            ###############################################
            Some streets may have mixed with city_name 
            => normalization would differentiate the street_name only
            ###############################################
        - post_date ===> post_date_normalized
            ###############################################
            Take out the min post date to be first post 
            => Then considering time_delta for each post from that first_post 
            => This post_date_normalized is input to the Linear Regression Model (for first attempt)

    ################ Any improvement may be taken place later #######################

    '''

    # Normalize address_district for later used \
    # (if this would be one of the input for the model)
    posts = normalize_streets(posts)

    # We will call make_up_alternative to fix area if found 0 or negative
    # If area_cal is not improved, drop the record
    for x in range(0, len(posts)):
        if (posts['area_cal'].iloc[x] == 0) or (posts.loc['area_cal'].iloc[x] < 0):
            response = requests.post(url='http://localhost:5010/api/make_up_alternative', json={
                "post_id": posts['id'].iloc[x],
                "request_number_list": [2]
            }, headers={})
            posts['area_cal'].iloc[x] = response.value[0]

    posts = posts[posts['area_cal'] > 0]

    # Calculating price from price_sell
    prices = []
    posts.index = [*range(0, len(posts))]
    for x in range(0, len(posts)):
        realestate_type = posts['realestate_type'].iloc[x]
        price_sell = posts['price_sell'].iloc[x]
        area_cal = posts['area_cal'].iloc[x]
        floor = posts['floor'].iloc[x]
        price_m2 = convert_price_sell(
            realestate_type, price_sell, area_cal, floor)
        prices.append(price_m2)
    posts['price_m2'] = prices

    # Since that we just focus on price_sell of considering real estates, \
    # we ommit any real estates with
    posts = posts.dropna()
    # Reinitialize index
    posts.index = [*range(0, len(posts))]
    for x in range(0, len(posts)):
        if (posts['price_per_m2'].iloc[x] < 0):
            posts.drop(posts.index[x])

    # Calculating posts_dates
    calculate_post_dates(posts)
    return posts


def normalize_streets(posts):
    posts.index = [*range(0, len(posts))]
    streets = []
    for x in range(0, len(posts)):
        street = ""
        address_district = posts['address_district'].iloc[x]
        if (address_district == None) | (address_district == ""):
            streets.append(address_district)
        else:
            bags = address_district.split(",")
            street = bags[0]
            streets.append(street)
    posts['address_district_normalized'] = streets
    return posts


def calculate_post_dates(posts):
    # Calculating posts
    posts.index = [*range(0, len(posts))]
    first_post = min(posts['post_date'])
    posts['time_since_first_post'] = round(
        (posts['post_date'] - float(first_post))/3600/24, 2)
    posts = posts.sort_values(by="time_since_first_post", ascending=True)
    last_time = -1
    for x in range(0, len(posts['time_since_first_post'])):
        post_date = posts['post_date'].iloc[x]
        if (post_date != last_time):
            last_time = post_date
        else:
            # add in epsilon number (range from -0.99 tp 0.99) to differentiate same - date records
            post_date = post_date + random.uniform(-0.99, 0.99)
    return posts


# def get_historical_data(coeff: float, intercept: float, current_date, start_date):
#     historical_data = []
#     previous_date = start_date
#     while previous_date <= current_date:
#         historical_data.append(
#             dict(date=previous_date, price=datetime.timestamp(previous_date) * coeff + intercept))
#         previous_date += timedelta(days=1)

#     return historical_data


# def get_forecast_data(coeff: float, intercept: float, current_date, period: int):

#     end_date = start_date + relativedelta(months=period)
#     forecast_data = []
#     while current_date <= end_date:
#         forecast_data.append(dict(date=current_date, price=datetime.timestamp(
#             current_date) * coeff + intercept))

#     return forecast_data


def get_dates_and_price(coeff: float, intercept: float, start_date, end_date):
    result = []
    while start_date <= end_date:
        result.append(dict(date=start_date, price=datetime.timestamp(
            start_date) * coeff + intercept))
    return result


def get_sample_data(X, y):
    sample_data = []
    for i in range(0, len(X)):
        sample_data.append(dict(date=X[i], price=y[i]))
    return sample_data


predict_realestate_price(realestate_type=2, address_district=11, period=3)
