import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
import scipy.stats as stats
import statistics
#from geopy.distance import geodesic

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder



from sklearn.model_selection import train_test_split,cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
#import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error
import warnings
warnings.filterwarnings('ignore')
df_train = pd.read_csv('train.csv')
#df_train.tail()
#df_train.columns
#print("Train Dataset :", df_train.shape)
#df_train.info()
#df_train.describe().T
#df_train.describe(exclude=np.number).T
#df_train.columns[df_train.isnull().any()].tolist()
for column in df_train.columns:
    print(column)
    print(df_train[column].value_counts())
    print("------------------------------------")

#Update Column Names
def update_column_name(df):
    #Renaming Weatherconditions column
    df.rename(columns={'Weatherconditions': 'Weather_conditions'},inplace=True)
    df.rename(columns={'Time_taken(min)': 'Time_taken_in_min'},inplace=True)
    df.columns = [x.lower() for x in df.columns]


update_column_name(df_train)
#print(df_train.columns)

#df_train['delivery_person_id'].value_counts()
#Extract relevant values from column
def extract_column_value(df):
    #Extract time and convert to int
    df['time_taken_in_min'] = df['time_taken_in_min'].apply(lambda x: int(x.split(' ')[1].strip()))
    #Extract Weather conditions
    df['weather_conditions'] = df['weather_conditions'].apply(lambda x: x.split(' ')[1].strip())
    #Extract city code from Delivery person ID
    df['city_code']=df['delivery_person_id'].str.split("RES", expand=True)[0]

extract_column_value(df_train)
#df_train[['time_taken_in_min','weather_conditions','city_code']].head()

#df_train['time_taken_in_min']
#Check for Duplicate Values
if (len(df_train[df_train.duplicated()])>0):
    print("There are Duplicate values present")
else:
    print("There is no duplicate value present")


#Update datatypes
def update_datatype(df):
    df['delivery_person_age'] = df['delivery_person_age'].astype('float64')
    df['delivery_person_ratings'] = df['delivery_person_ratings'].astype('float64')
    df['multiple_deliveries'] = df['multiple_deliveries'].astype('float64')
    df['order_date']=pd.to_datetime(df['order_date'],format="%d-%m-%Y")

update_datatype(df_train)

#Convert String 'NaN' to np.nan
def convert_nan(df):
    df.replace('NaN', float(np.nan), regex=True,inplace=True)
    df.replace('nan', float(np.nan), regex=True,inplace=True)
convert_nan(df_train)


#Check null values
df_train.isnull().sum().sort_values(ascending=False)

def handle_null_values(df):
    df["delivery_person_age"] = df["delivery_person_age"].fillna(df["delivery_person_age"].astype("float").median())
    df['weather_conditions'].fillna(df['weather_conditions'].mode()[0], inplace=True)
    df['city'].fillna(df['city'].mode()[0], inplace=True)
    df['festival'].fillna(df['festival'].mode()[0], inplace=True)
    df['multiple_deliveries'].fillna(df['multiple_deliveries'].mode()[0], inplace=True)
    df['road_traffic_density'].fillna(df['road_traffic_density'].mode()[0], inplace=True)     # ====> Sub problem explore another model for prediction
    df['delivery_person_ratings'].fillna(df['delivery_person_ratings'].median(), inplace=True)
    df["delivery_person_ratings"] = df["delivery_person_ratings"].replace(6, 5)
    df.replace(np.nan,np.random.choice(df['delivery_person_age']) , regex=True,inplace=True)

handle_null_values(df_train)
#df_train.isnull().sum()

#df_train.isnull().sum()

def extract_date_features(data):
    data["day"] = data.order_date.dt.day
    data["month"] = data.order_date.dt.month
    data["quarter"] = data.order_date.dt.quarter
    data["year"] = data.order_date.dt.year
    data['day_of_week'] = data.order_date.dt.day_of_week.astype(int)
    data["is_month_start"] = data.order_date.dt.is_month_start.astype(int)
    data["is_month_end"] = data.order_date.dt.is_month_end.astype(int)
    data["is_quarter_start"] = data.order_date.dt.is_quarter_start.astype(int)
    data["is_quarter_end"] = data.order_date.dt.is_quarter_end.astype(int)
    data["is_year_start"] = data.order_date.dt.is_year_start.astype(int)
    data["is_year_end"] = data.order_date.dt.is_year_end.astype(int)
    data['is_weekend'] = np.where(data['day_of_week'].isin([5,6]),1,0)

extract_date_features(df_train)
#df_train.head()

#Calculate Time Difference

def calculate_picked_time(row):
    # Add a day offset if 'time_order_picked' is less than 'time_orderd'
    if row['time_order_picked'] < row['time_orderd']:
        return row['order_date'] + pd.DateOffset(days=1) + row['time_order_picked']
    else:
        return row['order_date'] + row['time_order_picked']

def calculate_time_diff(df):
    # Find the difference between ordered time & picked time
    df['time_orderd'] = pd.to_timedelta(df['time_orderd'])
    df['time_order_picked'] = pd.to_timedelta(df['time_order_picked'])

    df['time_order_picked_formatted'] = df.apply(calculate_picked_time, axis=1)
    df['time_ordered_formatted'] = df['order_date'] + df['time_orderd']

    df['time_order_picked_formatted'] = pd.to_datetime(df['time_order_picked_formatted'])
    df['time_ordered_formatted'] = pd.to_datetime(df['time_ordered_formatted'])

    df['order_prepare_time'] = (df['time_order_picked_formatted'] - df['time_ordered_formatted']).dt.total_seconds() / 60

    # Handle null values by filling with the median
    df['order_prepare_time'].fillna(df['order_prepare_time'].median(), inplace=True)

    # Drop all the time & date related columns
    df.drop(['time_orderd', 'time_order_picked', 'time_ordered_formatted', 'time_order_picked_formatted', 'order_date'], axis=1, inplace=True)


calculate_time_diff(df_train)
#df_train.head()


#Calculate distance between restaurant location & delivery location

# using haversine method
def haversine_distance(loc_list):
    # earth's radius in km
    R = 6371.0

    # convert lat and lon from deg to radians
    lat1,lon1,lat2,lon2 = map(np.radians,loc_list)
    # diff between lat and lon
    d_lat = lat2 - lat1
    d_lon = lon2 - lon1
    # applying haversine formula
    a = np.sin(d_lat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(d_lon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    d = R * c
    return round(d,2)

loc_cols = ["restaurant_latitude", "restaurant_longitude", "delivery_location_latitude", "delivery_location_longitude"]

df_train['restaurant_latitude'] = pd.to_numeric(df_train['restaurant_latitude'], errors='coerce')
df_train['restaurant_longitude'] = pd.to_numeric(df_train['restaurant_longitude'], errors='coerce')
df_train['delivery_location_latitude'] = pd.to_numeric(df_train['delivery_location_latitude'], errors='coerce')
df_train['delivery_location_longitude'] = pd.to_numeric(df_train['delivery_location_longitude'], errors='coerce')

distance = []

    # Iterate over each row in the DataFrame
for _, row in df_train.iterrows():
    # Extract values for the location columns
    location_list = [row[col] for col in loc_cols]

    # Calculate haversine distance
    distance.append(haversine_distance(location_list))

# Add the distances as a new column
df_train["distance"] = distance


#df_train.head()


df_train = df_train.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
# To remove the rows with zero lat and long
def drop_zero_lat_long(data):
    dataframe = data[-((data["restaurant_latitude"]==0.0) & (data["restaurant_longitude"]==0.0)) ]
    return dataframe


df_train = drop_zero_lat_long(df_train)

weather_mappings = {
    "Sunny": 0,
    "Cloudy": 1,
    "Fog": 2,
    "Windy": 3,
    "Stormy": 4,
    "Sandstorms": 5,
}

traff_den_mappings = {
    "Low": 0,
    "Medium": 1,
    "High": 2,
    "Jam": 3,
}

order_type_mappings = {
    "Snack": 0,
    "Drinks": 1,
    "Buffet": 2,
    "Meal": 3
}

vehicle_mappings = {
    "bicycle": 0,
    "scooter": 1,
    "electric_scooter": 2,
    "motorcycle": 3,
}

festival_mappings = {
    "No": 0,
    "Yes": 1
}

city_area_mappings = {
    "Semi-Urban": 0,
    "Urban": 1,
    "Metropolitian": 2,
}

city_mappings = {
    "INDO": 0,
    "BANG": 1,
    "COIMB": 2,
    "CHEN": 3,
    "HYD": 4,
    "RANCHI": 5,
    "MYS": 6,
    "DEH": 7,
    "KOC": 8,
    "PUNE": 9,
    "LUDH": 10,
    "KNP": 11,
    "MUM": 12,
    "KOL": 13,
    "JAP": 14,
    "SUR": 15,
    "GOA": 16,
    "AURG": 17,
    "AGR": 18,
    "VAD": 19,
    "ALH": 20,
    "BHP": 21
}

mnth_mappings = {
    "January": 0,
    "February": 1,
    "March": 2,
    "April": 3,
    "May": 4,
    "June": 5,
    "July": 6,
    "August": 7,
    "September": 8,
    "October": 9,
    "November": 10,
    "December": 11
}

class WeekdayImputer(BaseEstimator, TransformerMixin):
    """ Impute missing values in 'weekday' column by extracting dayname from 'dteday' column """

    def __init__(self, variable: str, date_var:str):

        if not isinstance(variable, str):
            raise ValueError("variable name should be a string")
        if not isinstance(date_var, str):
            raise ValueError("date variable name should be a string")

        self.variable = variable
        self.date_var = date_var

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        # we need the fit statement to accomodate the sklearn pipeline
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        # convert 'dteday' column to Datetime datatype
        X[self.date_var] = pd.to_datetime(X[self.date_var], format='%Y-%m-%d')

        wkday_null_idx = X[X[self.variable].isnull() == True].index
        X.loc[wkday_null_idx, self.variable] = X.loc[wkday_null_idx, self.date_var].dt.day_name().apply(lambda x: x[:3])

        # drop 'dteday' column after imputation
        X.drop(self.date_var, axis=1, inplace=True)

        return X


class WeathersitImputer(BaseEstimator, TransformerMixin):
    """ Impute missing values in 'weathersit' column by replacing them with the most frequent category value """

    def __init__(self, variable: str):

        if not isinstance(variable, str):
            raise ValueError("variable name should be a string")

        self.variable = variable

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        # we need the fit statement to accomodate the sklearn pipeline
        X = X.copy()
        self.fill_value = X[self.variable].mode()[0]

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X[self.variable] = X[self.variable].fillna(self.fill_value)

        return X


class Mapper(BaseEstimator, TransformerMixin):
    """
    Ordinal categorical variable mapper:
    Treat column as Ordinal categorical variable, and assign values accordingly
    """

    def __init__(self, variable:str, mappings:dict):

        if not isinstance(variable, str):
            raise ValueError("variable name should be a string")

        self.variable = variable
        self.mappings = mappings

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        # we need the fit statement to accomodate the sklearn pipeline
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X[self.variable] = X[self.variable].map(self.mappings).astype(int)

        return X


class OutlierHandler(BaseEstimator, TransformerMixin):
    """
    Change the outlier values:
        - to upper-bound, if the value is higher than upper-bound, or
        - to lower-bound, if the value is lower than lower-bound respectively.
    """

    def __init__(self, variable:str):

        if not isinstance(variable, str):
            raise ValueError("variable name should be a string")

        self.variable = variable

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        # we need the fit statement to accomodate the sklearn pipeline
        X = X.copy()
        q1 = X.describe()[self.variable].loc['25%']
        q3 = X.describe()[self.variable].loc['75%']
        iqr = q3 - q1
        self.lower_bound = q1 - (1.5 * iqr)
        self.upper_bound = q3 + (1.5 * iqr)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        for i in X.index:
            if X.loc[i, self.variable] > self.upper_bound:
                X.loc[i, self.variable]= self.upper_bound
            if X.loc[i, self.variable] < self.lower_bound:
                X.loc[i, self.variable]= self.lower_bound

        return X


class WeekdayOneHotEncoder(BaseEstimator, TransformerMixin):
    """ One-hot encode weekday column """

    def __init__(self, variable:str):

        if not isinstance(variable, str):
            raise ValueError("variable name should be a string")

        self.variable = variable
        self.encoder = OneHotEncoder(sparse_output=False)

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        # we need the fit statement to accomodate the sklearn pipeline
        X = X.copy()
        self.encoder.fit(X[[self.variable]])
        # Get encoded feature names
        self.encoded_features_names = self.encoder.get_feature_names_out([self.variable])

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        encoded_weekdays = self.encoder.transform(X[[self.variable]])
        # Append encoded weekday features to X
        X[self.encoded_features_names] = encoded_weekdays

        # drop 'weekday' column after encoding
        X.drop(self.variable, axis=1, inplace=True)

        return X
    
df_train.describe() ## feature scaling is required
pd.set_option('display.max_columns',50)
df_train.head(2)
df_train.columns
prep_data= df_train.copy()
prep_data["weather_conditions"] = prep_data["weather_conditions"].map(weather_mappings)
prep_data["road_traffic_density"] = prep_data["road_traffic_density"].map(traff_den_mappings)
prep_data["type_of_order"] = prep_data["type_of_order"].map(order_type_mappings)
prep_data["type_of_vehicle"] = prep_data["type_of_vehicle"].map(vehicle_mappings)
prep_data["city"] = prep_data["city"].map(city_area_mappings)
prep_data["festival"] = prep_data["festival"].map(festival_mappings)
prep_data["month"] = prep_data["month"].map(mnth_mappings)

prep_data.isna().sum()
prep_data.shape
prep_data.describe()
# prep_data[['distance', 'order_prepare_time']
prep_data[prep_data['distance']>30]
prep_data['distance'].mean()
prep_data['distance'].median()
prep_data['distance'] = np.where(prep_data['distance']>30,prep_data['distance'].median(),prep_data['distance'])
print(prep_data['order_prepare_time'].mean())
print(prep_data['order_prepare_time'].median())
# prep_data[['distance', 'order_prepare_time']
prep_data[prep_data['order_prepare_time']>50]
prep_data['order_prepare_time'] = np.where(prep_data['order_prepare_time']>60,prep_data['order_prepare_time'].median(),prep_data['order_prepare_time'])
prep_data.head()
target = ['time_taken_in_min']
features = ['delivery_person_age', 'delivery_person_ratings','weather_conditions', 'road_traffic_density', 'vehicle_condition', 'type_of_order',
            'type_of_vehicle', 'multiple_deliveries', 'festival', 'city', 'day_of_week', 'is_weekend', 'is_month_start',	'is_month_end',
            'quarter', 'distance', 'order_prepare_time']

prep_data.describe()
# divide train and test
X_train, X_test, y_train, y_test = train_test_split(

        prep_data[features],     # predictors
        prep_data[target],       # target
        test_size = 0.2,
        random_state= 42,   # set the random seed here for reproducibility
    )

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
#from xgboost import XGBRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV, cross_val_score, RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
#import xgboost as xgb
from catboost import CatBoostRegressor

from sklearn.metrics import mean_squared_error, r2_score, mean_squared_error, mean_absolute_percentage_error
#X_train.head(2)


catboost =  CatBoostRegressor()
catboost.fit(X_train, y_train, verbose=False)
y_pred_cat = catboost.predict(X_test)
print("R2 score:", r2_score(y_test, y_pred_cat))
print("Mean squared error:", mean_squared_error(y_test, y_pred_cat))
print(f"Root mean squared error:{np.sqrt(mean_squared_error(y_test,y_pred_cat))}")
print("Mean Absolute Pecentage error:",mean_absolute_percentage_error(y_test,y_pred_cat))
cat_data = df_train.copy()
cat_data['distance'] = np.where(cat_data['distance']>30,cat_data['distance'].median(),cat_data['distance'])
cat_data['order_prepare_time'] = np.where(cat_data['order_prepare_time']>60,cat_data['order_prepare_time'].median(),cat_data['order_prepare_time'])
target_cat = ['time_taken_in_min']
features_cat = ['delivery_person_age', 'delivery_person_ratings','weather_conditions', 'road_traffic_density', 'vehicle_condition', 'type_of_order',
            'type_of_vehicle', 'multiple_deliveries', 'festival', 'city', 'day_of_week', 'is_weekend', 'is_month_start',	'is_month_end',
            'quarter', 'distance', 'order_prepare_time']

cat_data[features_cat].head(2)
cat_data[['is_weekend','is_month_start','is_month_end']].head(2)
object_cols_indices = list(cat_data.select_dtypes(include='object').columns)
cat_data
object_cols_indices =['weather_conditions',
 'road_traffic_density',
 'type_of_order',
 'type_of_vehicle',
 'festival',
 'city',
 'is_month_start',
 'is_month_end',
 'is_weekend']

# divide train and test
X_train_cat, X_test_cat, y_train_cat, y_test_cat = train_test_split(

        cat_data[features_cat],     # predictors
        cat_data[target_cat],       # target
        test_size = 0.2,
        random_state= 42,   # set the random seed here for reproducibility
    )


catboost_2 =  CatBoostRegressor()
catboost_2.fit(X_train_cat, y_train_cat, verbose=False, cat_features=object_cols_indices)
y_pred_cat = catboost_2.predict(X_test_cat)
#print(X_test_cat[1])
print("R2 score:", r2_score(y_test_cat, y_pred_cat))
print("Mean squared error:", mean_squared_error(y_test_cat, y_pred_cat))
print(f"Root mean squared error:{np.sqrt(mean_squared_error(y_test_cat,y_pred_cat))}")
print("Mean Absolute Pecentage error:",mean_absolute_percentage_error(y_test_cat,y_pred_cat))



import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from typing import Union
import pandas as pd
import numpy as np

############################## Weather API start ###################################
def weather_api(city_map):
    # import required modules
    import requests, json

# Enter your API key here
    api_key = ""

# importing requests and json
    import requests, json
# base URL
    BASE_URL = "https://api.openweathermap.org/data/2.5/weather?"
    API_KEY = ""



    city_mappings = {
        "INDORE": 0,
        "BANGALORE": 1,
        "COIMBATORE": 2,
        "CHENNAI": 3,
        "HYDERABAD": 4,
        "RANCHI": 5,
        "MYSORE": 6,
        "DELHI": 7,
        "KOCHI": 8,
        "PUNE": 9,
        "LUDHIANA": 10,
        "KANPUR": 11,
        "MUMBAI": 12,
        "KOLKATTA": 13,
        "JAIPUR": 14,
        "SURAT": 15,
        "GOA": 16,
        "AURANGABAD": 17,
        "AGRA": 18,
        "VADODRA": 19,
        "ALLAHABAD": 20,
        "BHOPAL": 21
    }


    def get_key_city(val_city):
  
        for key, value in city_mappings.items():
            if val_city == value:
                return key

        return "key doesn't exist"


    CITY=get_key_city(city_map)

    weather_mappings = {
        "Sunny": 1,
        "clear sky":2,
        "Cloudy": 3,
        "haze": 6,
        "Fog": 7,
        "Windy": 3,
        "Stormy": 4,
        "Sandstorms": 5,
        "mist":8
    }




# upadting the URL
    URL = BASE_URL + "q=" + CITY + "&appid=" + API_KEY
# HTTP request
    response = requests.get(URL)
# checking the status code of the request
    if response.status_code == 200:
   # getting data in the json format
        data = response.json()
   # getting the main dict block
        main = data['main']
   # getting temperature
        temperature = main['temp']
   # getting the humidity
        humidity = main['humidity']
   # getting the pressure
        pressure = main['pressure']
   # weather report
        report = data['weather']
        weather=report[0]['description']
        print(f"{CITY:-^30}")
        print(f"Temperature: {temperature}")
        print(f"Humidity: {humidity}")
        print(f"Pressure: {pressure}")
        print(f"Weather Report: {report[0]['description']}")
    else:
   # showing the error message
        print("Error in the HTTP request")


    def get_key_weather(val_weather):
  
        for key, value in weather_mappings.items():
            if val_weather == key:
                return value

        return "key doesn't exist"

    Value=get_key_weather(weather)
    return(Value)
############################## Weather API close ###################################


############################## Holiday API start ###################################
def holiday_api():
    import requests
    from datetime import datetime
    from typing import List



    GOOGLE_CALENDAR_API_KEY=''
    if not GOOGLE_CALENDAR_API_KEY:
        raise Exception("The GOOGLE_CALENDAR_API_KEY environment variable is not set.")

    def fetch_holidays(country_code: str, year: str):
        url = f"https://www.googleapis.com/calendar/v3/calendars/{country_code}%23holiday%40group.v.calendar.google.com/events"
        params = {
            'key': GOOGLE_CALENDAR_API_KEY,
            'timeMin': f'{year}-01-01T00:00:00Z',
            'timeMax': f'{year}-12-31T23:59:59Z',
            'singleEvents': True,
            'orderBy': 'startTime'
        }
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json().get('items', [])
    

    country_code='en.indian'
    year=2024
    holidays = fetch_holidays(country_code, year)




    from datetime import date

# Returns the current local date
    today = date.today()
    print("Today date is: ", today)
    today='2024-08-15' # Testing
    print("Today date is: ", today)

    isholiday=0
    for i in holidays:
    
        if i['start']['date'] == today:
            isholiday=1
            print("Today is ",i['summary'])
            break
    return(isholiday)

############################## Holiday API close ###################################

############################## Traffic API start ###################################
def traffic_density(source,destination):
    import requests
    import smtplib 

# API key
    api_key =''


    home=source
    work=destination
# base url
    url = "https://maps.googleapis.com/maps/api/distancematrix/json?units=imperial&"


    r = requests.get(url + "origins=" + home + "&destinations=" + work + "&key=" + api_key) 
 
# return time as text and as seconds
    time = r.json()["rows"][0]["elements"][0]["duration"]["text"]       
    seconds = r.json()["rows"][0]["elements"][0]["duration"]["value"]
    distance = r.json()["rows"][0]["elements"][0]["distance"]["text"]

    
    time_hr=time.split()[0]
    time_hr=float(time_hr)
    time_hr=(time_hr/60)
    print('time in hr:',time_hr)
    distance_km=distance.split()[0]
    distance_km=float(distance_km)
    distance_km= int(distance_km*1.6)
    print('distance in km:',distance_km)
    speed=int(distance_km/time_hr)
    print('speed:',speed)

    if(speed > 14):
        road_traffic_density=2

    else:
        road_traffic_density=1
     
     
    return(road_traffic_density)

############################## Traffic API close ###################################
def make_prediction(*, input_data: Union[pd.DataFrame, dict]) -> dict:
    """Make a prediction using a saved model """
    validated_data = pd.DataFrame(input_data)

    #Data before uptaking api
    print(validated_data)


    
    # For weather #
    val1=data_in['city'][0]
    weather_prev=validated_data['weather_conditions'][0]
    print('\nweather_prev:',validated_data['weather_conditions'][0])
    weather_result=weather_api(val1)
    validated_data['weather_conditions'][0]=weather_result
    print('weather_now:',validated_data['weather_conditions'][0])


    #For holiday or festival
    print('\nfestival_prev:',validated_data['festival'][0])
    holiday_result=holiday_api()
    validated_data['festival'][0]=holiday_result
    print('festival_now:',validated_data['festival'][0])

    
    #For traffic density
    print('\nroad_traffic_density_prev:',validated_data['road_traffic_density'][0])
    validated_data['road_traffic_density'][0]=traffic_density('12.91682,77.62339','12.92266,77.61736')
    print('road_traffic_density_now:',validated_data['road_traffic_density'][0])


    #Predicting 
    predictions = catboost_2.predict(validated_data)
    print('\n ####prediction:',predictions)
    

    return predictions


#For inference
if __name__ == "__main__":

    data_in={
    'delivery_person_age':[3],
    'delivery_person_ratings':[4],
    'weather_conditions':[0],
    'road_traffic_density':[2],
    'vehicle_condition':[2],
    'type_of_order':[1],
    'type_of_vehicle':[2],
    'multiple_deliveries':[8],
    'festival':[0],
    'city':[5],
    'day_of_week':[3],
    'is_weekend':[0],
    'is_month_start':[1],
    'is_month_end':[1],
    'quarter':[1],
    'distance':[6],
    'order_prepare_time':[40]}



make_prediction(input_data = data_in)


