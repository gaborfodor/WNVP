import pandas as pd
from sklearn import ensemble, preprocessing
from os.path import join, exists
from os import getcwd, makedirs
import datetime as dt
from sklearn import metrics

# Functions to extract year, month, day and weekday from dataset
def create_year(x):
    return x.split('-')[0]

def create_month(x):
    return x.split('-')[1]

def create_day(x):
    return x.split('-')[2]

def create_weekday(x):
    date_time = dt.datetime.strptime(x, '%Y-%m-%d')
    return date_time.strftime('%w')

time0 = dt.datetime.now()
# Load dataset
base_folder = getcwd()
data_folder = join(base_folder, 'data')
subm_folder = join(base_folder, 'submissions')
if not exists(subm_folder):
    makedirs(subm_folder)

train = pd.read_csv(join(data_folder, 'train.csv'))
test = pd.read_csv(join(data_folder, 'test.csv'))
sample = pd.read_csv(join(data_folder, 'sampleSubmission.csv'))
weather = pd.read_csv(join(data_folder, 'weather.csv'))

# Get labels
labels = train.WnvPresent.values

# Not using codesum for this benchmark
weather = weather.drop('CodeSum', axis=1)

# Split station 1 and 2 and join horizontally
weather_stn1 = weather[weather['Station'] == 1]
weather_stn2 = weather[weather['Station'] == 2]
weather_stn1 = weather_stn1.drop('Station', axis=1)
weather_stn2 = weather_stn2.drop('Station', axis=1)
weather = weather_stn1.merge(weather_stn2, on='Date')

# replace some missing values and T with -1
weather = weather.replace('M', -1)
weather = weather.replace('-', -1)
weather = weather.replace('T', -1)
weather = weather.replace(' T', -1)
weather = weather.replace('  T', -1)

train['year'] = train.Date.apply(create_year)
train['month'] = train.Date.apply(create_month)
train['day'] = train.Date.apply(create_day)
train['weekday'] = train.Date.apply(create_weekday)
test['year'] = test.Date.apply(create_year)
test['month'] = test.Date.apply(create_month)
test['day'] = test.Date.apply(create_day)
test['weekday'] = test.Date.apply(create_weekday)

# Add integer latitude/longitude columns
train['Lat_int'] = train.Latitude.apply(lambda x: int(x*5))
train['Long_int'] = train.Longitude.apply(lambda x: int(x*5))
test['Lat_int'] = test.Latitude.apply(lambda x: int(x*5))
test['Long_int'] = test.Longitude.apply(lambda x: int(x*5))

# drop address columns
train = train.drop(['Address', 'AddressNumberAndStreet', 'NumMosquitos'], axis=1 )
test = test.drop(['Id', 'Address', 'AddressNumberAndStreet'], axis=1)

# Merge with weather data
train = train.merge(weather, on='Date')
test = test.merge(weather, on='Date')
train = train.drop(['Date'], axis=1)
test = test.drop(['Date'], axis=1)

# Convert categorical data to numbers
lbl = preprocessing.LabelEncoder()
lbl.fit(list(train['Species'].values) + list(test['Species'].values))
train['Species'] = lbl.transform(train['Species'].values)
test['Species'] = lbl.transform(test['Species'].values)

lbl.fit(list(train['Street'].values) + list(test['Street'].values))
train['Street'] = lbl.transform(train['Street'].values)
test['Street'] = lbl.transform(test['Street'].values)

lbl.fit(list(train['Trap'].values) + list(test['Trap'].values))
train['Trap'] = lbl.transform(train['Trap'].values)
test['Trap'] = lbl.transform(test['Trap'].values)

# drop columns with -1s
train = train.ix[:, (train != -1).any(axis=0)]
test = test.ix[:, (test != -1).any(axis=0)]

features = list(train.columns)
features.remove('WnvPresent')
features.remove('day')
features.remove('year')
# Random Forest Classifier CV
for year in set(train.year):
    X = train[train.year != year][features]
    V = train[train.year == year][features]
    labelsX = train[train.year != year]['WnvPresent']
    labelsV = train[train.year == year]['WnvPresent']
    rf = ensemble.RandomForestClassifier(n_estimators=5000, min_samples_split=1)
    rf.fit(X, labelsX)
    predV = rf.predict_proba(V)[:, 1]
    fpr_valid, tpr_valid, thresholds = metrics.roc_curve(labelsV, predV, pos_label=1)
    auc_valid = metrics.auc(fpr_valid, tpr_valid)
    print 'rf', year, auc_valid

# create predictions and submission file
All = train[features]
labelsAll = train['WnvPresent']
rf = ensemble.RandomForestClassifier(n_estimators=5000, min_samples_split=1, random_state=1987)
rf.fit(All, labelsAll)
print rf
predictions = rf.predict_proba(test[features])[:, 1]
sample['WnvPresent'] = predictions
sample.to_csv(join(subm_folder, 'realmodel.csv'), index=False)

time1 = dt.datetime.now()
print 'total time:', (time1-time0).seconds, 'sec'
