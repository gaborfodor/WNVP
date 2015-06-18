import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os import getcwd, makedirs
from os.path import join, exists
import datetime as dt
from sklearn.cluster import KMeans
from sklearn import metrics

time0 = dt.datetime.now()
# set folders
base_folder = getcwd()
data_folder = join(base_folder, 'data')
subm_folder = join(base_folder, 'submissions')
du_folder = join(base_folder, 'du')
if not exists(du_folder):
    makedirs(du_folder)
# read data
train = pd.read_csv(join(data_folder, 'train.csv'))
test = pd.read_csv(join(data_folder, 'test.csv'))
ss = pd.read_csv(join(data_folder, 'sampleSubmission.csv'))
ss_auc = pd.read_csv(join(data_folder, 'submission_auc.csv'))

train['Id'] = np.arange(len(train))
# date -> year, month
train['year'] = map(lambda x: int(x.split('-')[0]), train['Date'])
train['month'] = map(lambda x: int(x.split('-')[1]), train['Date'])
test['year'] = map(lambda x: int(x.split('-')[0]), test['Date'])
test['month'] = map(lambda x: int(x.split('-')[1]), test['Date'])

###############################################################################
# Monthly features
###############################################################################
test['t_0'] = ((test['month'] == 5)*1 + (test['month'] == 6)*1)
test['t_1'] = (test['month'] == 10)*1
i = 2
for y in list(set(test['year'])):
    for m in [7, 8, 9]:
        test['t_' + str(i)] = ((test['year'] == y)*1 * (test['month'] == m)*1)
        i += 1
# create month submissions
a = 0
for i in range(14):
    ss['WnvPresent'] = test['t_'+str(i)]
    ss.to_csv(join(subm_folder, 'test_t'+str(i)+'.csv'), index=False)
    a += ss['WnvPresent'].sum()

# train
train['t_0'] = ((train['month'] == 5)*1 + (train['month'] == 6)*1)
train['t_1'] = (train['month'] == 10)*1
i = 2
for y in list(set(train['year'])):
    for m in [7, 8, 9]:
        train['t_' + str(i)] = ((train['year'] == y)*1 * (train['month'] == m)*1)
        i += 1

print train.shape[1] - 2 == test.shape[1]  # Ok
###############################################################################
# Shifted Monthly features
###############################################################################
date_times = [dt.datetime.strptime(d, '%Y-%m-%d') for d in list(test['Date'])]
date_times_2w = [d + dt.timedelta(days=14) for d in date_times]
test['month_2w'] = [d.month for d in date_times_2w]
test['w_0'] = ((test['month_2w'] == 5)*1 + (test['month_2w'] == 6)*1 + (test['month_2w'] == 7)*1)
test['w_1'] = (test['month_2w'] == 10)*1
i = 2
for y in list(set(test['year'])):
    for m in [8, 9]:
        test['w_' + str(i)] = ((test['year'] == y)*1 * (test['month_2w'] == m)*1)
        i += 1

# create shifted month submissions
a = 0
for i in range(9):
    ss['WnvPresent'] = test['w_'+str(i)]
    ss.to_csv(join(subm_folder, 'test_w'+str(i)+'.csv'), index=False)
    a += ss['WnvPresent'].sum()

# train
date_times = [dt.datetime.strptime(d, '%Y-%m-%d') for d in list(train['Date'])]
date_times_2w = [d + dt.timedelta(days=14) for d in date_times]
train['month_2w'] = [d.month for d in date_times_2w]
train['w_0'] = ((train['month_2w'] == 5)*1 + (train['month_2w'] == 6)*1 + (train['month_2w'] == 7)*1)
train['w_1'] = (train['month_2w'] == 10)*1
i = 2
for y in list(set(train['year'])):
    for m in [8, 9]:
        train['w_' + str(i)] = ((train['year'] == y)*1 * (train['month_2w'] == m)*1)
        i += 1

print train.shape[1] - 2 == test.shape[1]  # Ok
###############################################################################
# Species Features
###############################################################################
test['s_0'] = (test['Species'] == 'UNSPECIFIED CULEX')*1
test['s_1'] = ((test['Species'] == 'CULEX ERRATICUS')*1 + (test['Species'] == 'CULEX SALINARIUS')*1 + (test['Species'] == 'CULEX TARSALIS')*1 + (test['Species'] == 'CULEX TERRITANS')*1)
i = 2
for y in list(set(test['year'])):
    for s in ['CULEX PIPIENS', 'CULEX PIPIENS/RESTUANS', 'CULEX RESTUANS']:
        test['s_' + str(i)] = ((test['year'] == y)*1 * (test['Species'] == s)*1)
        i += 1

# create species submissions
a = 0
for i in range(14):
    ss['WnvPresent'] = test['s_'+str(i)]
    ss.to_csv(join(subm_folder, 'test_s'+str(i)+'.csv'), index=False)
    a += ss['WnvPresent'].sum()

# train
train['s_0'] = (train['Species'] == 'UNSPECIFIED CULEX')*1
train['s_1'] = ((train['Species'] == 'CULEX ERRATICUS')*1 + (train['Species'] == 'CULEX SALINARIUS')*1 + (train['Species'] == 'CULEX TARSALIS')*1 + (train['Species'] == 'CULEX TERRITANS')*1)
i = 2
for y in list(set(train['year'])):
    for s in ['CULEX PIPIENS', 'CULEX PIPIENS/RESTUANS', 'CULEX RESTUANS']:
        train['s_' + str(i)] = ((train['year'] == y)*1 * (train['Species'] == s)*1)
        i += 1

print train.shape[1] - 2 == test.shape[1]  # Ok
###############################################################################
# Location Features
###############################################################################
# Create Location Clusters
kmeans = KMeans(init='k-means++', n_clusters=10, n_init=10, random_state=1987)
kmeans.fit(train[['Latitude','Longitude']])
train['location'] = kmeans.predict(train[['Latitude','Longitude']])
test['location'] = kmeans.predict(test[['Latitude','Longitude']])

# plot clusters
fig, ax = plt.subplots(ncols=2, sharex=True, sharey=True)
ax[0].scatter(train['Latitude'], train['Longitude'], s=50, c=train['location'], alpha=.2)
ax[0].grid()
ax[0].set_title('train')
ax[1].scatter(test['Latitude'], test['Longitude'], s=50, c=test['location'], alpha=.2)
ax[1].grid()
ax[1].set_title('test')
ax[0].set_xlabel('Latitude')
ax[1].set_xlabel('Latitude')
ax[0].set_ylabel('Longitude')
plt.suptitle('Location Clusters')
fig.savefig(join(du_folder, 'Clusters.png'), dpi=300)

# species with positive train samples and  relevant season
train['pipiens_restuans'] = ((train['Species'] == 'CULEX PIPIENS')*1 + (train['Species'] == 'CULEX PIPIENS/RESTUANS')*1 + (train['Species'] == 'CULEX RESTUANS')*1)
train['season'] = ((train['month'] == 7)*1 + (train['month'] == 8)*1 + (train['month'] == 9)*1)
test['pipiens_restuans'] = ((test['Species'] == 'CULEX PIPIENS')*1 + (test['Species'] == 'CULEX PIPIENS/RESTUANS')*1 + (test['Species'] == 'CULEX RESTUANS')*1)
test['season'] = ((test['month'] == 7)*1 + (test['month'] == 8)*1 + (test['month'] == 9)*1)

train.groupby(['season']).mean()['WnvPresent']
for l in list(set(test['location'])):
    test['l_' + str(l)] = (test['location'] == l)*1 * test['pipiens_restuans'] * test['season']

for l in list(set(train['location'])):
    train['l_' + str(l)] = (train['location'] == l)*1 * train['pipiens_restuans'] * train['season']

# create species submissions
a = 0
for l in list(set(test['location'])):
    ss['WnvPresent'] = test['l_'+str(l)]
    ss.to_csv(join(subm_folder, 'test_l'+str(l)+'.csv'), index=False)
    a += ss['WnvPresent'].sum()

print train.shape[1] - 2 == test.shape[1]  # Ok
###############################################################################
# Test Record Counts
###############################################################################
# DateCount
dt_count = test.groupby('Date').count()[['Id']]
dt_count.columns = ['DateCount']
test = pd.merge(test, dt_count, how='inner', left_on='Date', right_index=True)
cnt_pred = np.array(test['DateCount'])
cnt_pred = 1.0*(cnt_pred - cnt_pred.min()) / (cnt_pred.max() - cnt_pred.min())
ss['WnvPresent'] = cnt_pred
ss.to_csv(join(subm_folder, 'DateCount.csv'), index=False)

###############################################################################
# Training simulation
###############################################################################
result = []
for repeat in range(100):
    print repeat,
    train['random'] = np.random.rand(len(train))
    public =  train[train['random'] < .3]
    private =  train[train['random'] >= .3]
    features = list(train.columns)
    to_remove = ['Date', 'Address', 'Species', 'Block', 'Street', 'Trap', 'AddressNumberAndStreet', 'Latitude', 'Longitude', 'AddressAccuracy', 'NumMosquitos', 'WnvPresent', 'Id',  'year', 'month', 'month_2w',  'location', 'pipiens_restuans', 'season', 'random']
    features = [f for f in features if f not in to_remove]
    feature_auc = []
    for f in features:
        fpr_public, tpr_public, thresholds = metrics.roc_curve(np.array(public['WnvPresent']), np.array(public[f]), pos_label=1)
        auc_public = metrics.auc(fpr_public, tpr_public)
        feature_auc.append([f, auc_public])
    
    feature_auc_df = pd.DataFrame(feature_auc, columns=['f', 'auc_public'])
    feature_auc_df['importance'] = np.abs(.5 - feature_auc_df['auc_public'])
    feature_auc_df = feature_auc_df.sort('importance')
    
    for alpha in [.33, .5, .6, .75, 1., 1.5, 2.]:
        for n in range(len(feature_auc_df) - 10):
            ensemble_public_prediction = np.zeros(len(public))
            ensemble_private_prediction = np.zeros(len(private))
            for i, row in feature_auc_df[n:].iterrows():
                f = row['f']
                auc = row['auc_public']
                if(auc > .5):
                    ensemble_public_prediction += (np.abs(auc - .5)**alpha) * np.array(public[f])
                    ensemble_private_prediction += (np.abs(auc - .5)**alpha) * np.array(private[f])
                else:
                    ensemble_public_prediction -= (np.abs(auc - .5)**alpha) * np.array(public[f])
                    ensemble_private_prediction -= (np.abs(auc - .5)**alpha) * np.array(private[f])
            
            ensemble_public_prediction = (ensemble_public_prediction - ensemble_public_prediction.min()) / (ensemble_public_prediction.max() - ensemble_public_prediction.min())
            ensemble_private_prediction = (ensemble_private_prediction - ensemble_private_prediction.min()) / (ensemble_private_prediction.max() - ensemble_private_prediction.min())
            
            fpr_public, tpr_public, thresholds = metrics.roc_curve(np.array(public['WnvPresent']), ensemble_public_prediction, pos_label=1)
            auc_public = metrics.auc(fpr_public, tpr_public)
            
            fpr_private, tpr_private, thresholds = metrics.roc_curve(np.array(private['WnvPresent']), ensemble_private_prediction, pos_label=1)
            auc_private = metrics.auc(fpr_private, tpr_private)
            result.append([n, alpha, auc_public, auc_private])

result_df = pd.DataFrame(result, columns=['n', 'alpha', 'auc_public',
                                          'auc_private'])
result_df.to_csv(join(du_folder, 'SplitTest.csv'), index=False)
print result_df.shape
fig, ax = plt.subplots(ncols=2, sharey=True)
for alpha in [.33, .5, .6, .75, 1., 1.5, 2.]:
    a = result_df[result_df['alpha'] == alpha]
    private = a.groupby('n').mean()['auc_private']
    public = a.groupby('n').mean()['auc_public']
    ax[0].plot(public, 'o--', linewidth=2, alpha=.5, label = str(np.round(alpha,3)))
    ax[1].plot(private, 'o--', linewidth=2, alpha=.5)

ax[0].grid()
ax[1].grid()
ax[0].legend(loc=0)
ax[0].set_title('Public Split')
ax[1].set_title('Private Split')
ax[0].set_xlabel('number of features')
ax[1].set_xlabel('number of features')
ax[0].set_ylabel('AUC')
fig.savefig(join(du_folder, 'SubmissionWeighting.png'), dpi=300)

fig, ax = plt.subplots()
plt.plot(fpr_public, tpr_public, 'b-', alpha=.3, linewidth=2)
plt.plot(fpr_private, tpr_private, 'g-', alpha=.3, linewidth=2)
plt.grid()
fig.savefig(join(du_folder, 'ROC.png'), dpi=300)

###############################################################################
# Create Final Submission
###############################################################################
alpha = .5
ss_auc['importance'] = np.abs(.5 - ss_auc['auc'])
ss_auc = ss_auc.sort('importance')
ensemble_pred = np.zeros(len(ss))
for drop in [0, 5, 10, 15, 20]:
    for i, row in ss_auc[drop:].iterrows(): # drop the most random 5 submissions
        prev_subm = pd.read_csv(join(subm_folder, row['f']))
        if(row['auc'] > .5):
            ensemble_pred += (np.abs(row['auc'] - .5))**alpha * np.array(prev_subm['WnvPresent'])
        else:
            ensemble_pred -= (np.abs(row['auc'] - .5))**alpha * np.array(prev_subm['WnvPresent'])
    
    ensemble_pred = (ensemble_pred - ensemble_pred.min()) / (ensemble_pred.max() - ensemble_pred.min())
    ss['WnvPresent'] = ensemble_pred
    
    rm = pd.read_csv(join(subm_folder, 'realmodel.csv'))
    ensemble_pred += 1.5 * np.array(rm['WnvPresent'])
    ensemble_pred = (ensemble_pred - ensemble_pred.min()) / (ensemble_pred.max() - ensemble_pred.min())
    ss['WnvPresent'] = ensemble_pred
    
    # Decrease the Species and Time weights where there is not WNV in training
    test = test.sort('Id')
    ensemble_pred -= 10 * np.array(test['s_0'])
    ensemble_pred -= 10 * np.array(test['s_1'])
    ensemble_pred -= 10 * np.array(test['t_0'])
    ensemble_pred -= 1 * np.array(test['t_1'])
    ensemble_pred = (ensemble_pred - ensemble_pred.min()) / (ensemble_pred.max() - ensemble_pred.min())
    ss['WnvPresent'] = ensemble_pred
    ss.to_csv(join(subm_folder, 'alpha_%s_drop_%s_selection.csv'%(alpha, drop)), index=False)

time1 = dt.datetime.now()
print 'total time:', (time1-time0).seconds, 'sec'