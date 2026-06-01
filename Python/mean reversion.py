import math
import pandas as pd
import pickle
from datetime import datetime
from catboost import CatBoostClassifier

from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

from bots.botlibs.labeling_lib import *
from bots.botlibs.tester_lib import tester
from bots.botlibs.export_lib import export_model_to_ONNX


def get_prices() -> pd.DataFrame:
    p = pd.read_csv('files/'+hyper_params['symbol']+'.csv', sep='\s+')
    pFixed = pd.DataFrame(columns=['time', 'close'])
    pFixed['time'] = p['<DATE>'] + ' ' + p['<TIME>']
    pFixed['time'] = pd.to_datetime(pFixed['time'], format='mixed')
    pFixed['close'] = p['<CLOSE>']
    pFixed.set_index('time', inplace=True)
    pFixed.index = pd.to_datetime(pFixed.index, unit='s')
    return pFixed.dropna()

def get_features(data: pd.DataFrame) -> pd.DataFrame:
    pFixed = data.copy()
    pFixedC = data.copy()
    count = 0

    for i in hyper_params['periods']:
        pFixed[str(count)] = pFixedC.rolling(i).mean()
        count += 1
    
    for i in hyper_params['periods_meta']:
        pFixed[str(count)+'meta_feature'] = pFixedC.rolling(i).skew()
        count += 1

    # for i in hyper_params['periods_meta']:
    #     pFixed[str(count)+'meta_feature'] = pFixedC.rolling(i).std()
    #     count += 1

    # for i in hyper_params['periods_meta']:
    #     pFixed[str(count)+'meta_feature'] = pFixedC - pFixedC.rolling(i).mean()
    #     count += 1

    return pFixed.dropna()

def test_model(result: list, stop: float, take: float, plt = False):
    pr_tst = get_features(get_prices())
    X = pr_tst[pr_tst.columns[1:]]
    X_meta = X.copy()
    X = X.loc[:, ~X.columns.str.contains('meta_feature')]
    X_meta = X_meta.loc[:, X_meta.columns.str.contains('meta_feature')]

    pr_tst['labels'] = result[0].predict_proba(X)[:,1]
    pr_tst['meta_labels'] = result[1].predict_proba(X_meta)[:,1]
    pr_tst['labels'] = pr_tst['labels'].apply(lambda x: 0.0 if x < 0.5 else 1.0)
    pr_tst['meta_labels'] = pr_tst['meta_labels'].apply(lambda x: 0.0 if x < 0.5 else 1.0)
    return tester(pr_tst, stop, take, hyper_params['forward'], hyper_params['backward'], hyper_params['markup'], plt)
    
def clustering(dataset, n_clusters: int) -> pd.DataFrame:
    data = dataset[(dataset.index < hyper_params['forward']) & (dataset.index > hyper_params['backward'])].copy()
    meta_X = data.loc[:, data.columns.str.contains('meta_feature')]
    data['clusters'] = KMeans(n_clusters=n_clusters).fit(meta_X).labels_
    return data

def fit_final_models(clustered, meta) -> list:
    # features for model\meta models. We learn main model only on filtered labels 
    X, X_meta = clustered[clustered.columns[:-1]], meta[meta.columns[:-1]]
    X = X.loc[:, ~X.columns.str.contains('meta_feature')]
    X_meta = X_meta.loc[:, X_meta.columns.str.contains('meta_feature')]
    
    # labels for model\meta models
    y = clustered['labels']
    y_meta = meta['clusters']
    
    y = y.astype('int16')
    y_meta = y_meta.astype('int16')

    # train\test split
    train_X, test_X, train_y, test_y = train_test_split(
        X, y, train_size=0.7, test_size=0.3, shuffle=True)
    
    train_X_m, test_X_m, train_y_m, test_y_m = train_test_split(
        X_meta, y_meta, train_size=0.7, test_size=0.3, shuffle=True)


    # learn main model with train and validation subsets
    model = CatBoostClassifier(iterations=1000,
                               custom_loss=['Accuracy'],
                               eval_metric='Accuracy',
                               verbose=False,
                               use_best_model=False,
                               task_type='CPU',
                               thread_count=-1)
    model.fit(train_X, train_y, eval_set=(test_X, test_y),
              early_stopping_rounds=30, plot=False)
    
    # learn meta model with train and validation subsets
    meta_model = CatBoostClassifier(iterations=500,
                                    custom_loss=['F1'],
                                    eval_metric='F1',
                                    verbose=False,
                                    use_best_model=True,
                                    task_type='CPU',
                                    thread_count=-1)
    meta_model.fit(train_X_m, train_y_m, eval_set=(test_X_m, test_y_m),
              early_stopping_rounds=25, plot=False)

    
    R2 = test_model([model, meta_model], hyper_params['stop_loss'], hyper_params['take_profit'])
    if math.isnan(R2):
        R2 = -1.0
        print('R2 is fixed to -1.0')
    print('R2: ' + str(R2))

    return [R2, model, meta_model]


hyper_params = {
    'symbol': 'EURGBP_H1',
    'export_path': '/Users/dmitrievsky/Library/Containers/com.isaacmarovitz.Whisky/Bottles/54CFA88F-36A3-47F7-915A-D09B24E89192/drive_c/Program Files/MetaTrader 5/MQL5/Include/Mean reversion/',
    # 'export_path': '/Users/dmitrievsky/Library/Containers/com.isaacmarovitz.Whisky/Bottles/54CFA88F-36A3-47F7-915A-D09B24E89192/drive_c/Program Files (x86)/RoboForex MT4 Terminal/MQL4/Include/',
    'model_number': 0,
    'markup': 0.00010,
    'stop_loss':  0.02000,
    'take_profit': 0.00200,
    'periods': [i for i in range(5, 300, 30)],
    'periods_meta': [10],
    'backward': datetime(2000, 1, 1),
    'forward': datetime(2021, 1, 1),
    'n_clusters': 10,
    'rolling': 200,
}


# LEARNING LOOP
dataset = get_features(get_prices())
models = []
for i in range(1):
    data = clustering(dataset, n_clusters=hyper_params['n_clusters'])
    sorted_clusters = data['clusters'].unique()
    sorted_clusters.sort()
    for clust in sorted_clusters:
        clustered_data = data[data['clusters'] == clust].copy()
        if len(clustered_data) < 500:
            print('too few samples: {}'.format(len(clustered_data)))
            continue
    
        clustered_data = get_labels_filter(clustered_data, 
                                           rolling=hyper_params['rolling'],
                                           quantiles=[0.45, 0.55],
                                           polyorder=3
                                            )
        # clustered_data = get_labels_multiple_filters(clustered_data, 
        #                                              rolling_periods=[50, 100, 200], 
        #                                              quantiles=[.45, .55], 
        #                                              window=100, 
        #                                              polyorder=3)
        # clustered_data = get_labels_filter_bidirectional(clustered_data, 
        #                                                  rolling1=50, 
        #                                                  rolling2=200, 
        #                                                  quantiles=[.45, .55], 
        #                                                  polyorder=3)
        # clustered_data = get_labels_mean_reversion(clustered_data,
        #                                             markup = hyper_params['markup'],
        #                                             min_l=1, max_l=15, 
        #                                             rolling=0.5, 
        #                                             quantiles=[.45, .55], 
        #                                             method='spline', shift=0)
        # clustered_data = get_labels_mean_reversion_multi(clustered_data, 
        #                                                  markup = hyper_params['markup'], 
        #                                                  min_l=1, max_l=15, 
        #                                                  windows=[0.2, 0.3, 0.5], 
        #                                                  quantiles=[.45, .55])
        # clustered_data = get_labels_mean_reversion_v(clustered_data,
        #                                             markup = hyper_params['markup'],
        #                                             min_l=1, max_l=15, 
        #                                             rolling=0.2, 
        #                                             quantiles=[.45, .55], 
        #                                             method='spline', 
        #                                             shift=0, 
        #                                             volatility_window=100)

        print(f'Iteration: {i}, Cluster: {clust}')
        clustered_data = clustered_data.drop(['close', 'clusters'], axis=1)

        meta_data = data.copy()
        meta_data['clusters'] = meta_data['clusters'].apply(lambda x: 1 if x == clust else 0)
        models.append(fit_final_models(clustered_data, meta_data.drop(['close'], axis=1)))

# TESTING & EXPORT
models.sort(key=lambda x: x[0])
test_model(models[-1][1:], hyper_params['stop_loss'], hyper_params['take_profit'], plt=True)

export_model_to_ONNX(model = models[-1],
                     symbol = hyper_params['symbol'],
                     periods = hyper_params['periods'],
                     periods_meta = hyper_params['periods_meta'],
                     model_number = hyper_params['model_number'],
                     export_path = hyper_params['export_path'])


models[-1][1].get_best_score()['validation']


#save and load set files
# with open('/Users/dmitrievsky/Desktop/py files/bots/set_files/' + hyper_params['symbol'] + '_hp.pkl', 'wb') as file:
#     pickle.dump(hyper_params, file)

# # Loading a dictionary from a file
# with open('/Users/dmitrievsky/Desktop/py files/bots/set_files/' + hyper_params['symbol'] + '_hp.pkl', 'rb') as file:
#     hyper_params = pickle.load(file)
