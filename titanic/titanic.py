import tensorflow_datasets as tfds
import pandas as pd
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

def get_data():
    ds_train_tf, ds_validation_tf,ds_test_tf = tfds.load(
        name='titanic',
        split=['train[:70%]', 'train[70%:80%]', 'train[80%:90%]'],
        as_supervised=True
    )
    print(type(ds_train_tf))
    n = 1
    flag = 0
    for x, y in ds_train_tf.as_numpy_iterator():
        # print(f"Predictors X={x}")
        # print(f"Callbacks y={y}")
        if flag == 0:
            train_DataFrame = pd.DataFrame.from_dict(x ,orient='index').T
            print('First Row')
            print(train_DataFrame)
            print('-------------------------------------------')
            flag = 1
        else:
            iter_DataFrame = pd.DataFrame(x, index= [n])
            train_DataFrame = pd.concat([train_DataFrame, iter_DataFrame])
            n = n + 1
        
    print(train_DataFrame)

    _list = [y for x,y in ds_train_tf.as_numpy_iterator()]
    
    callbacks_DataFrame = pd.DataFrame()
    callbacks_DataFrame['Class'] = _list

    print(callbacks_DataFrame)
        
    return train_DataFrame, callbacks_DataFrame

   
def get_prediction_data():
    df_pred = pd.read_csv('titanic/DL_Task_3_Titain_reserved.csv', na_values=b'Unknown', dtype= 'str', encoding='utf-8')
    # print(df_pred)
    return df_pred

def data_edit(df, df_c):
    df['Class'] = df_c['Class']
    df['age'] = np.where(df['age'] < 0, 0, df['age'])
    print(df)

def plotter(df, df_c):
    age = df['age'].tolist()
    print(f'Максимальный возраст пассажира титаника: {max(age)},Минимальный: {min(age)}')
    bins = [0,10,20,30,40,50,60,70,80]
    plt.hist(df['age'], bins = bins)
    plt.show()
    # plt.hist(df['age'])





def main():
    train, callback = get_data()
    get_prediction_data()
    data_edit(train, callback)
    plotter(train, callback)

if __name__ == '__main__':
    main()