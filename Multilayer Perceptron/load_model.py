import pandas as pd
from sklearn.preprocessing import MinMaxScaler


training_data_df = pd.read_csv( './DataSet/sales_data_training.csv', dtype=float )
training_data_df.head()


X_training = training_data_df.drop('total_earnings', axis=1).values
Y_training = training_data_df[ ['total_earnings'] ].values
print (X_training.shape, Y_training.shape)


test_data_df = pd.read_csv('./DataSet/sales_data_test.csv',dtype=float)
test_data_df.head()


X_test = test_data_df.drop('total_earnings', axis=1).values
Y_test = test_data_df[ ['total_earnings'] ].values
print ( X_test.shape, Y_test.shape )


X_scaler = MinMaxScaler(feature_range=(0,1))
Y_scaler = MinMaxScaler(feature_range=(0,1) )



X_scaled_training = X_scaler.fit_transform( X_training )
Y_scaled_training = Y_scaler.fit_transform( Y_training )



X_scaled_test = X_scaler.transform( X_test )
Y_scaled_test = Y_scaler.transform( Y_test )


print ("Y values can be obtained by multiplying {:,.4f} and adding {:,.2f} ".format(Y_scaler.scale_[0], Y_scaler.min_[0]) )

data = {
    'X_train': X_scaled_training,
    'X_test': X_scaled_test,
    'Y_train': Y_scaled_training,
    'Y_test': Y_scaled_test,
    'MulFactor': Y_scaler.scale_[0],
    'AddFactor': Y_scaler.min_[0]
}
