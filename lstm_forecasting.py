import numpy as np
import os, sys
import itertools
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

def scaler_transform(dataset):
    d_min, d_max = dataset.min(), dataset.max()
    return [(dataset-d_min)/(d_max - d_min), d_min, d_max]

def inv_transform(scaled_dataset, d_min, d_max):
    return scaled_dataset*(d_max - d_min) + d_min

def create_dataset(dataset, LOOK_BACK=1, features_axis=1):#dataset_shape = samples * features
    dataX, dataY = [], []
    
    for i in range(dataset.shape[1]-LOOK_BACK):
        if(features_axis==1):
            sample = dataset[i:(i+LOOK_BACK),:]
            dataX.append(sample)
            dataY.append(dataset[i + LOOK_BACK,:])
        if(features_axis==0):
            sample = dataset[:,i:(i+LOOK_BACK)]
            dataX.append(sample)
            dataY.append(dataset[:,i+LOOK_BACK])
    return np.array(dataX), np.array(dataY)

def LSTM_groningen(def_series, B_temp, LOOK_BACK=2, TRAIN_RATIO=0.67, LSTM_NEURONS = 100, LSTM_EPOCHS = 10, features_axis = 0, NUM_POINTS_PLOT=1):
    
    epochs = def_series.shape[-1]
    def_series_scaled, d_min, d_max = scaler_transform(def_series.values)
    
    dataset = def_series_scaled
    
    train_size = int(epochs * TRAIN_RATIO)
    test_size = epochs - train_size
    train, test = dataset[:,0:train_size], dataset[:,train_size:]  
    
    trainX, trainY = create_dataset(train, LOOK_BACK, features_axis=features_axis)
    testX, testY = create_dataset(test, LOOK_BACK, features_axis=features_axis)
    
    
    num_train_trun_samp = trainX.shape[0]
    num_test_trun_samp = testX.shape[0]
    num_PS_points = trainX.shape[1]
    
    print('Initial training shape')
    print('trainX shape', trainX.shape, 'trainX shape', trainY.shape)
    print('testX.shape', testX.shape, 'testY.shape', testY.shape)
    
    #new training strategy
    trainX = trainX.reshape(trainX.shape[0]*trainX.shape[1], trainX.shape[2], 1)
    trainY = trainY.reshape(trainY.shape[0]*trainY.shape[1])
    
    testX = testX.reshape(testX.shape[0]*testX.shape[1], testX.shape[2], 1)
    testY = testY.reshape(testY.shape[0]*testY.shape[1])
    
    print('New training shape')
    print('trainX shape', trainX.shape, 'trainX shape', trainY.shape)
    print('testX.shape', testX.shape, 'testY.shape', testY.shape)
    
    #***********************
    # model
    #***********************
    checkpoint_path = "training_1/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create a callback that saves the model's weights
    cp_callback = ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)
    
    model = Sequential()
    #Model 2: stack of LSTM layers
    
    model.add(LSTM(LSTM_NEURONS, input_shape=(LOOK_BACK, trainX.shape[2]), return_sequences=True))
    model.add(LSTM(LSTM_NEURONS))   
    model.add(Dense(trainX.shape[2]))
    print(model.summary())
    model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy'])
    history = model.fit(trainX, trainY, epochs=LSTM_EPOCHS, batch_size=50, verbose=1, callbacks=[cp_callback], validation_split=0.33)
    
    
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    
    #***********************
    # make predictions 
    #***********************
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)
    
    print("train_predict_shape is", trainPredict.shape)
    print("test_predict_shape is", testPredict.shape)
    

    
    # invert predictions
    trainPredict = inv_transform(trainPredict, d_min, d_max)
    trainY = inv_transform(trainY, d_min, d_max)
    testPredict = inv_transform(testPredict, d_min, d_max)
    testY = inv_transform(testY, d_min, d_max)
    
    # calculate root mean squared error
    train_error = trainY - trainPredict.flatten()
    test_error = testY - testPredict.flatten()
    trainScore = np.sqrt(np.mean((train_error.mean()-train_error)**2))
    print('Train Score: %.2f RMSE' % (trainScore))
    testScore = np.sqrt(np.mean((test_error.mean()-test_error)**2))
    print('Test Score: %.2f RMSE' % (testScore))
    
    print(np.sqrt(mean_squared_error(trainY, trainPredict.flatten())))
    print(np.sqrt(mean_squared_error(testY, testPredict.flatten())))
    
    trainPredict = trainPredict.reshape(num_train_trun_samp, num_PS_points)
    testPredict = testPredict.reshape(num_test_trun_samp, num_PS_points)
    
    trainPredictPlot = np.empty_like(dataset)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[:,LOOK_BACK:len(trainPredict)+LOOK_BACK] = trainPredict.T
    # shift test predictions for plotting
    testPredictPlot = np.empty_like(dataset)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[:,train_size+LOOK_BACK:epochs] = testPredict.T
    
    # plot baseline and predictions
    BLOCK_COLOURS = ['r', 'g', 'b', 'y', 'c', 'm']
    
    color  = itertools.cycle(BLOCK_COLOURS)
    [plt.plot(def_series.values[i, :].T,  label = 'DePSI deformation', color=next(color)) for i in range(NUM_POINTS_PLOT)]
    
    color  = itertools.cycle(BLOCK_COLOURS)
    [plt.plot(trainPredictPlot[i, :].T, '--',  label = 'Train Prediction', color=next(color)) for i in range(NUM_POINTS_PLOT)]
    
    color  = itertools.cycle(BLOCK_COLOURS)
    [plt.plot(testPredictPlot[i,:].T, '.-',  label = 'Test Prediction', color=next(color)) for i in range(NUM_POINTS_PLOT)]
    plt.xlabel('Epochs')
    plt.ylabel('LOS deformation (mm/y)')
    plt.legend()
    
    plt.show()
    
if __name__=='__main__':
    
    # Tasks 1: 
    #make data frame and store in "defo_df"
    
    lstm_prediction.LSTM_groningen(defo_df, _, LOOK_BACK=25, TRAIN_RATIO=0.6, LSTM_NEURONS = 30, LSTM_EPOCHS = 1, NUM_POINTS_PLOT = 3)
    
