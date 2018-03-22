import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import label_binarize
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation
from keras.layers.core import Dropout
from keras.regularizers import l2

def _model():
    model = Sequential()
    model.add(Dense(output_dim=148, input_dim=148, W_regularizer=l2(0.0001)))
    model.add(Activation('relu'))
    model.add(Dropout(0.8))
    model.add(Dense(output_dim=148, input_dim=148, W_regularizer=l2(0.0001)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(output_dim=148, input_dim=148, W_regularizer=l2(0.0001)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(output_dim=148, input_dim=148, W_regularizer=l2(0.0001)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(output_dim=1, input_dim=148, W_regularizer=l2(0.0001)))
    model.add(Activation('linear'))
    return model

def main():
    # load Data
    df = pd.read_hdf('data.h5', key='train_reg', mode='r').dropna()
    df_val = pd.read_hdf('data.h5', key='test_reg', mode='r').dropna()

    # set data
    trX, trY = np.array(df.drop('label', axis=1)), np.array(df['label'])
    valX, valY = np.array(df_val.drop('label', axis=1)), np.array(df_val['label'])

    # set model
    model = _model()

    # load autoencoder
    encoder0 = load_model('encoder0.h5')
    encoder1 = load_model('encoder1.h5')
    encoder2 = load_model('encoder2.h5')
    encoder3 = load_model('encoder3.h5')
    encoder4 = load_model('encoder4.h5')

    # set initial weights
    w = model.get_weights()
    w[0] = encoder0.get_weights()[0]
    w[1] = encoder0.get_weights()[1]
    w[2] = encoder1.get_weights()[0]
    w[3] = encoder1.get_weights()[1]
    w[4] = encoder2.get_weights()[0]
    w[5] = encoder2.get_weights()[1]
    w[6] = encoder3.get_weights()[0]
    w[7] = encoder3.get_weights()[1]
    w[8] = encoder4.get_weights()[0]
    w[9] = encoder4.get_weights()[1]
    model.set_weights(w)

    # compile
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])

    # training
    history = model.fit(trX, trY, nb_epoch=1000, verbose=2, batch_size=32, validation_data=(valX, valY))

    # save model
    model.save('DNN_R.h5')

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

if __name__ == '__main__':
    main()
