import os
# Turn off TensorFlow warning messages in program output

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas as pd
import functions as func
from keras.preprocessing import sequence
import tensorflow as tf
import numpy as np




# print('Read database')
# df_Train = pd.read_csv('Database/DB_Train.csv')
# df_Test = pd.read_csv('Database/DB_Test.csv')
#
# print('Preprocess database')
# df_Train['Verbatim'] = df_Train['Verbatim'].apply(func.text_preprocess)
# df_Test['Verbatim'] = df_Test['Verbatim'].apply(func.text_preprocess)
#
# print('Drop NULL value in database')
# df_Train = df_Train.loc[(df_Train['Verbatim'] != '') & (len(df_Train['Verbatim']) > 5)]
# df_Test = df_Test.loc[(df_Test['Verbatim'] != '') & (len(df_Test['Verbatim']) > 5)]
#
# print('Save Preprocessed database')
# df_Train.to_csv('DB_Train_Preprocessed.csv', sep=';', encoding='utf8', index=False)
# df_Test.to_csv('DB_Test_Preprocessed.csv', sep=';', encoding='utf8', index=False)


print('Load Preprocessed database')
df_Train = pd.read_csv('DB_Train_Preprocessed.csv', sep=';', encoding='utf8')
df_Test = pd.read_csv('DB_Test_Preprocessed.csv', sep=';', encoding='utf8')


print('Create vocabulary list')
arr = np.concatenate(((df_Train['Verbatim'].str.split(' ')).to_numpy().flatten(),
                      (df_Test['Verbatim'].str.split(' ')).to_numpy().flatten()), axis=None)
lstWords = list()
for a in arr:
    lstWords.extend(a)
lstWords = list(dict.fromkeys(lstWords))

_, dictVocab = func.one_hot_encoding(lstWords, dict())


print('Initialize constant variable')
VOCAB_SIZE = len(lstWords) + 1
MAXLEN = 250
BATCH_SIZE = 64


print('Encoding database')
train_data = df_Train['Verbatim'].str.split(' ').to_numpy()
train_labels = df_Train.loc[:, ['Positive', 'Negative', 'Improve']].to_numpy()
test_data = df_Test['Verbatim'].str.split(' ').to_numpy()
test_labels = df_Test.loc[:, ['Positive', 'Negative', 'Improve']].to_numpy()

for idx, val in np.ndenumerate(train_data):
    train_data[idx], dictVocab = func.one_hot_encoding(val, dictVocab)

for idx, val in np.ndenumerate(test_data):
    test_data[idx], dictVocab = func.one_hot_encoding(val, dictVocab)

train_data = sequence.pad_sequences(train_data, MAXLEN)
test_data = sequence.pad_sequences(test_data, MAXLEN)

isNewTraining = False

if isNewTraining:
    print('Creating the Model')
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(VOCAB_SIZE, 32),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dense(3, activation='sigmoid')  # sigmoid relu softmax tanh
    ])

    print('Training')
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])  # rmsprop adam

    epochs = 10
    history = model.fit(train_data, train_labels, epochs=epochs, validation_split=0.2)

    results = model.evaluate(test_data, test_labels)
    print(results)

    print('Saving model')
    model.save('saved_model/textClassification.h5')

else:

    print('Loading model')
    model = tf.keras.models.load_model('saved_model/textClassification.h5')


print('Making Predictions')
dfPre = pd.Series(['qu???n 12 hi???n_t???i r???t ??t chi_nh??nh techcombank c???n ?????t th??m chi_nh??nh tr??n c??c tuy???n ???????ng l???n nh?? l??_v??n_kh????ng l??_th???_ri??ng nguy???n_???nh_th??? hi???p th??nh'])
# 'Kh??ng h??i l??ng v???i nh???ng giao d???ch r??t ti???n m???t t???i qu???y'
# 'D???ch v??? t???t, nh??n vi??n ch??m s??c kh??ch h??ng t???t'

dfPre = dfPre.apply(func.text_preprocess)

testArr = dfPre.str.split(' ').to_numpy()

for idx, val in np.ndenumerate(testArr):
    testArr[idx], dictVocab = func.one_hot_encoding(val, dictVocab)

testArr = sequence.pad_sequences(testArr, MAXLEN)


def predict(encoded_arr):
    for encoded_text in encoded_arr:
        pred = np.zeros((1, 250))
        pred[0] = encoded_text
        result = model.predict(pred)
        print(result[0])


predict(testArr)

print('Done')











