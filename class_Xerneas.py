import pandas as pd
from keras.preprocessing import sequence
import tensorflow as tf
import numpy as np
import re
from underthesea import word_tokenize


class Xerneas:

    def __init__(self, dfTrain, dfTest):
        self.dfTrain = pd.DataFrame(dfTrain)
        self.dfTest = pd.DataFrame(dfTest)

        self.bang_nguyen_am = [
            ['a', 'à', 'á', 'ả', 'ã', 'ạ', 'a'],
            ['ă', 'ằ', 'ắ', 'ẳ', 'ẵ', 'ặ', 'aw'],
            ['â', 'ầ', 'ấ', 'ẩ', 'ẫ', 'ậ', 'aa'],
            ['e', 'è', 'é', 'ẻ', 'ẽ', 'ẹ', 'e'],
            ['ê', 'ề', 'ế', 'ể', 'ễ', 'ệ', 'ee'],
            ['i', 'ì', 'í', 'ỉ', 'ĩ', 'ị', 'i'],
            ['o', 'ò', 'ó', 'ỏ', 'õ', 'ọ', 'o'],
            ['ô', 'ồ', 'ố', 'ổ', 'ỗ', 'ộ', 'oo'],
            ['ơ', 'ờ', 'ớ', 'ở', 'ỡ', 'ợ', 'ow'],
            ['u', 'ù', 'ú', 'ủ', 'ũ', 'ụ', 'u'],
            ['ư', 'ừ', 'ứ', 'ử', 'ữ', 'ự', 'uw'],
            ['y', 'ỳ', 'ý', 'ỷ', 'ỹ', 'ỵ', 'y']
        ]

        self.nguyen_am_to_ids = dict()

        for i in range(len(self.bang_nguyen_am)):
            for j in range(len(self.bang_nguyen_am[i]) - 1):
                self.nguyen_am_to_ids[self.bang_nguyen_am[i][j]] = (i, j)

        self.dictVocab = dict()

        self.VOCAB_SIZE = -1
        self.MAXLEN = -1
        self.BATCH_SIZE = -1

        self.arr_y = ['Positive', 'Negative', 'Improve']

        self.xTrain = []
        self.yTrain = []
        self.xTest = []
        self.yTest = []

        self.model = None
        self.history = None




    def preprocessDb(self):

        print('Preprocess database')
        self.dfTrain['Verbatim'] = self.dfTrain['Verbatim'].apply(self.text_preprocess)
        self.dfTest['Verbatim'] = self.dfTest['Verbatim'].apply(self.text_preprocess)

        print('Drop NULL value in database')
        self.dfTrain = self.dfTrain.loc[(self.dfTrain['Verbatim'] != '') & (len(self.dfTrain['Verbatim']) > 5)]
        self.dfTest = self.dfTest.loc[(self.dfTest['Verbatim'] != '') & (len(self.dfTest['Verbatim']) > 5)]

        print('Save Preprocessed database')
        self.dfTrain.to_csv('DB_Train_Preprocessed.csv', sep=';', encoding='utf8', index=False)
        self.dfTest.to_csv('DB_Test_Preprocessed.csv', sep=';', encoding='utf8', index=False)


    def createVocabularyList(self):

        arr = np.concatenate(((self.dfTrain['Verbatim'].str.split(' ')).to_numpy().flatten(),
                              (self.dfTest['Verbatim'].str.split(' ')).to_numpy().flatten()), axis=None)
        lstWords = list()
        for a in arr:
            lstWords.extend(a)
        lstWords = list(dict.fromkeys(lstWords))

        _, self.dictVocab = self.one_hot_encoding(lstWords, dict())
        self.VOCAB_SIZE = len(lstWords) + 1


    def dbToArr(self):

        self.xTrain = self.dfTrain['Verbatim'].str.split(' ').to_numpy()
        self.yTrain = self.dfTrain.loc[:, self.arr_y].to_numpy()
        self.xTest = self.dfTest['Verbatim'].str.split(' ').to_numpy()
        self.yTest = self.dfTest.loc[:, self.arr_y].to_numpy()


    def encodingArr(self, arr):

        for idx, val in np.ndenumerate(arr):
            arr[idx], self.dictVocab = self.one_hot_encoding(val, self.dictVocab)

        return sequence.pad_sequences(arr, self.MAXLEN)


    def createTrainingModel(self):
        print('Creating the Model')
        self.model = tf.keras.Sequential([
            tf.keras.layers.Embedding(self.VOCAB_SIZE, 32),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dense(len(self.arr_y), activation='sigmoid')  # sigmoid relu softmax tanh
        ])

        print('Training')
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])  # rmsprop adam

        epochs = 10
        self.history = self.model.fit(self.xTrain, self.yTrain, epochs=epochs, validation_split=0.2)

        results = self.model.evaluate(self.xTest, self.yTest)
        print('loss:{}, acc{}'.format(results[0], results[1]))

        print('Saving model')
        self.model.save('saved_model/textClassification.h5')
        print('Saved model')


    def loadModel(self):
        print('Loading model')
        self.model = tf.keras.models.load_model('saved_model/textClassification.h5')


    def makingPredictions(self, dfPre):

        dfPreEncoded = dfPre.apply(self.text_preprocess)
        arrPre = dfPreEncoded.str.split(' ').to_numpy()
        arrPre = self.encodingArr(arrPre)
        self.predict(dfPre, arrPre)


    def predict(self, dfPre, arrPre):
        for vbtim, textEncoded in zip(dfPre, arrPre):
            pred = np.zeros((1, self.MAXLEN))
            pred[0] = textEncoded
            result = self.model.predict(pred)

            strPre = '\n# Inputted verbatim: {}'.format(vbtim)
            for a, b in zip(self.arr_y, result[0]):
                strPre += '\n- {}: {}%'.format(a, round(b*100, 5))

            print(strPre)


    @staticmethod
    def remove_html(txt):
        return re.sub(r'<[^>]*>', '', txt)


    @staticmethod
    def loaddicchar():
        dic = {}
        char1252 = 'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ'.split('|')
        charutf8 = "à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ".split('|')
        for idx in range(len(char1252)):
            dic[char1252[idx]] = charutf8[idx]
        return dic


    def convert_unicode(self, txt):
        dicchar = self.loaddicchar()

        return re.sub(r'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ',
            lambda x: dicchar[x.group()], txt)


    def vn_word_to_telex_type(self, word):
        bang_ky_tu_dau = ['', 'f', 's', 'r', 'x', 'j']
        dau_cau = 0
        new_word = ''
        for char in word:
            x, y = self.nguyen_am_to_ids.get(char, (-1, -1))
            if x == -1:
                new_word += char
                continue
            if y != 0:
                dau_cau = y
            new_word += self.bang_nguyen_am[x][-1]
        new_word += bang_ky_tu_dau[dau_cau]
        return new_word


    def vn_sentence_to_telex_type(self, sentence):

        words = sentence.split()
        for index, word in enumerate(words):
            words[index] = self.vn_word_to_telex_type(word)
        return ' '.join(words)


    def chuan_hoa_dau_tu_tieng_viet(self, word):
        if not self.is_valid_vietnam_word(word):
            return word

        chars = list(word)
        dau_cau = 0
        nguyen_am_index = []
        qu_or_gi = False
        for index, char in enumerate(chars):
            x, y = self.nguyen_am_to_ids.get(char, (-1, -1))
            if x == -1:
                continue
            elif x == 9:  # check qu
                if index != 0 and chars[index - 1] == 'q':
                    chars[index] = 'u'
                    qu_or_gi = True
            elif x == 5:  # check gi
                if index != 0 and chars[index - 1] == 'g':
                    chars[index] = 'i'
                    qu_or_gi = True
            if y != 0:
                dau_cau = y
                chars[index] = self.bang_nguyen_am[x][0]
            if not qu_or_gi or index != 1:
                nguyen_am_index.append(index)
        if len(nguyen_am_index) < 2:
            if qu_or_gi:
                if len(chars) == 2:
                    x, y = self.nguyen_am_to_ids.get(chars[1])
                    chars[1] = self.bang_nguyen_am[x][dau_cau]
                else:
                    x, y = self.nguyen_am_to_ids.get(chars[2], (-1, -1))
                    if x != -1:
                        chars[2] = self.bang_nguyen_am[x][dau_cau]
                    else:
                        chars[1] = self.bang_nguyen_am[5][dau_cau] if chars[1] == 'i' else self.bang_nguyen_am[9][dau_cau]
                return ''.join(chars)
            return word

        for index in nguyen_am_index:
            x, y = self.nguyen_am_to_ids[chars[index]]
            if x == 4 or x == 8:  # ê, ơ
                chars[index] = self.bang_nguyen_am[x][dau_cau]
                # for index2 in nguyen_am_index:
                #     if index2 != index:
                #         x, y = nguyen_am_to_ids[chars[index]]
                #         chars[index2] = bang_nguyen_am[x][0]
                return ''.join(chars)

        if len(nguyen_am_index) == 2:
            if nguyen_am_index[-1] == len(chars) - 1:
                x, y = self.nguyen_am_to_ids[chars[nguyen_am_index[0]]]
                chars[nguyen_am_index[0]] = self.bang_nguyen_am[x][dau_cau]
                # x, y = nguyen_am_to_ids[chars[nguyen_am_index[1]]]
                # chars[nguyen_am_index[1]] = bang_nguyen_am[x][0]
            else:
                # x, y = nguyen_am_to_ids[chars[nguyen_am_index[0]]]
                # chars[nguyen_am_index[0]] = bang_nguyen_am[x][0]
                x, y = self.nguyen_am_to_ids[chars[nguyen_am_index[1]]]
                chars[nguyen_am_index[1]] = self.bang_nguyen_am[x][dau_cau]
        else:
            # x, y = nguyen_am_to_ids[chars[nguyen_am_index[0]]]
            # chars[nguyen_am_index[0]] = bang_nguyen_am[x][0]
            x, y = self.nguyen_am_to_ids[chars[nguyen_am_index[1]]]
            chars[nguyen_am_index[1]] = self.bang_nguyen_am[x][dau_cau]
            # x, y = nguyen_am_to_ids[chars[nguyen_am_index[2]]]
            # chars[nguyen_am_index[2]] = bang_nguyen_am[x][0]
        return ''.join(chars)


    def is_valid_vietnam_word(self, word):
        chars = list(word)
        nguyen_am_index = -1
        for index, char in enumerate(chars):
            x, y = self.nguyen_am_to_ids.get(char, (-1, -1))
            if x != -1:
                if nguyen_am_index == -1:
                    nguyen_am_index = index
                else:
                    if index - nguyen_am_index != 1:
                        return False
                    nguyen_am_index = index
        return True


    def chuan_hoa_dau_cau_tieng_viet(self, sentence):

        sentence = sentence.lower()
        words = sentence.split()
        for index, word in enumerate(words):
            words[index] = self.chuan_hoa_dau_tu_tieng_viet(word)
        return ' '.join(words)


    @staticmethod
    def remove_stopwords(line):
        # Danh sách stopword
        stopword = set(open('stopwords.txt', encoding='utf8').read().split('\n'))

        words = []
        for word in line.strip().split():
            if word not in stopword:
                words.append(word)
        return ' '.join(words)


    def text_preprocess(self, document):
        # xóa html code
        document = self.remove_html(document)
        # chuẩn hóa unicode
        document = self.convert_unicode(document)
        # chuẩn hóa cách gõ dấu tiếng Việt
        document = self.chuan_hoa_dau_cau_tieng_viet(document)
        # tách từ
        document = word_tokenize(document, format='text')
        # đưa về lower
        document = str(document).lower()
        # xóa các ký tự không cần thiết
        document = re.sub(r'[^\s\wáàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđ_]', ' ', document)
        # xóa khoảng trắng thừa
        document = re.sub(r'\s+', ' ', document).strip()
        # # xóa stopword
        # document = remove_stopwords(document)
        return document

    @staticmethod
    def one_hot_encoding(lstWords, dictVocab):

        if len(dictVocab.keys()) == 0:
            word_encoding = 1
        else:
            word_encoding = len(dictVocab.keys())

        encoding = list()

        for word in lstWords:
            if word in dictVocab:
                code = dictVocab[word]
                encoding.append(code)
            else:
                dictVocab[word] = word_encoding
                encoding.append(word_encoding)
                word_encoding += 1

        return encoding, dictVocab