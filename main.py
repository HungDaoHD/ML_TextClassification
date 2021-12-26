import os
# Turn off TensorFlow warning messages in program output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas as pd
from class_Xerneas import Xerneas



isCreateCsv = False

if isCreateCsv:

    print('Read database')
    df_Train = pd.read_csv('Database/DB_Train.csv')
    df_Test = pd.read_csv('Database/DB_Test.csv')
    xer = Xerneas(df_Train, df_Test)
    xer.preprocessDb()

else:

    print('Load Preprocessed database')
    df_Train = pd.read_csv('DB_Train_Preprocessed.csv', sep=';', encoding='utf8')
    df_Test = pd.read_csv('DB_Test_Preprocessed.csv', sep=';', encoding='utf8')
    xer = Xerneas(df_Train, df_Test)


print('Create vocabulary list')
xer.createVocabularyList()


print('Initialize constant variable')
xer.MAXLEN = 250
xer.BATCH_SIZE = 64


print('Encoding database')
xer.dbToArr()
xer.xTrain = xer.encodingArr(xer.xTrain)
xer.xTest = xer.encodingArr(xer.xTest)


isNewTraining = False

if isNewTraining:

    xer.createTrainingModel()

else:

    xer.loadModel()


print('Making Predictions')
dfPre = pd.Series(['ứng dụng cần điều chỉnh lại giao diện dễ nhìn'])
# quận 12 hiện_tại rất ít chi_nhánh techcombank cần đặt thêm chi_nhánh trên các tuyến đường lớn như lê_văn_khương lê_thị_riêng nguyễn_ảnh_thủ hiệp thành
# Không hài lòng với những giao dịch rút tiền mặt tại quầy
# Dịch vụ tốt, nhân viên chăm sóc khách hàng tốt

xer.makingPredictions(dfPre)







print('\nDone')











