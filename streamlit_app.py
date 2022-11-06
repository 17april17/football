from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np # Библиотека работы с массивами
import pandas as pd # Библиотека pandas
from sklearn.preprocessing import LabelEncoder, StandardScaler # Функции для нормализации данных
from sklearn import preprocessing # Пакет предварительной обработки данных
import streamlit as st
import io
import requests

params=st.experimental_get_query_params()

date=params.get('date')[0]
id_match=params.get('id_match')[0]
tournament=params.get('tournament')[0]
team_1=params.get('team_1')[0]
team_2=params.get('team_2')[0]
elapsed_time=params.get('elapsed_time')[0]
score_1=params.get('score_1')[0]
score_2=params.get('score_2')[0]
coef_1=params.get('coef_1')[0]
coef_x=params.get('coef_x')[0]
coef_2=params.get('coef_2')[0]
total=params.get('total')[0]
coef_total_over=params.get('coef_total_over')[0]
yellow_card_1=params.get('yellow_card_1')[0]
yellow_card_2=params.get('yellow_card_2')[0]
red_card_1=params.get('red_card_1')[0]
red_card_2=params.get('red_card_2')[0]
corner_1=params.get('corner_1')[0]
corner_2=params.get('corner_2')[0]
shots_on_1=params.get('shots_on_1')[0]
shots_on_2=params.get('shots_on_2')[0]
shots_off_1=params.get('shots_off_1')[0]
shots_off_2=params.get('shots_off_2')[0]
attacks_1=params.get('attacks_1')[0]
attacks_2=params.get('attacks_2')[0]
dan_attacks_1=params.get('dan_attacks_1')[0]
dan_attacks_2=params.get('dan_attacks_2')[0]
penalty_1=params.get('penalty_1')[0]
penalty_2=params.get('penalty_2')[0]
free_kick_1=params.get('free_kick_1')[0]
free_kick_2=params.get('free_kick_2')[0]
sub_1=params.get('sub_1')[0]
sub_2=params.get('sub_2')[0]
link=params.get('link')[0]

st.markdown('## '+date)
st.markdown('### '+tournament)
st.markdown('## '+team_1+' - '+team_2)
st.markdown('### Тотал больше: '+total)
st.markdown('### Коэф.: '+coef_total_over)
st.markdown('### Счёт: '+score_1+':'+score_2)

df = pd.read_csv('FootballLiveTotal.csv', encoding= 'cp1251', sep=';', header=0, index_col=0) # Загружаем базу

df.loc[-1] = [date,id_match,tournament,team_1,team_2,'какой-то матч',elapsed_time,score_1,score_2,coef_1,coef_x,coef_2,total,coef_total_over,yellow_card_1,yellow_card_2,red_card_1,red_card_2,corner_1,corner_2,shots_on_1,shots_on_2,shots_off_1,shots_off_2,attacks_1,attacks_2,dan_attacks_1,dan_attacks_2,penalty_1,penalty_2,free_kick_1,free_kick_2,sub_1,sub_2,0,0,0]

dataset = df.values                 # Берем только значения массива(без индексов)
X = dataset[:,6:-1].astype(float)   # Присваиваем им тип данных - float данным

#Нормируем данные
X_ElapsedTime=preprocessing.scale(X[:,0])
X_Score1=preprocessing.scale(X[:,1])
X_Score2=preprocessing.scale(X[:,2])
X_Coef1=preprocessing.scale(X[:,3])
X_CoefX=preprocessing.scale(X[:,4])
X_Coef2=preprocessing.scale(X[:,5])
X_Total=preprocessing.scale(X[:,6])
X_CoefTotalOver=preprocessing.scale(X[:,7])
X_YellowCard1=preprocessing.scale(X[:,8])
X_YellowCard2=preprocessing.scale(X[:,9])
X_RedCard1=preprocessing.scale(X[:,10])
X_RedCard2=preprocessing.scale(X[:,11])
X_Corner1=preprocessing.scale(X[:,12])
X_Corner2=preprocessing.scale(X[:,13])
X_ShotsOn1=preprocessing.scale(X[:,14])
X_ShotsOn2=preprocessing.scale(X[:,15])
X_ShotsOff1=preprocessing.scale(X[:,16])
X_ShotsOff2=preprocessing.scale(X[:,17])
X_Attacks1=preprocessing.scale(X[:,18])
X_Attacks2=preprocessing.scale(X[:,19])
X_DanAttacks1=preprocessing.scale(X[:,20])
X_DanAttacks2=preprocessing.scale(X[:,21])
X_Penalty1=preprocessing.scale(X[:,22])
X_Penalty2=preprocessing.scale(X[:,23])
X_FreeKick1=preprocessing.scale(X[:,24])
X_FreeKick2=preprocessing.scale(X[:,25])
X_Sub1=preprocessing.scale(X[:,26])
X_Sub2=preprocessing.scale(X[:,27])

#Объединяем нормированные данные
X_All = np.hstack((X_ElapsedTime.reshape(-1,1),
                        X_Score1.reshape(-1,1),
                        X_Score2.reshape(-1,1),
                        #X_Coef1.reshape(-1,1),
                        #X_CoefX.reshape(-1,1),
                        #X_Coef2.reshape(-1,1),
                        X_Total.reshape(-1,1),
                        #X_CoefTotalOver.reshape(-1,1),
                        X_YellowCard1.reshape(-1,1),
                        X_YellowCard2.reshape(-1,1),
                        X_RedCard1.reshape(-1,1),
                        X_RedCard2.reshape(-1,1),
                        X_Corner1.reshape(-1,1),
                        X_Corner2.reshape(-1,1),
                        X_ShotsOn1.reshape(-1,1),
                        X_ShotsOn2.reshape(-1,1),
                        X_ShotsOff1.reshape(-1,1),
                        X_ShotsOff2.reshape(-1,1),
                        X_Attacks1.reshape(-1,1),
                        X_Attacks2.reshape(-1,1),
                        X_DanAttacks1.reshape(-1,1),
                        X_DanAttacks2.reshape(-1,1),
                        X_Penalty1.reshape(-1,1),
                        X_Penalty2.reshape(-1,1),
                        X_FreeKick1.reshape(-1,1),
                        X_FreeKick2.reshape(-1,1),
                        X_Sub1.reshape(-1,1),
                        X_Sub2.reshape(-1,1)))
X_pred_All=X_All[-1].reshape(1,-1)

model = keras.models.load_model('my_model_10.h5')

X_Pre_Info = (df['fldTournament']) #Выборка текстовых данных

#Преобразовываем текстовые данные в числовые/векторные

maxWordsCount = 100 #Определяем максимальное количество слов/индексов, учитываемое при обучении текстов

tokenizer = Tokenizer(num_words=maxWordsCount, 
                      filters='!"#$%&()*+,-–—./:;<=>?@[\\]^_`{|}~\t\n\xa0', 
                      lower=True, 
                      split=' ', 
                      oov_token='unknown', 
                      char_level=False)
tokenizer.fit_on_texts(X_Pre_Info) #"скармливаем" наши тексты, т.е даём в обработку методу, который соберет словарь частотности
items = list(tokenizer.word_index.items())  #Вытаскиваем индексы слов для просмотра
#Переводим в индексы выборку
X_Info_Indexes = tokenizer.texts_to_sequences(X_Pre_Info)
#Преобразовываем обучающую выборку из последовательности индексов в матрицы нулей и единиц по принципу Bag of Words
X_Info = tokenizer.sequences_to_matrix(X_Info_Indexes) #Подаем X_Info_Indexes в виде списка чтобы метод успешно сработал

X_Info_1 = X_Info[-1].reshape(1,-1)

prediction = model.predict([X_pred_All,X_Info_1])
st.markdown('### Вероятность следующего гола: '+str(round(prediction[0,0]*100,2))+'%')

if prediction[0,0]*100>=50 and float(coef_total_over)>=2:
  text_message=id_match+'\n'+date+'\n'+tournament+'\n'+team_1+' - '+team_2+'\nСчет: '+score_1+':'+score_2+'\nТотал больше: '+total+'\nКоэф.: '+coef_total_over+'\nВероятн.след.гола: '+str(round(prediction[0,0]*100,2))+'%'+'\n'+link
  requests.get('https://api.telegram.org/bot5333091819:AAFvBqeHc_6yNxv6T6LuMKpHV7F-6464_f4/sendMessage?chat_id=237194020&text='+text_message)
