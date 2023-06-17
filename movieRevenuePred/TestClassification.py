# imports
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import time
import pickle




# reading the 3 csv files
data1 = pd.read_csv('movies-revenue-test-samples.csv')
data2 = pd.read_csv('movie-voice-actors-test-samples.csv')
data3 = pd.read_csv('movie-director-test-samples.csv')

# loading encoding & transforming the data
CountSaved1 = pickle.load(open('Cencoder1.pkl', 'rb'))
data2 = data2.join(CountSaved1.transform(data2['voice-actor']).add_suffix('_count'))
CountSaved2 = pickle.load(open('Cencoder2.pkl', 'rb'))
data2 = data2.join(CountSaved2.transform(data2['character']).add_suffix('_count'))
CountSaved3 = pickle.load(open('Cencoder3.pkl', 'rb'))
data3 = data3.join(CountSaved3.transform(data3['director']).add_suffix('_count'))

# grouping data
data2.rename(columns = {'voice-actor_count':'voice_actor_count'}, inplace = True)
data2.rename(columns={"movie": "movie_title"}, inplace=True)
data2 = data2.groupby('movie_title')['voice_actor_count','character_count'].agg(list)

# using merge function to merge the 3 tables together
data1 = pd.merge(data1, data2, on='movie_title', how='left')
data3.rename(columns={"name": "movie_title"}, inplace=True)
data1 = pd.merge(data1, data3, on='movie_title', how='left')
print(data1)

# dropping the old columns
# data2.drop('voice-actor', inplace=True, axis=1)
# data2.drop('character', inplace=True, axis=1)
data1.drop('director', inplace=True, axis=1)

# dropping columns
data1.drop('movie_title', inplace=True, axis=1)
data1.drop('character_count', inplace=True, axis=1)

# date
data1['Year'] = pd.DatetimeIndex(data1['release_date']).year
data1['Month'] = pd.DatetimeIndex(data1['release_date']).month
data1['Day'] = pd.DatetimeIndex(data1['release_date']).day
data1.drop('release_date', inplace=True, axis=1)

# dropping columns
data1.drop('Year', inplace=True, axis=1)
data1.drop('Day', inplace=True, axis=1)
# Monthmean = pickle.load(open('Monthmeann .pkl','rb'))
# MPAAmode = pickle.load(open('MPAAmode .pkl','rb'))
# genremode = pickle.load(open('genremode .pkl','rb'))
# filling missing values with the saved model
data1['Month'].fillna(value=6.665226781857451, inplace=True)
data1['MPAA_rating'].fillna(value='PG', inplace=True)
data1['genre'].fillna(value='Comedy', inplace=True)

# if os.path.getsize('MpaaCOL.pkl') > 0:
#     with open('MpaaCOL.pkl', "rb") as f:
#         unpickler = pickle.Unpickler(f)
#         MpaaCOL = unpickler.load()
# one hot encoder for MPAA_rating and genre
# MpaaCOL = open('MpaaCOL.pkl','rb')
# mpaacol = pickle.load(MpaaCOL)
# MpaaCOL.close()
mpaacol = ['MPAA_rating_G', 'MPAA_rating_Not Rated', 'MPAA_rating_PG', 'MPAA_rating_PG-13', 'MPAA_rating_R']
for l in mpaacol:
    data1[l] = np.where(data1['MPAA_rating']==l,1,0)
# genreCOL = []
genreCOL = ['genre_Action', 'genre_Adventure',
       'genre_Black Comedy', 'genre_Comedy', 'genre_Concert/Performance',
       'genre_Documentary', 'genre_Drama', 'genre_Horror', 'genre_Musical',
       'genre_Romantic Comedy', 'genre_Thriller/Suspense', 'genre_Western']
for g in genreCOL:
    data1[g] = np.where(data1['genre']==g,1,0)

data1.drop('genre', inplace=True, axis=1)
data1.drop('MPAA_rating', inplace=True, axis=1)
print(data1)
data1.to_csv('checking.csv')

# working with the voice actor column
data1['voice_actor_count'] = data1['voice_actor_count'].fillna(0)
VoiceActorList = data1['voice_actor_count'].tolist() # list of lists
newlist = []
for List in VoiceActorList:
    summ = 0
    if (List==0):
        newlist.append(0)
    else:
        for element in List:
            summ = summ+element
        newlist.append(summ)
data1['voice_actor_count'] = newlist

# filling data with null values or zero with mean saved
# VAmean = pickle.load(open('VOICEmean.pkl','wb'))
data1['voice_actor_count']=data1['voice_actor_count'].replace(0,8.159593971143149)
# Dirmean = pickle.load(open('Dirmean .pkl','wb'))
data1['director_count'].fillna(value=2.7142857142857086, inplace=True)

# X and Y
Y = data1['MovieSuccessLevel']
data1.drop('MovieSuccessLevel', axis=1, inplace=True)

# Models
# knn
my_knn_model = pickle.load(open("My_KNN_model.sav", 'rb'))
result = my_knn_model.predict(data1)
print("accuracy of KNN: " ,accuracy_score(Y,result))

# kernel rbf
SVM_RBF = pickle.load(open("My_SVM_Rbf_model.sav",'rb'))
prediction = SVM_RBF.predict(data1)
accuracy = np.mean(prediction == Y)
print("Accuracy of SVM rbf", accuracy)

# kernel Linear
SVM_Linear = pickle.load(open("My_SVM_Linear_model.sav",'rb'))
prediction = SVM_Linear.predict(data1)
accuracy = np.mean(prediction == Y)
print("Accuracy of SVM linear", accuracy)

# kernel linear
SVM_Poly = pickle.load(open("My_SVM_Poly_model.sav",'rb'))
prediction = SVM_Poly.predict(data1)
accuracy = np.mean(prediction == Y)
print("Accuracy of SVM poly", accuracy)

# decision tree
decision_tree = pickle.load(open("My_dTree_model.sav",'rb'))
y_pred2 = decision_tree.predict(data1)
print("accuracy of decision tree: " ,accuracy_score(Y,y_pred2))

# Random forest
randomforest = pickle.load(open("My_randomforest_model.sav",'rb'))
y_pred = randomforest.predict(data1)
print("accuracy of random forest: ",accuracy_score(Y,y_pred))

