# imports
from sklearn import svm
import numpy as np
import pickle
import time
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import category_encoders as ce
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# reading the 3  csv files
data1 = pd.read_csv('movies-revenue-classification.csv')
data2 = pd.read_csv('movie-voice-actors.csv')
data3 = pd.read_csv('movie-director.csv')

# performing some of the preproccesing on data 2 before merging
# encoding
count_encoder1 = ce.CountEncoder()
count_encoder1.fit(data2['voice-actor'])
CountSaved1 = pickle.dump(count_encoder1, open('Cencoder1.pkl','wb'))
data2 = data2.join(count_encoder1.transform(data2['voice-actor']).add_suffix('_count'))

count_encoder2 = ce.CountEncoder()
count_encoder2.fit(data2['character'])
CountSaved2 = pickle.dump(count_encoder2, open('Cencoder2.pkl','wb'))
data2 = data2.join(count_encoder2.transform(data2['character']).add_suffix('_count'))

count_encoder3 = ce.CountEncoder()
count_encoder3.fit(data3['director'])
CountSaved3 = pickle.dump(count_encoder3, open('Cencoder3.pkl','wb'))
data3 = data3.join(count_encoder3.transform(data3['director']).add_suffix('_count'))


data2.drop('voice-actor', inplace=True, axis=1)
data2.drop('character', inplace=True, axis=1)
data3.drop('director', inplace=True, axis=1)

# grouping data
data2.rename(columns = {'voice-actor_count':'voice_actor_count'}, inplace = True)
data2.rename(columns={"movie": "movie_title"}, inplace=True)
data2 = data2.groupby('movie_title')['voice_actor_count','character_count'].agg(list)


# using merge function to merge the 3 tables together
data1 = pd.merge(data1, data2, on='movie_title', how='left')
data3.rename(columns={"name": "movie_title"}, inplace=True)
data1 = pd.merge(data1, data3, on='movie_title', how='left')


# checking the unique values in each column
print("No of unique values in genre:",data1.genre.nunique())
print("No of unique values in director:",data1.director_count.nunique())
print("No of unique values in movie_title:",data1.movie_title.nunique())
print("No of unique values in MPAA_rating:",data1.MPAA_rating.nunique())

# dropping columns that we don't need since they have large number of unique values
data1.drop('movie_title', inplace=True, axis=1)
data1.drop('character_count', inplace=True, axis=1)

# date
data1['Year'] = pd.DatetimeIndex(data1['release_date']).year
data1['Month'] = pd.DatetimeIndex(data1['release_date']).month
data1['Day'] = pd.DatetimeIndex(data1['release_date']).day
data1.drop('release_date', inplace=True, axis=1)

# Plotting relation of each column with Y
# sns.countplot('director_count',data=data1,hue='MovieSuccessLevel')
# sns.countplot('genre',data=data1,hue='MovieSuccessLevel')
# sns.countplot('MPAA_rating',data=data1,hue='MovieSuccessLevel')
# sns.countplot('Year',data=data1,hue='MovieSuccessLevel')
# sns.countplot('Day',data=data1,hue='MovieSuccessLevel')
# sns.countplot('Month', data=data1, hue='MovieSuccessLevel')

# dropping columns
data1.drop('Year', inplace=True, axis=1)
data1.drop('Day', inplace=True, axis=1)
# data1.drop('Month', inplace=True, axis=1)
# filling null values
data1['Month'].fillna(value=data1['Month'].mean(), inplace=True)
data1['MPAA_rating'].fillna(value=data1['MPAA_rating'].mode()[0], inplace=True)
data1['genre'].fillna(value=data1['genre'].mode()[0], inplace=True)

Monthmeann = 6.665226781857451
Monthmean = pickle.dump(Monthmeann, open('Monthmeann .pkl','wb'))
MPAAmodee = 'PG'
MPAAmode = pickle.dump(MPAAmodee, open('MPAAmode .pkl','wb'))
data1['genre'].fillna(value=data1['genre'].mode()[0], inplace=True)
genremodee = 'Comedy'
genremode = pickle.dump(genremodee, open('genremode .pkl','wb'))



# one hot encoder for MPAA_rating and genre
encoder = OneHotEncoder()
data1 = pd.get_dummies(data1, columns=['MPAA_rating'])
MpaaList = ['MPAA_rating_G', 'MPAA_rating_Not Rated', 'MPAA_rating_PG', 'MPAA_rating_PG-13', 'MPAA_rating_R']
MpaaCOL = pickle.dump(MpaaList, open('MpaaCOL.pkl','wb'))

encoder2 = OneHotEncoder()
data1 = pd.get_dummies(data1, columns=['genre'])
genreList = ['genre_Action', 'genre_Adventure',
       'genre_Black Comedy', 'genre_Comedy', 'genre_Concert/Performance',
       'genre_Documentary', 'genre_Drama', 'genre_Horror', 'genre_Musical',
       'genre_Romantic Comedy', 'genre_Thriller/Suspense', 'genre_Western']
genreCOL = pickle.dump(genreList, open('genreCOL.pkl','wb'))

# filling the null values with zeroes
data1['voice_actor_count'] = data1['voice_actor_count'].fillna(0)

# working with the voice actor column
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

# filling data with null values or zero with mean
data1['voice_actor_count']=data1['voice_actor_count'].replace(0, data1['voice_actor_count'].mean())
print(data1['voice_actor_count'].mean())
VOICEmean = 8.159593971143149
VAmean = pickle.dump(VOICEmean, open('VOICEmean.pkl','wb'))
data1['director_count'].fillna(value=data1['director_count'].mean(), inplace=True)
print(data1['director_count'].mean())
Dirmean = 2.7142857142857086
Dirmean = pickle.dump(Dirmean, open('Dirmean .pkl','wb'))
# data1.drop('voice_actor_count', axis=1, inplace=True)

# train test split
Y = data1['MovieSuccessLevel']
data1. drop('MovieSuccessLevel', axis=1, inplace=True)
Xtrain, Xtest, Ytrain, Ytest = train_test_split(data1, Y, random_state=42, test_size=0.2, shuffle=True)

# model
# kernel rbf
SVM = svm.SVC(kernel='rbf',C=4)
trainstart_time = time.time()
SVM.fit(Xtrain, Ytrain)
SVM_RBF = pickle.dump(SVM, open("My_SVM_Rbf_model.sav",'wb'))
trainend_time = time.time()

teststart_time = time.time()
prediction = SVM.predict(Xtest)
testend_time = time.time()
accuracy = np.mean(prediction == Ytest)
print("Train time of SVM rbf", trainstart_time -trainend_time)
print("Accuracy of SVM rbf", accuracy)
print("Test time of SVM rbf", teststart_time -testend_time)
#
# kernel poly
SVM = svm.SVC(kernel='poly',C=0.1,gamma= 0.9,degree=2)
SVM.fit(Xtrain, Ytrain)
SVM_Poly = pickle.dump(SVM, open("My_SVM_Poly_model.sav",'wb'))
prediction = SVM.predict(Xtest)
accuracy = np.mean(prediction == Ytest)
print("Accuracy of SVM poly", accuracy)
#
# kernel linear
SVM = svm.SVC(kernel='linear',C=0.5)
SVM.fit(Xtrain, Ytrain)
SVM_Linear = pickle.dump(SVM, open("My_SVM_Linear_model.sav",'wb'))
prediction = SVM.predict(Xtest)
accuracy = np.mean(prediction == Ytest)
print("Accuracy of SVM linear", accuracy)
#
# decision tree
dTree = DecisionTreeClassifier()
dTree.fit(Xtrain,Ytrain)
decision_tree = pickle.dump(dTree, open("My_dTree_model.sav",'wb'))
y_pred2 = dTree.predict(Xtest)
print("accuracy of decision tree: " ,accuracy_score(Ytest,y_pred2))

#KNN
trainstart_time = time.time()
classifier = KNeighborsClassifier(n_neighbors =8, metric = 'manhattan', p = 5)
classifier.fit(Xtrain, Ytrain)


# Save the trained model as a pickle string.
saved_model = pickle.dump(classifier, open("My_KNN_model.sav",'wb'))
trainend_time = time.time()
print('Model is saved into to disk successfully Using Pickle')
my_knn_model = pickle.load(open("My_KNN_model.sav", 'rb'))
teststart_time = time.time()
result = my_knn_model.predict(Xtest)

testend_time = time.time()
print("Train time of knn", trainstart_time -trainend_time)
print("accuracy of KNN: " ,accuracy_score(Ytest,result))
print("Test time of knn", teststart_time -testend_time)
# bar graph for knn
Plotdaata=['accuracy','training time','test time']
plotnum = [accuracy, abs(trainstart_time -trainend_time),abs(teststart_time -testend_time)]
plt.bar(Plotdaata, plotnum)
plt.title("data")
plt.ylabel("time")
plt.show()

# # Random forest
rf = RandomForestClassifier()
rf.fit(Xtrain,Ytrain)
randomforest = pickle.dump(rf, open("My_randomforest_model.sav",'wb'))
y_pred = rf.predict(Xtest)
print("accuracy of random forest: ",accuracy_score(Ytest,y_pred))

