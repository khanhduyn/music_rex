# PUT ALL YOUR NON-FUNCTION CODE OVER HERE
# EGS: IMPORT STATEMENTS, LOADING PICKLE FILES / MODELS, DATASET/JSON PROCESSING, ETC.

import pickle
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA, SparsePCA, TruncatedSVD
from sklearn.base import TransformerMixin
from xgboost import XGBClassifier
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA, SparsePCA, TruncatedSVD
from sklearn.base import TransformerMixin

from sklearn.metrics import jaccard_similarity_score

#######################################

def extract_lyric_features(X):
    return X['lyrics_features']

def extract_audio_features(X):
    return X.drop('lyrics_features', axis=1)

get_lyric_features = FunctionTransformer(extract_lyric_features, validate=False)
get_audio_features = FunctionTransformer(extract_audio_features, validate=False)

from string import punctuation
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from nltk.stem.snowball import SnowballStemmer

def clean_text_train(text):
    processing_text = text.lower()
    translate_func = str.maketrans('', '', punctuation)
    processing_text = processing_text.translate(translate_func)
    stemmer = SnowballStemmer('english')
    clean_token = [ stemmer.stem(word) for word in processing_text.split() if word not in ENGLISH_STOP_WORDS ]
    processing_text = ' '.join(clean_token)
    return processing_text

#################################################################################
# from xgboost import XGBClassifier
# step_audio_lyric_tfidf_XGB = [
#     ('Feature_Union', FeatureUnion([
#         ('lyric_features', Pipeline([
#             ('selector', get_lyric_features),
#             ('vect', TfidfVectorizer())
#         ])),
#         ('audio_features', Pipeline([
#             ('selector', get_audio_features)
#         ]))
#     ])),
#     ('clf', XGBClassifier())
# ]
# pipe_tfidf = Pipeline(step_audio_lyric_tfidf_XGB)


# train_database = pd.read_pickle('data/my_database_new.pickle')
feature_col_names = [
                        'key',
                        'energy',
                        'liveliness',
                        'tempo',
                        'speechiness',
                        'acousticness',
                        'instrumentalness',
                        'time_signature',
                        'duration',
                        'loudness',
                        'valence',
                        'danceability',
                        'mode',
                        'time_signature_confidence',
                        'tempo_confidence',
                        'key_confidence',
                        'mode_confidence'
                    ]
# n_df_audio_features = pd.DataFrame(train_database.loc[:, 'audio_features'].tolist(), columns=feature_col_names)
# n_df_audio_features_genres = pd.concat([train_database['genres'], n_df_audio_features], axis=1)



# n_df_audio_features_genres = pd.concat([train_database['genres'], n_df_audio_features], axis=1)
# clean_text_lambda = lambda x: clean_text_train(' '.join(x)) if len(x) > 0 else np.nan
# n_df_lyrics_features = train_database.loc[:, 'lyrics_features'].apply(clean_text_lambda)


# n_df_audio_features_lyrics_genres = pd.concat([train_database.genres, n_df_lyrics_features, n_df_audio_features], axis = 1)

# from sklearn.model_selection import train_test_split
# df_audio_features_lyrics_genres_dropna = n_df_audio_features_lyrics_genres.dropna()
# X = df_audio_features_lyrics_genres_dropna.drop('genres',axis=1)
# y = df_audio_features_lyrics_genres_dropna.genres
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# pipe_tfidf.fit(X_train, y_train)
# print(classification_report(y_test, pipe_tfidf.predict(X_test)))
# predict_genres = pipe_tfidf

# pickle.dump(pipe_tfidf, open('omg_tfidf.pkl', 'wb'))
predict_genres = pickle.load(open('omg_tfidf.pkl', 'rb'))
# print(classification_report(y_test, predict_genres.predict(X_test)))
#################################################################################



# get_lyric_features = FunctionTransformer(lambda x: x['lyrics_features'], validate=False)
# get_audio_features = FunctionTransformer(lambda x: x.drop('lyrics_features', axis=1), validate=False)
#######################################

# predict_genres = pickle.load(open('data/pipe_tfidf_xgb.pkl', 'rb'))
predict_moods_lyric  = pickle.load(open('data/music_rec/lyrics_predict_moods.chain.pickle', 'rb'))
predict_moods_audio = pickle.load(open('data/music_rec/audio_predict_moods.chain.pickle', 'rb'))

#######################################


my_database    = pickle.load(open('data/music_rec/my_database_new.pickle', 'rb'))
df_moods = my_database.moods.str.join(',').str.get_dummies(sep=',')
df_genres_moods = pd.concat((my_database.genres, df_moods), axis=1)


#######################################
from string import punctuation
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from nltk.stem.snowball import SnowballStemmer

translator = str.maketrans('','', punctuation)
stemmer = SnowballStemmer('english')

def clean_text(raw_text):
    
    clean_words = []
    
    raw_text = raw_text.lower()
    raw_text = raw_text.translate(translator)
    
    for word in raw_text.split():
        
        if word not in ENGLISH_STOP_WORDS:
            
            clean_words.append(stemmer.stem(word))
    
    return " ".join(clean_words)

def clean_text_dks(text):
    processing_text = text.lower()
    translate_func = str.maketrans('', '', punctuation)
    processing_text = processing_text.translate(translate_func)
    stemmer = SnowballStemmer('english')
    clean_token = [ stemmer.stem(word) for word in processing_text.split() if word not in ENGLISH_STOP_WORDS ]
    processing_text = ' '.join(clean_token)
    return processing_text


#######################################    


from collections import namedtuple
Song = namedtuple("Song", ["artist", "title"])

# REMEMBER TO PLACE YOUR FILES (.PICKLE ETC.) IN THE FOLDER ABOVE THIS ONE I.E.
# IN THE SAME FOLDER AS RUN_APP.PY

# THIS IS YOUR MAIN FUNCTION!
def recommend_similar_songs(audio_features, lyrics_features):
    genre = []
    moods = []
    similarity_genres = []
    print(audio_features)
    print(type(audio_features))
    audio_features = np.array(audio_features)
    
    if lyrics_features is not None:
        
        lyrics_features_clean = clean_text(lyrics_features)
        
        moods_1 = predict_moods_audio.predict_proba(audio_features.reshape(1, -1))
        moods_2 = predict_moods_lyric.predict_proba([lyrics_features_clean])
        moods = (moods_1*2 + moods_2*1)*1/3
        
        df_audio_features = pd.DataFrame([audio_features], columns=feature_col_names)
        df_lyrics_features = pd.DataFrame([lyrics_features_clean], columns=['lyrics_features'])
        df = pd.concat([df_lyrics_features, df_audio_features], axis=1)
        genre = predict_genres.predict(df)
        print(genre)

    else:
        moods = predict_moods_audio.predict_proba(audio_features.reshape((1, -1)))
    
    if len(genre) > 0:
        similarity_genres = my_database.genres.apply(lambda x: jaccard_similarity_score(genre, [x]))
        
    similarity = df_moods.apply(lambda x: cosine_similarity(moods, np.array(x).reshape(1, -1))[0, 0], 
                                       axis=1)
    similarity = similarity*2 + similarity_genres
    similarity = similarity.sort_values(ascending=False)[0:50]
    
    top_10 = my_database.iloc[similarity.sample(10).index]
    
    result = [ Song(artist=row["artist"], title=row['name'])._asdict() for idx, row in top_10.iterrows()]
    final_result_dictionary = dict(playlist=result)
    print(result)
    return result

# THIS FUNCTION CONVERTS THE AUDIO FEATURES INTO A LIST BEFORE SENDING THEM TO
# recommend_similar_songs
def get_similar_songs(features, lyrics):
  print(features)
  print(lyrics)

  # features is a dict. convert it to a list using the same order as the assignments...
  audio_feature_headers = ['key', 'energy', 'liveness', 'tempo', 'speechiness', 'acousticness', 'instrumentalness', 'time_signature', 'duration', 'loudness', 'valence', 'danceability', 'mode', 'time_signature_confidence', 'tempo_confidence', 'key_confidence', 'mode_confidence']
  audio_features_list = []

  for audio_feature_name in audio_feature_headers:
      audio_features_list.append(features[audio_feature_name])

  # Provide the lyrics as is; a string
  final_dict = recommend_similar_songs(audio_features_list, lyrics)

  return final_dict
