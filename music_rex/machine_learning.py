# PUT ALL YOUR NON-FUNCTION CODE OVER HERE
# EGS: IMPORT STATEMENTS, LOADING PICKLE FILES / MODELS, DATASET/JSON PROCESSING, ETC.

# REMEMBER TO PLACE YOUR FILES (.PICKLE ETC.) IN THE FOLDER ABOVE THIS ONE I.E.
# IN THE SAME FOLDER AS RUN_APP.PY

from collections import namedtuple

# THIS IS YOUR MAIN FUNCTION!
def recommend_similar_songs(audio_features, lyrics_features):
    # PUT YOUR FUNCTION CODE HERE!
    Song = namedtuple("Song", ["artist", "title"])

    song_1 = Song(artist="kanye west", title='i am a god')
    song_2 = Song(artist="linkin park", title="crawling")
    # etc.

# Return your results as a dict containing a key called 'playlist' , which contains a list of the song tuples.
    final_results = [song_1._asdict(), song_2._asdict()] # this example uses only 2 songs but you need to return 10 :)```
    return final_results

# THIS FUNCTION CONVERTS THE AUDIO FEATURES INTO A LIST BEFORE SENDING THEM TO
# recommend_similar_songs
def get_similar_songs(features, lyrics):
  print(features)
  print(lyrics)

  # features is a dict. convert it to a list using the same order as the assignments...
  audio_feature_headers = ['key', 'energy', 'liveness', 'tempo', 'speechiness', 'acousticness', 'instrumentalness', 'time_signature', 'duration', 'loudness', 'valence', 'danceability', 'mode', 'time_signature_confidence', 'tempo_confidence', 'key_confidence', 'mode_confidence']
  audio_features_list = []

#   for audio_feature_name in audio_feature_headers:
#       audio_features_list.append(features[audio_feature_name])

  # Provide the lyrics as is; a string

  return recommend_similar_songs(1, 2)
