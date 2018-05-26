from __future__ import print_function    # (at top of module)
from spotipy.oauth2 import SpotifyClientCredentials
from musixmatch import Musixmatch
import json
import spotipy
import requests
import os

from logger import logger

CLIENT_ID='37b5edf45b91479d9d01614b6f8de4b2'
CLIENT_SECRET='0ebe2de09ed84139ba1a6fb237076b82'
MUSIXMATCH_KEY='e09ead8b7feaea2847f5195cbcac1e73'

class MusicRex(object):
    def __init__(self, client_id=None, client_secret=None):
        # if not client_id:
        #     client_id = os.getenv('CLIENT_ID')
        # if not client_secret:
        #     client_secret = os.getenv('CLIENT_SECRET')

        # if not client_id or not client_secret:
        #     raise Exception("No CLIENT_ID or CLIENT_SECRET in environment.")

        self.musixmatch_key = MUSIXMATCH_KEY
        self.client_id = CLIENT_ID
        self.client_secret = CLIENT_SECRET
        self.client_credentials_manager = SpotifyClientCredentials(self.client_id, self.client_secret)
        self.sp = spotipy.Spotify(client_credentials_manager=self.client_credentials_manager)
        self.mm = Musixmatch(self.musixmatch_key)

    def get_top_track(self, artist, track):
        artist_name_formatted = artist.replace(' ', '+')
        track_formatted = track.replace(' ', '+')
        url = 'https://api.spotify.com/v1/search?q=artist:{}&track:{}&type=track'.format(artist_name_formatted, track_formatted)

        r = requests.get(url, headers=self.sp._auth_headers())

        if not r.status_code == 200:
            logger.error("No response")
            return

        tracks = json.loads(r.content)['tracks']
        if tracks['total'] > 1:
            top_track = tracks['items'][0]
            return top_track
        else:
            logger.warn("No tracks found")
            return

    def get_audio_track_features(self, track_id):
        audio_features_dict = {}
        # These are the keys that are in the in the features
        basic_audio_features = self.sp.audio_features([track_id])[0]
        for k in ['energy', 'liveness', 'speechiness', 'acousticness',
                  'instrumentalness', 'valence', 'danceability']:
            audio_features_dict[k] = basic_audio_features[k]

        analysis_features = self._get_analysis_features(track_id)

        audio_features_dict.update(analysis_features)
        return audio_features_dict

    def _get_analysis_features(self, track_id):
        analysis = self.sp.audio_analysis(track_id)

        analysis_features = {}
        # These are the keys that are in the analysis
        for k in ['key', 'tempo', 'time_signature', 'duration', 'loudness',
                  'mode', 'time_signature_confidence', 'tempo_confidence',
                  'key_confidence', 'mode_confidence']:
            analysis_features[k] = analysis['track'][k]
        return analysis_features

    def get_lyrics(self, artist, track):
        resp = self.mm.track_search(q_track=track, q_artist=artist, page_size=1, page=1, s_track_rating='desc')
        print(json.dumps(resp, indent=4, sort_keys=True))
        top_track = resp["message"]["body"]["track_list"][0]["track"]
        track_id = top_track["track_id"]
        lyrics_resp = self.mm.track_lyrics_get(track_id)
        lyrics = lyrics_resp["message"]["body"]["lyrics"]["lyrics_body"]
        print(lyrics)
        return lyrics