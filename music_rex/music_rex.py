from __future__ import print_function    # (at top of module)
from spotipy.oauth2 import SpotifyClientCredentials
import lyricsgenius as genius
import json
import spotipy
import requests
import os

CLIENT_ID='37b5edf45b91479d9d01614b6f8de4b2'
CLIENT_SECRET='0ebe2de09ed84139ba1a6fb237076b82'

GENIUS_SECRET='1FUgpdlEn3dW7-pe-2XIk8GEIUeFZFLhCCL-2cYBDQeHSojkn80ouMXLktndCjrbBIvmBcTYD1YXgS_3v4eK6g'
GENIUS_ID='2PPh8E5cmYzeGo5urNOcJ1rgXUaLm8robEoME4cLb_R7UGjaOK4XidvZvW4cIplK'
GENIUS_ACCESS_TOKEN ='orm_c1zHIuqrvH3fuaCKocSMwlF9N3VgQpQLtT95RBpk-Dyzp-ZeYHXlYBK6FMl7'

class MusicRex(object):
    def __init__(self, client_id=None, client_secret=None):
        # if not client_id:
        #     client_id = os.getenv('CLIENT_ID')
        # if not client_secret:
        #     client_secret = os.getenv('CLIENT_SECRET')

        # if not client_id or not client_secret:
        #     raise Exception("No CLIENT_ID or CLIENT_SECRET in environment.")

        self.client_id = CLIENT_ID
        self.client_secret = CLIENT_SECRET
        self.genius_access_token = GENIUS_ACCESS_TOKEN
        self.client_credentials_manager = SpotifyClientCredentials(self.client_id, self.client_secret)
        self.sp = spotipy.Spotify(client_credentials_manager=self.client_credentials_manager)
        self.genius = genius.Genius(self.genius_access_token)

    def get_top_track(self, artist, track):
        artist_name_formatted = artist.replace(' ', '+')
        track_formatted = track.replace(' ', '+')
        url = 'https://api.spotify.com/v1/search?q=artist:{}&track:{}&type=track'.format(artist_name_formatted, track_formatted)

        r = requests.get(url, headers=self.sp._auth_headers())

        if not r.status_code == 200:
            print("No response")
            return

        tracks = r.json()['tracks']
        if tracks['total'] > 1:
            top_track = tracks['items'][0]
            return top_track
        else:
            print("No tracks found")
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
        print(artist)
        print(track)
        song = self.genius.search_song(track, artist_name=artist)
        return song.lyrics

    def get_lyrics_mm(self, artist, track):
        resp = self.mm.track_search(q_track=track, q_artist=artist, page_size=1, page=1, s_track_rating='desc')
        print(json.dumps(resp, indent=4, sort_keys=True))
        top_track = resp["message"]["body"]["track_list"][0]["track"]
        track_id = top_track["track_id"]
        lyrics_resp = self.mm.track_lyrics_get(track_id)
        lyrics = lyrics_resp["message"]["body"]["lyrics"]["lyrics_body"]
        print(lyrics)
        return lyrics