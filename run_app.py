from flask import Flask, request, render_template, jsonify
from flask_cors import CORS

import json
import pickle as pkl
import http.client

from music_rex.music_rex import MusicRex
from music_rex.machine_learning import get_similar_songs

app = Flask(__name__, static_url_path='/static')
CORS(app)


mr = MusicRex()

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/submit', methods=["POST"])
def submit():
    return jsonify("ok"), (http.client.OK)
    args = request.get_json(force=True)
    song_title = args["song_title"]
    artist = args["artist"]
    track = mr.get_top_track(artist, song_title)
    features = mr.get_audio_track_features(track['id'])
    success = (( features is not None) and len(features) > 0)

    return jsonify(features), (http.client.OK if success else http.client.BAD_REQUEST)

@app.route('/lyrics', methods=["GET"])
def lyrics():
    return jsonify("ok"), (http.client.OK)
    song_title = request.args.get('song_title')
    artist = request.args.get('artist')
    lyrics = mr.get_lyrics(artist, song_title)
    success = (( lyrics is not None) and len(lyrics) > 0)
    return jsonify(lyrics), (http.client.OK if success else http.client.BAD_REQUEST)

@app.route('/playlist', methods=["POST"])
def playlist():
    # print("Calling playlist")
    # args = request.get_json(force=True)
    # print(args)
    # lyrics = args["lyrics"]
    # features = args["features"]
    # recommendations = get_similar_songs(features, lyrics)
    recommendations = get_similar_songs(1, 2)
    return jsonify(recommendations), (http.client.OK)

@app.route('/create_playlist', methods=["POST"])
def create_playlist():
    args = request.get_json(force=True)
    print(args)
    playlist = args['playlist']
    tracks = mr.get_playlist(playlist)
    return jsonify(tracks), (http.client.OK)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True, passthrough_errors=False)
