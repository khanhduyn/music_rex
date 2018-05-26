from flask import Flask, request, render_template, jsonify
import json
import pickle as pkl
import httplib

from music_rex.music_rex import MusicRex

app = Flask(__name__, static_url_path='/static')

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/submit', methods=["POST"])
def submit():
    song_title = request.form.get('songTitle')
    artist = request.form.get('artist')
    print(song_title)
    print(artist)

    mr = MusicRex()
    track = mr.get_top_track(artist, song_title)
    features = mr.get_audio_track_features(track['id'])
    print(json.dumps(features, indent=4, sort_keys=True))
    success = (( features is not None) and len(features) > 0)

    return jsonify(features), (httplib.OK if success else httplib.BAD_REQUEST)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True, passthrough_errors=False)
