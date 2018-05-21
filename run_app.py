from flask import Flask, request, render_template
import json
import pickle as pkl

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/submit', methods=["POST"])
def submit():
    song_title = request.form.get('song_title')
    artist = request.form.get('artist')
    import ipdb; ipdb.set_trace()
    return '', (httplib.NO_CONTENT if success else httplib.BAD_REQUEST)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True, passthrough_errors=False)
