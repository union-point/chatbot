from flask import Flask, render_template, request, jsonify
from assisent import get_Chat_response
app = Flask(__name__)


def is_valid_URL(url):
    try:
        from urllib.parse import urlparse
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False

@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    input = request.form["msg"]
    if is_valid_URL(input):
        return get_Chat_response(input)
    return "Please enter a valid URL"


if __name__ == '__main__':
    app.run()
