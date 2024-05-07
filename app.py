from flask import Flask, render_template, request, jsonify
from assisent import get_Chat_response
app = Flask(__name__)

@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    input = request.form["msg"]
    return get_Chat_response(input)


if __name__ == '__main__':
    app.run()
