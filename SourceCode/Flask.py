from flask import Flask, request, jsonify, render_template
from main1 import process_frame
import threading


app = Flask(__name__)

def background_task():
    pass

@app.before_first_request
def start_background_task():
    threading.Thread(target=background_task, daemon=True).start()

@app.route("/")
def index():
    return render_template("buttons.html")


@app.route("/process_frame", methods=["POST"])
def process():
    data = request.json
    image = data["image"]

    result = process_frame(image)

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)