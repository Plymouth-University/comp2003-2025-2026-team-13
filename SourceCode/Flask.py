from flask import Flask, request, jsonify, render_template
from main1 import process_frame
import threading
import base64
import numpy as np
import cv2

app = Flask(__name__)

def background_task():
    pass

@app.before_first_request
def start_background_task():
    threading.Thread(target=background_task, daemon=True).start()

@app.route("/")
def index():
    return render_template("buttons.html")

@app.route("/settings")
def settings():
    return render_template("Settings.html")



@app.route("/process_frame", methods=["POST"])
def process():
    data = request.json
    image = data["image"]

    image = image.split(",")[1]
    img_bytes = base64.b64decode(image)
    np_arr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    result = process_frame(frame)
    print(result)
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)