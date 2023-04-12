"""
Simple app to upload an image via a flask_deployment form
and view the inference results on the image in the browser.
"""
import argparse
import io
import os

import cv2
import numpy as np
from PIL import Image
import datetime

import torch
from flask import Flask, render_template, request, redirect

app = Flask(__name__)
@app.route("/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if not file:
            return
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        results = model(img, size=640)
        # Save the image to a file in the static directory
        processed_image_path = "C:/Users/Admin/Desktop/New_folder/OJT_Project/Mô_hình_Demo/Demo/static/detected_card"
        results.save(processed_image_path)
        results = results.pandas().xyxy[0]
        results.sort_values(by=['xmin'], inplace=True)
        cardnumber = ""
        for i in results['name']:
            cardnumber += str(i)
        # Render the home.html template with the card number and processed image
        return render_template("home.html", cardnumber=cardnumber, processed_image='detected_card/image0.jpg')
    return render_template("index.html")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask app exposing yolov7 models")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()
    model = torch.hub.load('C:/Users/Admin/Desktop/New_folder/OJT_Project/Mô_hình_Demo/Model_YOLOv7', 'custom','C:/Users/Admin/Desktop/New_folder/OJT_Project/Mô_hình_Demo/Model_YOLOv7/runs/train/exp/weights/best.pt',force_reload =True,source='local')  # force_reload = recache latest code
    model.eval()
    # app.run(host="0.0.0.0", port=args.port)  # debug=True causes Restarting with stat
    app.run()
