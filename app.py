import numpy as np
from flask import Flask, redirect, request, render_template
import base64
import imutils
from cv2 import cv2
import os

app = Flask(__name__)

styleImageList = {
    1 : "candy.t7",
    2 : "composition_vii.t7",
    3 : "feathers.t7",
    4 : "la_muse.t7",
    5 : "mosaic.t7",
    6 : "starry_night.t7",
    7 : "the_scream.t7",
    8 : "udnie.t7",
    9 : "the_wave.t7"
}

def InvokeNeuralStyleTransfer(inp_img, styleNumber):
    net = cv2.dnn.readNetFromTorch('./models/' + styleImageList[int(styleNumber)])
    image = imutils.resize(inp_img, width=700)
    (h, w) = image.shape[:2]

    blob = cv2.dnn.blobFromImage(image, 1.0, (w, h),
                                (103.939, 116.779, 123.680), swapRB=False, crop=False)
    
    net.setInput(blob)
    output = net.forward()
    output = output.reshape((3, output.shape[2], output.shape[3]))
    output[0] += 103.939
    output[1] += 116.779
    output[2] += 123.680
    output = output.transpose(1, 2, 0)

    return output


ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=['GET', 'POST'])
def base():
    if request.method == 'GET':
        return render_template("base.html")
    else:
        if 'input_image' not in request.files:
            print("No file part")
            return redirect(request.url)

        file = request.files['input_image']

        if file.filename == '':
            print('No selected file')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            styleNumber = request.form.get('styleNumber')
            inp_img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
            output = InvokeNeuralStyleTransfer(inp_img, styleNumber)
            _, buffer = cv2.imencode('.jpg', output)
            OutputBase64String = base64.b64encode(buffer).decode('utf-8')
            return render_template("base.html", img=OutputBase64String)


if __name__ == "__main__":
    app.secret_key = 'qwertyuiop1234567890'
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
    
