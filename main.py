from flask import Flask, request, send_file, jsonify
from flask_cors import CORS

import gc
import os
import torch
import base64
from io import BytesIO
import torch.nn.functional as F
import torchvision as tv
from objectRemoval_engine import SimpleLama
from backgroundremover import utilities
from backgroundremover.bg import remove
from projectUtils import *

app = Flask(__name__)
app.config['SECRET_KEY'] = 'vnkdjnfjknfl1232#'
CORS(app)
app.app_context().push()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
simple_lama = SimpleLama(device=device)
# -------------------------------------------- ------------------ -----------------------------------------
# -------------------------------------------- index ------------------------------------------------------
# -------------------------------------------- ------------------ -----------------------------------------
@app.route('/')
def hello_world():
    return "hello , world"


# -------------------------------------------- ------------------ -----------------------------------------
# -------------------------------------------- BackGround Removal -----------------------------------------
# -------------------------------------------- ------------------ -----------------------------------------

model_choices = ["u2net", "u2net_human_seg", "u2netp"]
bgr_model = "u2net"
alpha_matting = False
alpha_matting_foreground_threshold = 24 # 240# The trimap foreground threshold.
alpha_matting_background_threshold = 10 # The trimap background threshold.
alpha_matting_erode_size = 10# Size of element used for the erosion.
alpha_matting_base_size = 1000 # The image base size.
workernodes = 8#1 # Number of parallel workers
gpubatchsize = 260#2 # GPU batchsize
framerate = -1 # override the frame rate
framelimit = -1 # Limit the number of frames to process for quick testing.
mattekey = False # Output the Matte key file , type=lambda x: bool(strtobool(x)),
transparentvideo = False # Output transparent video format mov
transparentvideoovervideo = False # Overlay transparent video over another video
transparentvideooverimage = False # Overlay transparent video over another video
transparentgif = False # Make transparent gif from video
transparentgifwithbackground = False # Make transparent background overlay a background image


@app.route('/removebg', methods=['POST'])
def rgb():

    print("/REMOVEBG new request coming")
    data = request.get_json()
    base64Image= data["image"]
    new_image = remove(
                    base64Image,
                    model_name=bgr_model,
                    alpha_matting=alpha_matting,
                    alpha_matting_foreground_threshold=alpha_matting_foreground_threshold,
                    alpha_matting_background_threshold=alpha_matting_background_threshold,
                    alpha_matting_erode_structure_size=alpha_matting_erode_size,
                    alpha_matting_base_size=alpha_matting_base_size,
                )
    
    
    print(f" BG-image base 64 = {new_image[20:]}")
    return jsonify({"bg_image":new_image})


#------------------------------------ Object removal ---------------------------------------------

"""
[{'startX': 1033.9316, 'startY': 1210.915, 'endX': 1033.9316, 'endY': 1267.9419, 'strokeWidth': 20}, {'startX': 1033.9316, 'startY': 1267.9419, 'endX': 1033.9316, 'endY': 1272.8931, 'strokeWidth': 20}, {'startX': 1033.9316, 'startY': 1272.8931, 'endX': 1033.9316, 'endY': 1288.873, 'strokeWidth': 20}, {'startX': 1033.9316, 'startY': 1288.873, 'endX': 1033.9316, 'endY': 1293.9355, 'strokeWidth': 20}]
"""

@app.route('/removeobj', methods=['POST'])
def object_removal():
    print("/REMOVE_OBJ new request coming")
    data = request.get_json()
    base64Image= data["image"]
    base64mask= data["mask"]
    size = data["size"]

    print(f"Size found : {size}")

    cv_img = base64toopencv(base64Image)
    cv_mask = base64toopencv(base64mask)
    cv2.imwrite("test.png",cv_img)
    h, w, c = cv_img.shape
    cv_mask = cv2.resize(cv_mask, (w, h))
    cv2.imwrite("test_mask.png",cv_mask)


    cv_mask = cv2.cvtColor(cv_mask, cv2.COLOR_BGR2RGB) 
    cv_mask = Image.fromarray(cv_mask).convert('L') 

    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB) 
    cv_img = Image.fromarray(cv_img) 

    result = simple_lama(cv_img, cv_mask)
    new_image = result
    bio = io.BytesIO()
    new_image.save(bio, "PNG")
    bio.seek(0)
    im_b64 = base64.b64encode(bio.getvalue()).decode()

    return jsonify({"bg_image":im_b64}) # end of function end point removeobj 













if __name__ == '__main__':
    app.run(port=8080, debug=True, use_reloader=False)
    #app.run(host='0.0.0.0', port=port)
