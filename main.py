from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import os
from backgroundremover import utilities
from backgroundremover.bg import remove

app = Flask(__name__)
app.config['SECRET_KEY'] = 'vnkdjnfjknfl1232#'
CORS(app)
app.app_context().push()
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


if __name__ == '__main__':
    app.run(port=8080, debug=True, use_reloader=False)
    #app.run(host='0.0.0.0', port=port)
