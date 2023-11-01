import cv2
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import Metadata
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
import cv2
import yaml
import numpy
from aioflask import Flask, jsonify, request
import time
from PIL import Image
import asyncio
import requests
import base64

app = Flask(__name__)

def getTime():
    # Get the current timestamp in seconds
    current_timestamp = time.time()

    # Extract the current minute, second, and millisecond
    current_minute = int((current_timestamp // 60) % 60)
    current_second = int(current_timestamp % 60)
    current_millisecond = int((current_timestamp % 1) * 1000)

    # Format the time as string (mm:ss.sss)
    currentTime = f"{current_minute:02}:{current_second:02}.{current_millisecond:03}"
    return currentTime

def image_to_base64(image):

    # Convert the image to RGB format (OpenCV uses BGR by default)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Encode the image to Base64
    retval, buffer = cv2.imencode('.jpg', image_rgb)
    base64_image = base64.b64encode(buffer).decode('utf-8')
    return base64_image
def calculateVolume(x, y, x2, y2, dpi, pixelcount):
    
    # Verileri tanımlayın
    volx = numpy.array([148020, 174503, 374399, 477370, 662724])
    voly = numpy.array([250, 330, 1000, 1500, 2500])

    # Polinom derecesini belirleyin
    polynomial_degree = 2

    # Polinom katsayılarını hesaplayın
    coefficients = numpy.polyfit(volx, voly, polynomial_degree)

    # Polinom denklemi oluşturun
    poly_eq = numpy.poly1d(coefficients)

    # Tahmin yapmak istediğiniz piksel sayısını belirtin
    new_x = pixelcount

    # Tahmin yapın
    predicted_y = poly_eq(new_x)
    
    volume = round(predicted_y)

    # Calculate pixels
    x = x2 - x
    y = y2 - y

    # Convert pixels to cm
    x = x * 2.54 / dpi
    y = y * 2.54 / dpi

    # Calculate diameter
    diameter = y / 2
    diameter = diameter * diameter
    
    # Dimensions
    if not ( x<= 0 and y<=0):
        x = round(x * 10,1)
        y = round(y * 10,1)
        x = float(x) * 1.554
        x = float(x) - 77
        y = float(y) * 1.134
        y = float(y) - 2.7
    else:
        x=0
        y=0
    # Calculate mask dimensions in millimeters
    mask_dimensions_mm = {"x": round(x * 1), "y": round(y * 1)}

    return volume, mask_dimensions_mm
# Find metadata file and load
with open("./model/metadata.yaml", "r") as f:
    metadata_dict = yaml.load(f, Loader=yaml.FullLoader)

# Setup metadata 
metadata = Metadata()
metadata.set(thing_classes=metadata_dict["thing_classes"])
metadata.set(thing_colors=metadata_dict["thing_colors"])
metadata.set(thing_dataset_id_to_contiguous_id=metadata_dict["thing_dataset_id_to_contiguous_id"])

# Load a configuration file
cfg = get_cfg()
cfg.merge_from_file("./model/config.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.10  # set threshold for this model
cfg.MODEL.DEVICE = "cuda"

# Replace with your .pth file path
model_path = "./model/model_final.pth"
cfg.MODEL.WEIGHTS = model_path

# Create a predictor
predictor = DefaultPredictor(cfg)

loop = asyncio.get_event_loop()

from concurrent.futures import ThreadPoolExecutor
executor = ThreadPoolExecutor()

@app.route("/", methods=["GET"])
async def index():
    return jsonify(
        {
            "ServiceName": "AcoRecycle Detectron2 API",
            "version": "0.1.0",
            "Environment": "Development",
        }
    )


@app.route("/", methods=["POST"])
async def main():
    image_files = request.files.getlist("image_files")
    
    response = {
        "results": None,
        "message": None,
        "timestamp": int(time.time()*1000),
    }

    if (predictor is None):
        response["message"] = "Model not found"

    else:
        filenames = []
        model_inputs = []
        for image_file in image_files:
            filenames.append(image_file.filename)
            image = Image.open(image_file)
            model_inputs.append(image)
        image = numpy.asarray(image)

        # Use the predictor to make a prediction
        print("Predict started at ",getTime())
        outputs = predictor(image)
        print("Predict finished at ",getTime())
        results_list = []
        instances = outputs["instances"]

        for i in range(len(instances)):
            prediction_dict = dict()
            prediction_dict["filename"] = filenames[0]

            pred_boxes = instances.get("pred_boxes")
            if pred_boxes is None and len(pred_boxes.tensor) == 0:
                print("No object detected!")
                response["message"] = "No object detected!"
                return jsonify(response)
            elif i < len(instances):
                pred = instances[i]
                y1, x1, y2, x2 = pred.pred_boxes.tensor[0].tolist()
                y1, x1, y2, x2 = int(y1), int(x1), int(y2), int(x2)  # convert to integers
                print(y1, ' ', x1,' ',y2,' ',x2)
                mask = pred.pred_masks[0][y1:y2, x1:x2].detach().cpu().numpy()
                threshold = 0.5
                mask = mask.astype(int)
                num_pixels = (pred.pred_masks[0].detach().cpu().numpy() == True).sum()
                print(num_pixels)

                # Toplam hacmi hesapla
                volume, mask_dimensions_mm = calculateVolume(y1, x1, y2, x2, 135,num_pixels)
                print("Volume -> ", volume," ", num_pixels)
                print("Mask Dimensions (mm) -> ", mask_dimensions_mm)
                if(volume <= 0):
                    volume = 0
                # Retrieve the prediction score and threshold
                scores = instances[i].scores.item()
                threshold = cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST

                # Retrieve the class tag (name)
                class_label = instances[i].pred_classes.item()
                class_tag = metadata.thing_classes[class_label]
                prediction_dict["predictions"] = {
                    "class_tag": class_tag,
                    "score": round(scores,2),
                    "threshold": threshold,
                    "mask_dimensions": mask_dimensions_mm,#{"x": round(x_dim,2), "y": round(y_dim,2)},
                    "volume": round(volume),
                    "num_pixels": int(num_pixels)
                    #"prediction_image_base64": prediction_base64
                }
                results_list.append(prediction_dict)
                
            else:
                print("Index out of range for instances!")
                
                response["message"] = "Index out of range for instances!"
                return jsonify(response)
       
       
        pred_boxes = instances.get("pred_boxes")
        if pred_boxes is not None and len(pred_boxes.tensor) > 0:
            loop = asyncio.get_event_loop()
            loop.create_task(save_processed_image(instances[0], image, filenames[0]))
        response["results"] = results_list
        response["message"] = "OK"


    return jsonify(response)

def draw_circle(image):
     # Define the coordinates (x, y) where you want to draw the point
    x, y = 242, 764
    # y1, x1, y2, x2
    # Define the color of the point in BGR format (blue, green, red)
    point_color = (255, 0, 0)  # Green color

    # Define the radius of the point (you can adjust this according to your needs)
    point_radius = 50

    # Draw the point on the image
    output_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    output_image = cv2.circle(output_image, (x, y), point_radius, point_color, -1)  # -1 for a filled circle
    return output_image

def draw_and_save_image(image, instance, filename):
        
    image_with_bb = Visualizer(image[:, :, ::-1], metadata=metadata, scale=1)
    image_with_bb = image_with_bb.draw_instance_predictions(instance.to("cpu"))

    #Visualizer nesnesinden çıktı fotoğrafı alın
    output_image = image_with_bb.get_image()[:, :, ::-1]  # BGR to RGB dönüşümü için ::-1 kullanılır.
    #output_image = draw_circle(output_image)
    # Convert the original image to Base64
    base64_image = image_to_base64(output_image)

    # Prepare the JSON payload
    payload = {
        "b64": base64_image,
        "filename": filename,
        "editedfilename": f"{time.time():.0f}_{filename}"
    }

    

async def save_processed_image(instance, image, filename):
    await asyncio.get_event_loop().run_in_executor(executor, draw_and_save_image, image, instance, filename)



if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8090)




