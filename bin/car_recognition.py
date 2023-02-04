import cv2
import torch
import pandas as pd
from mss import mss
import easyocr

reader = easyocr.Reader(['en'])
model = torch.hub.load("ultralytics/yolov5", 'custom', path="/Users/admin/Desktop/tesseract/venv/lib/5n6_mentor.pt",
                       trust_repo=True)
sct = mss()
vid = cv2.VideoCapture(0)

def check_dataframe(result_data):
    if result_data[result_data["name"] == 'car'].size > 0:
        is_license_plate_present(result_data)


def is_license_plate_present(result_data):
    if result_data[result_data["name"] == 'lisence-plate'].size > 0:
        is_license_plate_on_car(result_data)


def is_license_plate_on_car(result_data):
    cars = result_data[result_data["name"] == 'car']
    licenses = result_data[result_data["name"] == 'lisence-plate']
    for i in cars.index:
        xmin_car = cars["xmin"][i]
        ymin_car = cars["ymin"][i]
        xmax_car = cars["xmax"][i]
        ymax_car = cars["ymax"][i]
        for j in licenses.index:
            xmin_license = licenses["xmin"][j]
            ymin_license = licenses["ymin"][j]
            xmax_license = licenses["xmax"][j]
            ymax_license = licenses["ymax"][j]
            if xmin_car < xmin_license and ymin_car < ymin_license and xmax_car > xmax_license and ymax_car > ymax_license:
                screenshot = frame[int(ymin_license):int(ymax_license), int(xmin_license):int(xmax_license)]
                cv2.imwrite('screenshot.jpg', screenshot)
                text = reader.readtext(screenshot)
                # print("Detected text on the license plate: ", text)
                if text:
                    textR = text[0][1]
                    print(textR)
                else:
                    print('')
                break

while True:
    ret, frame = vid.read()
    result = model(frame, size=640)
    check_dataframe(pd.DataFrame(result.pandas().xyxy[0]))
