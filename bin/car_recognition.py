import cv2
import torch
from mss import mss
import pandas as pd

model = torch.hub.load("ultralytics/yolov5", 'custom', path="../resource/5n6_mentor.pt")
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
                print("*************************************************************Shazam open up*******************************************************")
                break


while True:
    ret, frame = vid.read()
    result = model(frame, size=640)
    check_dataframe(pd.DataFrame(result.pandas().xyxy[0]))