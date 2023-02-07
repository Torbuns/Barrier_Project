import re
from paddleocr import PaddleOCR, draw_ocr
import cv2
import numpy as np
import pandas as pd

COMMON_LICENSE_PATTERN = r"^([A-Z]){2}(\d){4}([A-Z]){2}$"
EXCLUSIVE_LICENSE_PATTERN = r"^[A-Z]+$"
POSSIBILITIES = {'O': '0', 'B': '8', 'I': '1'}
EN_UKR_LETTERS = ['A', 'B', 'C', 'E', 'H', 'I', 'K', 'M', 'O', 'P', 'T', 'X']
IF_NOT_UKR_POSSIBILITIES = {'D': 'O', 'F': 'E', 'G': '-', 'J': '-', 'L': '-', 'N': '-', 'Q': 'O',
                            'R': '-', 'S': '5', 'U': '-', 'V': '-', 'W': '-', 'Y': '-', 'Z': '2'}


def img_enlarging(inner_img: np.ndarray):
    if inner_img.shape[1] < 150:
        inner_img = cv2.resize(inner_img, None, fx=1.25, fy=1.25)
        inner_img = img_enlarging(inner_img)
    return inner_img


def string_cleaner(x):
    x = x.upper()
    for i in x:
        if i.isnumeric() or i.isalpha():
            continue
        else:
            x = x.replace(i, "")
    return x


def check_and_change_license(res_str: str) -> str:
    if re.fullmatch(COMMON_LICENSE_PATTERN, res_str):
        pass
    elif re.fullmatch(EXCLUSIVE_LICENSE_PATTERN, res_str):
        pass
    else:
        for letter_id in range(len(res_str)):
            if 0 <= letter_id <= 1 or (len(res_str) - 2) <= letter_id <= (len(res_str) - 1):
                if not res_str[letter_id].isalpha():
                    res_str = res_str.replace(res_str[letter_id], [val for val in POSSIBILITIES.keys() if POSSIBILITIES[
                        val] == res_str[letter_id]][0], 1)
                elif res_str[letter_id] not in EN_UKR_LETTERS:
                    res_str = res_str.replace(res_str[letter_id], IF_NOT_UKR_POSSIBILITIES[res_str[letter_id]], 1)
            elif 2 <= letter_id <= 5:
                if res_str[letter_id].isalpha():
                    res_str = res_str.replace(res_str[letter_id], POSSIBILITIES[res_str[letter_id]] if res_str[
                                                                                                           letter_id] in POSSIBILITIES.keys() else
                    IF_NOT_UKR_POSSIBILITIES[res_str[letter_id]] if
                    IF_NOT_UKR_POSSIBILITIES[res_str[letter_id]] != '-' else '', 1)
    return res_str


def try_to_get_res(resultt: list, reader: PaddleOCR, img_path: str):
    try:
        ress = resultt[0][0][1][0]
        print('ress: ', ress)
    except IndexError:
        img = cv2.imread(img_path)
        img_75 = img_enlarging(img)
        resultt = reader.ocr(img_75, cls=True)
        ress = try_to_get_res(resultt, reader, img_path)
    return ress


def get_valid_licenses():
    data = pd.read_csv('valid_licenses.txt', sep=';').dropna(axis=1)
    return [i for i in data['license_number']]


def open_or_not(license_num, reader, img_path):
    license_num = try_to_get_res(license_num, reader, img_path)
    license_num = string_cleaner(license_num)
    license_num = check_and_change_license(license_num)
    valid_licenses = get_valid_licenses()
    if license_num in valid_licenses:
        print(f"Open the gate for {license_num}!")
    else:
        print(f"You, {license_num}, shall not pass!")
