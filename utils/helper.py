import itertools
import logging
import math
import re
import cv2
import numpy as np


def softmax(x):
    e_x = np.exp(x)
    return e_x / np.sum(e_x)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))





def non_max_suppression_fast(labels, iou_threshold):
    # if there are no boxes, return an empty list
    if len(labels) == 0:
        return []

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    # print labels

    # w_2 = labels[:, 2] / 2.
    # h_2 = labels[:, 3] / 2.

    x1 = labels[:, 0]
    y1 = labels[:, 1]
    x2 = labels[:, 0] + labels[:, 2]
    y2 = labels[:, 1] + labels[:, 3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(
            idxs, np.concatenate(([last], np.where(overlap > iou_threshold)[0]))
        )

    return labels[pick]


def rotate(image, angle, shear=None):
    type_border = cv2.BORDER_CONSTANT

    h, w = image.shape[:2]

    angle = angle  # Angle in degrees.
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)

    if shear is not None:
        translation = -10 * (shear)

        # Application of the affine transform.
        translat_center_x = -(shear * w) / 2
        translat_center_y = -(shear * h) / 2

        M = M + np.float64(
            [
                [0, shear, translation + translat_center_x],
                [shear, 0, translation + translat_center_y],
            ]
        )

    final_image = cv2.warpAffine(
        image, M, (w, h), borderMode=type_border, flags=cv2.INTER_LANCZOS4
    )

    return final_image


def image_find_max_contours(image):
    _, gray_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    #image = cv2.medianBlur(image, 3)
    image = cv2.bilateralFilter(image, 11, 17, 17)
    canny_image = cv2.Canny(image, 75, 150)
    out = cv2.findContours(canny_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    countours = out[0]

    if len(countours) > 0:
        countours = sorted(countours, key=cv2.contourArea, reverse=True)[0]
        return countours
    else:
        return None


last_angle = None
c = 0



def prepare_plate(license_plate, angle_range=15):
    global last_angle, c
    gray_image = cv2.cvtColor(license_plate, cv2.COLOR_BGR2GRAY)
    global last_angle

    min_area = image_find_max_contours(gray_image)
    min_angle = 0
    if last_angle is None or c == 2:
        for angle in range(-angle_range, angle_range + 1, 5):
            area = image_find_max_contours(rotate(gray_image, angle))

            if (
                area is not None
                and min_area is None
                or area is not None
                and cv2.contourArea(area) < cv2.contourArea(min_area)
            ):
                min_area = area
                min_angle = angle

        if last_angle is None:
            last_angle = min_angle
        else:
            last_angle = (last_angle + min_angle) / 2.0

    license_plate = rotate(license_plate, last_angle)
    return license_plate


def softmax(x):
    """Compute softmax values for each sets of scores in x."""

    # print np.sum(np.exp(x), axis=-1)
    return np.exp(x) / np.expand_dims(np.sum(np.exp(x), axis=-1), -1)


codes_2 = []
reg_codes = range(1, 100, 1)
for i in reg_codes:
    if i < 10:
        si = "0" + str(i)
    else:
        si = str(i)

    codes_2.append(si)

codes_3 = [
    "101",
    "102",
    "103",
    "109",
    "111",
    "113",
    "116",
    "118",
    "121",
    "123",
    "124",
    "125",
    "126",
    "134",
    "136",
    "138",
    "142",
    "150",
    "152",
    "154",
    "159",
    "161",
    "163",
    "164",
    "173",
    "174",
    "176",
    "177",
    "178",
    "186",
    "190",
    "196",
    "197",
    "199",
    "716",
    "750",
    "763",
    "777",
    "799",
]


def decode_batch(out, letters, thresh_len=6):
    ret = []
    for j in range(out.shape[0]):
        out_softmax = softmax(out[j])
        to_keep = np.where(np.max(out_softmax[2:], 1) > 0.7)
        out_best = list(np.argmax(out_softmax[2:][to_keep], 1))
        #out_best = list(np.argmax(out_softmax[2:], 1))
        out_best = [k for k, g in itertools.groupby(out_best)]
        outstr = ""
        for c in out_best:
            # print c
            if c < len(letters):
                outstr += letters[c]

        if len(outstr) > thresh_len:
            regexp = re.findall("(?P<letter>[A-Z]+)(?P<number>\d+)", outstr)
            if len(regexp) == 2:
                if len(regexp[0][1]) == 3 and len(regexp[1][0]) == 2:
                    reg_number = regexp[1][1]
                    label = regexp[0][0] + regexp[0][1] + regexp[1][0] + reg_number
                    if reg_number in codes_3 or reg_number in codes_2:
                        ret.append(label)
            else:
                if len(regexp) == 1:
                    ret.append(outstr)

    return ret


def bbox_crop_plate(bboxes, image, plate_max_rotate_angle=10):
    plates = []
    width = image.shape[1]
    height = image.shape[0]
    for bbox in bboxes:

        values = str(bbox).strip('[]').split()
        numbers = [float(value) for value in values]
        x = int(numbers[0] * width)
        y = int(numbers[1] * height)
        w = int(numbers[2] * width)
        h = int(numbers[3] * height)

        x_min = int(x)
        y_min = int(y)
        x_max = int(x + w)
        y_max = int(y + h)
        img_crop = image[y_min:y_max, x_min:x_max]
        ph, pw = img_crop.shape[:2]
        if ph > 20 and pw > 20:
            plate_img = prepare_plate(img_crop, angle_range=plate_max_rotate_angle)
            ph, pw = plate_img.shape[:2]

            mean, std = cv2.meanStdDev(plate_img)
            if 0.90 * image.shape[0] > pw > ph and std[0] > 10:
                plates.append(plate_img)

    return plates






