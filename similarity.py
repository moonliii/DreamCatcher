import copy
import cv2
import sys
import math

image_path1 = ""
image_path2 = ""


def preprocess_image(image_path):
    image = cv2.imread(image_path)

    # error handling for loading images
    if image is None:
        print('Loading image failed')
        sys.exit(-1)

    # Gaussian blur
    blur_image = cv2.GaussianBlur(image, (3, 3), 0)

    # bgr to gray
    gray_image = cv2.cvtColor(blur_image, cv2.COLOR_BGR2GRAY)

    # threshold for converting grayscale to binary
    thresh_gray = 200

    # gray to binary
    retval, binary_image = cv2.threshold(gray_image, thresh_gray, 255, cv2.THRESH_BINARY)

    # white background to black, reverse the bits
    binary_image = cv2.bitwise_not(binary_image)

    return binary_image


def get_hu_moments(image_path):
    binary_image = preprocess_image(image_path)

    # get HuMoments
    moments = cv2.moments(binary_image)
    hu_moments = cv2.HuMoments(moments)

    return hu_moments


def process_hu(hu_moments, eps):
    # deep copy
    temp_hu_moments = copy.deepcopy(hu_moments)
    for i in range(7):
        abs_hu = abs(temp_hu_moments[i])
        if abs_hu != 0.0 and abs_hu > eps:
            temp_hu_moments[i] = -1 * math.copysign(1.0, temp_hu_moments[i]) * math.log(abs_hu, 10)
        else:
            temp_hu_moments[i] = 0.0
    return temp_hu_moments


def process_result(result, distance, flag):
    upper_threshold = 0.95
    distance_threshold = 0.4
    # process result
    if result > 0:
        # flag == True means similarity is low
        if flag:
            result /= 5
        # Add distance factor to unusually high similarity (fine-tune)
        elif result > upper_threshold and distance < distance_threshold:
            result -= distance
    else:
        # result < 0 means similarity is low
        result = (result + 1) / 100
    return result


# cosine similarity
def get_cos_similarity(image1, image2):
    sim_eps = 1e-9
    dis_eps = 1e-5

    result = 0.0
    norm1 = 0.0
    norm2 = 0.0
    flag = False
    distance = 0.0

    hu_moments1 = get_hu_moments(image1)
    hu_moments2 = get_hu_moments(image2)

    sim_hu_moments1 = process_hu(hu_moments1, sim_eps)
    sim_hu_moments2 = process_hu(hu_moments2, sim_eps)

    dis_hu_moments1 = process_hu(hu_moments1, dis_eps)
    dis_hu_moments2 = process_hu(hu_moments2, dis_eps)

    for i in range(7):
        result += sim_hu_moments1[i] * sim_hu_moments2[i]
        norm1 += sim_hu_moments1[i]**2
        norm2 += sim_hu_moments2[i]**2
        # On the same dimension,
        # the fact that one of the values equals to 0.0 while the other is not
        # means the similarity should be low
        if (sim_hu_moments1[i] == 0.0 and sim_hu_moments2[i] != 0.0) or\
            (sim_hu_moments2[i] == 0.0 and sim_hu_moments1[i] != 0.0):
            flag = True
        # CV_CONTOURS_MATCH_I2 distance
        if dis_hu_moments1[i] != 0.0 and dis_hu_moments2[i] != 0.0:
            distance += abs(dis_hu_moments2[i] - dis_hu_moments1[i])

    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    result = result/((norm1 * norm2)**0.5)
    result = result[0]

    distance = distance[0]

    result = process_result(result, distance, flag)

    return result


print(get_cos_similarity(image_path1, image_path2))
