# pip install pyserial
# Opencv 차선인식
import cv2
import numpy as np
import copy, time
import math

bottom_non = 1
upper_cut = 0.1

road_width = 500  # 도로 폭 mm

yellow_lower = (15, 130, 2) # (10, 140, 10)
yellow_upper = (40, 195 , 30) # (30, 190, 40)

virtuala_center_temp = []

Trekking = 0
Trekking_temp = 0
show = 0
margin = 2000
add = 445

Dis = 200

count = 0
img_count = 192
revers = 0
swich = 0
R = 5
rotate = 40
arduino = 0

left_line_color = (255, 0, 255)
right_line_color = (255, 0, 255)

f = 1

def onChange(x):
    pass

def track(width, height):
    cv2.namedWindow('show', cv2.WINDOW_NORMAL)
    cv2.createTrackbar("X", "show", 0, width-1, onChange)
    cv2.createTrackbar("Y", "show", 0, height-1, onChange)
    pass

def color_text(img, size = 15):
    global count
    if size > 15:
        size = 15
        print(f"Size_Range_Over \n Size => {size}")
    elif size <= 0:
        size = 1
        print(f"Size_Range_Over \n Size => {size}")

    if count == 0:
        h, w = img.shape[:2]
        track(w, h)
        count += 1

    if count == 1:
        try:
            img_X = cv2.getTrackbarPos("X", "show")
            img_Y = cv2.getTrackbarPos("Y", "show")
            print(img_X, img_Y)
        except cv2.error:
            # h, w = img.shape[:2]
            track(w, h)

    if len(img.shape) != 3:
        img = cv2.merge((img,img,img))

    move = (img_X, img_Y)  # (720, 1280, 3)

    rectang_p1 = (0 + move[0], 0 + move[1])
    rectang_p2 = (size + move[0], size + move[1])

    img_hls_texts = img[rectang_p1[1]:rectang_p2[1], rectang_p1[0]:rectang_p2[0], :]

    # print(img_hls_texts.shape)

    img_h_text = np.concatenate(img_hls_texts[:, :, 0], axis=0)
    img_l_text = np.concatenate(img_hls_texts[:, :, 1], axis=0)
    img_s_text = np.concatenate(img_hls_texts[:, :, 2], axis=0)

    print(f"img_h_text \n{img_h_text}")
    print(f"img_l_text \n{img_l_text}")
    print(f"img_s_text \n{img_s_text}")
    print(f"img_h_text = max : {img_h_text[img_h_text.argmax()]} || min : {img_h_text[img_h_text.argmin()]}")
    print(f"img_l_text = max : {img_l_text[img_l_text.argmax()]} || min : {img_l_text[img_l_text.argmin()]}")
    print(f"img_s_text = max : {img_s_text[img_s_text.argmax()]} || min : {img_s_text[img_s_text.argmin()]} \n")

    img = cv2.rectangle(img, rectang_p1, rectang_p2, (255, 255, 255), 3)

    cv2.imshow("show", img)

    cv2.waitKey(1)
    pass

def Perspective(img, Target_location, img_location):
    height, width = img.shape[:2]
    dst_size = (width, height)
    src = Target_location * np.float32(dst_size)
    dst = img_location * np.float32(dst_size)
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, dst_size)
    pass

def line_point(line, w, h): ## (array([759,   0]), array([972, 360]), array([1396,  719]))
    line_mid = line[0, h//2]
    try:
        line_normal = np.where((line[0, :, 0] >= 0) & (line[0, :, 0] <= w))[0]
        line_top = line[0, line_normal][0]
        line_bottom = line[0, line_normal][-1]
    except IndexError:
        line_top = np.array([0, 0])
        line_mid = np.array([0, h//2])
        line_bottom = np.array([0, h])
    line = (line_top, line_mid, line_bottom)
    return line
    pass

def degree(x_y_1, x_y_2):
    x = (x_y_1[0], x_y_2[0])
    y = (x_y_1[1], x_y_2[1])

    m = (y[1] - y[0]) / (x[1] - x[0])

    Degree = round(math.degrees((math.atan(m))), 2)

    if Degree < 0:
        Degree = 90 + (90 - abs(Degree))
    return Degree
    pass

def noise_remove(img, length):
    img_zero_y, img_zero_x = img.nonzero()
    for y in np.unique(img_zero_y):
        img_revmoe = []
        Count = 0
        x = img_zero_x[np.where(img_zero_y == y)]
        if len(x) > length:
            for num, X in enumerate(x):
                if X != x[-1]:
                    if x[num + 1] - X != 1:
                        if Count < length:
                            for i in range(Count):
                                if num - 1 >= 0:
                                    img_revmoe.append(x[num - i - 1])
                        img_revmoe.append(X)
                        Count = 0
                    elif x[num + 1] - X == 1:
                        Count += 1
                else:
                    if Count < length:
                        for i in range(Count):
                            if num - 1 >= 0:
                                img_revmoe.append(x[num - i - 1])
                    img_revmoe.append(X)
        else:
            for X in x:
                img_revmoe.append(X)
        if len(img_revmoe) != 0:
            img[y, img_revmoe] = 0

    return img
    pass

def line_():
    pass

pass
# capture = cv2.VideoCapture("nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)1280, height=(int)720,format=(string)NV12, framerate=(fraction)30/1 ! nvvidconv flip-method=0 ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink")
# capture = cv2.VideoCapture(1)
pass

name = ['WIN_20220101_12_00_43_Pro']
capture = cv2.VideoCapture(f'video/{name[count]}.mp4')

if not capture.isOpened():
    camera = 1
    img = cv2.imread("img/road.jpg")
    # img = cv2.imread("img/test_4.jpg")
    # img = cv2.imread(f"img/road/{img_count}.jpg")
    print('Camera Fail')
elif capture.isOpened():
    camera = 0
pass

while True:
    start_time = time.time()

    if camera == 0:
        ret, img_frame = capture.read()
    elif camera == 1:
        # img = cv2.imread("img/road.jpg")
        img = cv2.imread(f"img/road/{img_count}.jpg")
        img = cv2.flip(img, 1)
        img_frame = copy.deepcopy(img)
        # img_p = copy.deepcopy(img_pp)

    if img_frame is None:
        break

    img_frame = cv2.pyrDown(img_frame)


    # Trekking = Trekking_temp

    height, width = img_frame.shape[:2]

    # ## 이미지 임의 회전 TEST
    # M1 = cv2.getRotationMatrix2D((width // 2, height // 2), count, 1)
    # # img_cap = Perspective(img_frame, img_target_location, img_location) ## 캡쳐용
    # # img_cap = cv2.warpAffine(img_cap, M1, (width, height)) ## 캡쳐용
    # img_frame = cv2.warpAffine(img_frame, M1, (width, height))
    #
    # if count <= rotate and swich == 0:
    #     count += 1
    #     pass
    # elif count >= -rotate and swich == 1:
    #     count -= 1
    #     pass
    # if count > rotate:
    #     swich = 1
    # elif count < -rotate:
    #     swich = 0
    # ##

    img_gauss = cv2.GaussianBlur(img_frame, (5, 5), 0)

    img_hls = cv2.cvtColor(img_gauss, cv2.COLOR_BGR2HLS)

    TopLeft = (.45, .55)
    TopRight = (.55, .55)
    BottomLeft = (.2, .85)
    BottomRight = (.8, .85)

    img_target_location = np.float32([TopLeft, TopRight, BottomLeft, BottomRight])
    # img_target_location = np.float32([(0, 0), (1, 0), (0., 1.0), (1., 1.)])
    img_location = np.float32([(.0, .0), (1., .0), (.0, 1.), (1., 1.)])

    img_p = Perspective(img_hls, img_target_location, img_location)

    # img_p = cv2.pyrDown(img_p)

    color_text(img_p, 15)

    p_height, p_width = img_p.shape[:2]

    alpha = 0
    Fix_center = p_width // 2
    Corrected = Fix_center

    yellow = cv2.inRange(img_p, yellow_lower, yellow_upper)

    # yellow = noise_remove(yellow, 10)

    left_line = copy.deepcopy(yellow)
    right_line = copy.deepcopy(yellow)

    ploty = np.linspace(0, height - 1, height)
    cut_area = (width // 2, int(height * bottom_non))

    ## right
    right_line[-cut_area[1]:, :cut_area[0]] = 0
    right_line[:int(height * 0.2), cut_area[0]:] = 0
    right_zero_y, right_zero_x = right_line.nonzero()

    ## left
    left_line[-cut_area[1]:, cut_area[0]:] = 0
    left_line[:int(height * 0.2), :cut_area[0]] = 0
    left_zero_y, left_zero_x = left_line.nonzero()

    cv2.rectangle(right_line, (0, height), (cut_area[0], height - cut_area[1]), 255, 1)
    cv2.rectangle(left_line, (width, height), (cut_area[0], height - cut_area[1]), 255, 1)

    # f = 1

    if len(right_zero_y) != 0 or len(left_zero_y) != 0:
        ## 초기 그리기
        if len(right_zero_y) != 0 and len(left_zero_y) != 0:
            right_line_color = (0, 255, 0)
            right_a, right_b, right_c = np.polyfit(right_zero_y, right_zero_x, 2)
            right_fitx = right_a * ploty ** 2 + right_b * ploty + right_c
            right_point = np.array([np.transpose(np.vstack([right_fitx, ploty]))]).astype(np.int_)  ## 배열 형태 변경
            rights = line_point(right_point, width, height)

            left_line_color = (0, 255, 0)
            left_a, left_b, left_c = np.polyfit(left_zero_y, left_zero_x, 2)
            left_fitx = left_a * ploty ** 2 + left_b * ploty + left_c
            left_point = np.array([np.transpose(np.vstack([left_fitx, ploty]))]).astype(np.int_)  ## 배열 형태 변경
            lefts = line_point(left_point, width, height)

            right_degree = degree(rights[0], rights[2])
            left_degree = degree(lefts[0], lefts[2])
            mid_point = (np.mean((rights[1][0], lefts[1][0]), dtype=np.int_), height // 2)

            Fix_center = Fix_center + alpha
            one_pixel_mm = mid_point[0] / road_width  ## 1 픽셀 당 mm
            error = Fix_center - mid_point[0]  ## + right || - left
            error_dis = round(error * one_pixel_mm, 2)

            if error_dis <= Dis and error_dis >= -Dis:
                direction = "중앙"
                arduino_temp = "w"
            elif error_dis >= Dis:
                direction = "왼쪽"
                arduino_temp = "q"
            elif error_dis <= -Dis:
                direction = "오른쪽"
                arduino_temp = "e"

            if rights[0][0] == lefts[0][0] or abs(rights[0][0] - lefts[0][0]) < 10:
                # f = 0
                # print("? 예외")
                if rights[0][0] < width // 2:
                    direction = "왼쪽"
                    arduino_temp = "q"
                elif rights[0][0] >= width // 2:
                    direction = "오른쪽"
                    arduino_temp = "e"

            if rights[1][1] >= lefts[2][1]:
                # f = 0
                # print("왼쪽 예외")
                direction = "왼쪽"
                arduino = "w"
                arduino_temp = "q"
            elif lefts[1][1] >= rights[2][1]:
                # f = 0
                # print("오른쪽 예외")
                direction = "오른쪽"
                arduino = "w"
                arduino_temp = "e"
                pass

            direction_state = "직진"
            # print(f"각도 : left {left_degree} right {right_degree}")

        elif len(right_zero_y) != 0 or len(left_zero_y) != 0:
            trekking_zero_y, trekking_zero_x = yellow.nonzero()
            trekking_a, trekking_b, trekking_c = np.polyfit(trekking_zero_y, trekking_zero_x, 2)
            trekking_fitx = trekking_a * ploty ** 2 + trekking_b * ploty + trekking_c
            trekking_point = np.array([np.transpose(np.vstack([trekking_fitx, ploty]))]).astype(np.int_)  ## 배열 형태 변경
            trekkings = line_point(trekking_point, width, height)

            trekking_degree = degree(trekkings[0], trekkings[2])

            if trekking_degree < 90:
                # print(f"각도 우회전 : {trekking_degree}")
                mid_point = (trekkings[1][0] - add, trekkings[1][1])
                direction_state = "우회전"

                arduino_temp = "q"
            elif trekking_degree >= 90:
                # print(f"각도 좌회전 : {trekking_degree}")
                direction_state = "좌회전"
                mid_point = (trekkings[1][0] + add, trekkings[1][1])
                arduino_temp = "e"

            direction = "없음"

        ################################################################################

        # print(f'도로 상태 "{direction_state}" || 현제 차 위치 "{direction}" || 중앙 떨어진값 "{error_dis}" \n')

        if show == 0:
            ## 3채널 변경
            yellow_m = cv2.merge((yellow, yellow, yellow))
            right_line_m = cv2.merge((right_line, right_line, right_line))
            left_line_m = cv2.merge((left_line, left_line, left_line))
            ## 3채널 변경

            cv2.circle(yellow_m, (Fix_center, height // 2), R, (255, 0, 255), -1)  ## 차량 중앙 * FIX *
            cv2.circle(yellow_m, (Fix_center, 0), R, (255, 0, 255), -1)  ## 차량 중앙 * FIX *

            cv2.circle(yellow_m, mid_point, R, (0, 255, 0), -1)  ## 차선 중앙

            if len(right_zero_y) != 0 and len(left_zero_y) != 0:
                cv2.polylines(yellow_m, right_point, False, right_line_color, 10)  ## 도로 오른쪽
                cv2.circle(yellow_m, (rights[0][0], rights[0][1]), R, (255, 0, 0), -1)  ## 오른쪽 차선 끝
                cv2.circle(yellow_m, (rights[1][0], rights[1][1]), R, (255, 0, 255), -1)  ## 오른쪽 차선 중앙
                cv2.circle(yellow_m, (rights[2][0], rights[2][1]), R, (0, 0, 255), -1)  ## 오른쪽 차선 바닥

                cv2.polylines(yellow_m, left_point, False, left_line_color, 10)  ## 도로 왼쪽
                cv2.circle(yellow_m, (lefts[0][0], lefts[0][1]), R, (255, 0, 0), 2)  ## 왼쪽 차선 끝
                cv2.circle(yellow_m, (lefts[1][0], lefts[1][1]), R, (255, 0, 255), 2)  ## 왼쪽 차선 중앙
                cv2.circle(yellow_m, (lefts[2][0], lefts[2][1]), R, (0, 0, 255), 2)  ## 왼쪽 차선 바닥

            elif len(right_zero_y) == 0 or len(left_zero_y) == 0:
                cv2.polylines(yellow_m, trekking_point, False, (0, 255, 255), 10)  ## 도로 왼쪽
                cv2.circle(yellow_m, (trekkings[0][0], trekkings[0][1]), R, (255, 0, 0), 2)  ## 왼쪽 차선 끝
                cv2.circle(yellow_m, (trekkings[1][0], trekkings[1][1]), R, (255, 0, 255), 2)  ## 왼쪽 차선 중앙
                cv2.circle(yellow_m, (trekkings[2][0], trekkings[2][1]), R, (0, 0, 255), 2)  ## 왼쪽 차선 바닥

            ## img_cut area
            Perspective_line = np.int_(np.float32([TopLeft, TopRight, BottomRight, BottomLeft]) * (width, height))
            cv2.polylines(img_frame, [Perspective_line], True, (0, 255, 0), 2) ## 도로 오른쪽
            ##

            # img_frame = cv2.pyrDown(img_frame)

            h_1 = cv2.hconcat([left_line_m, right_line_m])
            h_2 = cv2.hconcat([img_frame, yellow_m])
            v_1 = cv2.vconcat([h_1, h_2])
            v_1 = cv2.resize(v_1, (int(width * 0.8), int(height * 0.8)))

            cv2.imshow("show_1", v_1)

    if revers == 0:
        img_count += 1
    elif revers == 1:
        img_count -= 1

    if img_count > 192 and revers == 0:
        img_count = 1
    elif img_count <= 0 and revers == 1:
        img_count = 192

    # print(img_count)

    key = cv2.waitKey(f)

    end_time = time.time()
    print(end_time - start_time)

    if key == 27:  # ESC 키
        # cv2.imwrite("/home/jetson/Desktop/gg.jpg", img_frame)
        # cv2.imwrite("img/test.jpg", img_cap)
        break

    if key == 47:
        if revers == 0:
            revers = 1
        elif revers == 1:
            revers = 0
        # cv2.waitKey(0)

capture.release()
cv2.destroyAllWindows()