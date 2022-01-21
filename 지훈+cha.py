
import myopencv as my
import cv2, copy
import numpy as np
import time
def binary(thresold, binary_img):
    binary = np.zeros_like(binary_img)
    binary[(binary_img >= thresold[0]) & (binary_img <= thresold[1])] = 255
    return binary
    pass

def cut_img(img,src,dst):
    height, width = img.shape[:2]
    dst_size = (width, height)
    src = src * np.float32([width, height])  ## width, height 비율 값
    dst = dst * np.float32(dst_size)  ## 이미지를 적용할 화면 비율
    M = cv2.getPerspectiveTransform(src, dst)  ## 자를 이미지 좌표값
    img_src = cv2.warpPerspective(img, M, dst_size)  ## 잘라낼 이미지, 잘라낼 이미지 영역값, 잘라낼 이미지를 붙일 영역 사이즈
    return img_src
    pass

margin = 150
nwindows = 9
minpix = 1

img_ = []
count = 0
frame_start = 0
frame_count = 0
frame = 0
A = 0
step = 1 # 한번에 넘어갈 프레임수
line_len = 100
road_width = 2.5
_ = True

left_a, left_b, left_c = [], [], []
right_a, right_b, right_c = [], [], []

leftx_base_ = []
rightx_base_ = []

leftx_base_step = []
rightx_base_step = []

leftx_ = []
lefty_ = []
rightx_ = []
righty_ = []

left_fit_ = np.empty(3)
right_fit_ = np.empty(3)

name = ['WIN_20220101_12_00_43_Pro']
Video = cv2.VideoCapture(f'video/{name[count]}.mp4')
while True:
    start=time.time()
    _, img = Video.read()

    #img = my.Undistort(img, 'giup.p')
    img=cv2.pyrDown(img)

    img_blur = cv2.blur(img, (5, 5), 0)

    img_hsv = cv2.cvtColor(img_blur, cv2.COLOR_BGR2HSV)


    height, width = img.shape[:2]

    if A == 0:
        A = 1
        pass
    ## 블랙박스
    TopLeft = (.45, .55)
    TopRight = (.535, .55)
    BottomLeft = (.2, .84)
    BottomRight = (.785, .84)


    ## 흰색
    white_lower = (110, 135, 120)
    white_upper = (190, 220, 235)
    ## 흰색

    factor = np.float32([width, height])
    src = np.float32([TopLeft, TopRight, BottomLeft, BottomRight])
    dst = np.float32([(.0, .0), (1., .0), (.0, 1.), (1., 1.)])

    img_perspect = cut_img(img, src, dst)

    temp = np.zeros_like(img)
    img_r, img_g, img_b = cv2.split(img_perspect)

    img_b, img_g, img_r = np.mean(img_b), np.mean(img_g), np.mean(img_r)
    #print(img_r, img_g, img_b)
    if (img_r >= 50 and img_r <= 80) and (img_g >= 70 and img_g <= 105) and (img_b >= 85 and img_b <= 125):
        white_lower1 = (150, 180, 195)  # (10, 140, 10)
        white_lower2 = (110, 135, 120)
        white_upper1 = (205, 225, 235)  # (30, 190, 40)
        white_upper2 = (145, 165, 190)
        img_mask11 = cv2.inRange(img_perspect, white_lower1, white_upper1)
        img_mask12 = cv2.inRange(img_perspect, white_lower2, white_upper2)
        img_mask1 = img_mask11 + img_mask12
        #print("dark")
    elif (img_r > 80 and img_r <= 105) and (img_g > 105 and img_g <= 134) and (img_b > 125 and img_b <= 158):
        white_lower1 = (120, 150, 160)  # (10, 140, 10)
        white_upper1 = (145, 180, 200)
        img_mask1 = cv2.inRange(img_perspect, white_lower1, white_upper1)

        #print("semi-dark")
    elif (img_r > 105 and img_r <= 135) and (img_g > 134 and img_g <= 165) and (img_b > 158 and img_b <= 180):
        white_lower = (150, 170, 180)  # (10, 140, 10)
        white_upper = (190, 210, 215)  # (30, 190, 40)
        img_mask1 = cv2.inRange(img_perspect, white_lower, white_upper)
        #print("semi-light")
    elif (img_r > 135 and img_r <= 175) and (img_g > 165 and img_g <= 190) and (img_b > 180 and img_b <= 230):
        white_lower1 = (150, 170, 180)  # (10, 140, 10)
        white_upper1 = (190, 210, 215)  # (30, 190, 40)
        white_lower2 = (170, 180, 190)  # (10, 140, 10)  (110, 135, 145)
        white_upper2 = (190, 210, 220)  # (30, 190, 40)  (160, 175, 175)
        img_mask11 = cv2.inRange(img_perspect, white_lower1, white_upper1)
        img_mask12 = cv2.inRange(img_perspect, white_lower2, white_upper2)
        img_mask1 = img_mask11 + img_mask12
        #print("light")
    else:
        if (img_g >= 155):
            white_lower = (150, 170, 180)  # (10, 140, 10)
            white_upper = (190, 210, 215)  # (30, 190, 40)
            img_mask1 = cv2.inRange(img_perspect, white_lower, white_upper)
            #print("else1")
        elif (img_b>=155):
            white_lower = (150, 170, 180)  # (10, 140, 10)
            white_upper = (190, 210, 215)  # (30, 190, 40)
            img_mask1 = cv2.inRange(img_perspect, white_lower, white_upper)
            #print("else2")
        elif (img_g>=130 or img_r>=155):
            white_lower2 = (150, 170, 180)  # (10, 140, 10)
            white_upper2 = (190, 210, 215)  # (30, 190, 40)
            # img_mask11 = cv2.inRange(img_perspect, white_lower1, white_upper1)
            img_mask1 = cv2.inRange(img_perspect, white_lower2, white_upper2)
            # img_mask1 = img_mask11+img_mask12

            #print("else3")
        else:
            white_lower1 = (48, 48, 40)
            white_upper1 = (95, 95, 75)
            white_lower2 = (120, 150, 160)  # (10, 140, 10)

            white_upper2 = (145, 180, 200)

            img_mask11 = cv2.inRange(img_perspect, white_lower1, white_upper1)
            img_mask12 = cv2.inRange(img_perspect, white_lower2, white_upper2)
            img_mask1 = img_mask11+img_mask12
            #print("else4")

    #img_white = cv2.inRange(img_perspect, white_lower, white_upper)
    img_white_line = cv2.bitwise_and(img_perspect, img_perspect, mask=img_mask1)

    img_white_line_blur = cv2.blur(img_white_line, (3, 3), 1)

    img_gray = cv2.cvtColor(img_white_line_blur, cv2.COLOR_BGR2GRAY)
    # 이미지 블러
    img_gray = cv2.GaussianBlur(img_gray, (3, 3), 0)
    _, img_bin = cv2.threshold(img_gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)

    img_canny = cv2.Canny(img_bin, 50, 150)

    ## 네모 상자 그리기
    histogram = np.sum(img_canny[height // 2:, :], axis=0)
    ## histogram 설명 : numpy.sum 설명 읽기
    ## x,y 2차원 배열을 y축 반틈 아래 부분을 y축을 제거 하고
    ## x축의 값만 더한 값
    midpoint = int(histogram.shape[0] / 2)
    ## midpoint 설명 : histogram의 반틈 길이 좌,우 나누기 위함
    leftx_base = np.argmax(histogram[:midpoint])
    ## leftx_base 설명 : histogram의 0 ~ midpoint 중 제일 큰값
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    ## rightx_base 설명 : histogram의 midpoint ~ end 중 제일 큰값

    leftx_base_.append(leftx_base)
    rightx_base_.append(rightx_base)

    leftx_base = np.int64(np.mean(leftx_base_[-10:]))
    rightx_base = np.int64(np.mean(rightx_base_[-10:]))

    window_height = int(height / nwindows)
    ## window_height 설명 : 사각형 범위 높이 겟수
    nonzero = img_canny.nonzero()
    ## nonzero 설명 : 0이 아닌 값인 x, y 인덱스값 분리
    nonzero_y = np.array(nonzero[0])
    ## nonzero_y 설명 : y축의 0이 아닌 인덱스
    nonzero_x = np.array(nonzero[1])
    ## nonzero_x 설명 : x축의 0이 아닌 인덱스

    leftx_current = leftx_base
    rightx_current = rightx_base

    left_lane_inds = []
    right_lane_inds = []

    # 네모 상자 그리기
    for window in range(nwindows):
        win_y_low = height - (window + 1) * window_height
        win_y_high = height - window * window_height

        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin

        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        cv2.rectangle(temp, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high),(100, 255, 255), 3)
        cv2.rectangle(temp, (win_xright_low, win_y_low), (win_xright_high, win_y_high),(100, 255, 255), 3)
        cv2.rectangle(img_canny, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (100, 255, 255), 3)
        cv2.rectangle(img_canny, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (100, 255, 255), 3)

        good_left_inds = ((nonzero_y >= win_y_low) &
                          (nonzero_y < win_y_high) &
                          (nonzero_x >= win_xleft_low) &
                          (nonzero_x < win_xleft_high)).nonzero()[0]
        ## good_left_inds 설명 : 왼쪽 사각형 범위 안에 x값의 0이 아닌 값 추출

        good_right_inds = ((nonzero_y >= win_y_low) &
                           (nonzero_y < win_y_high) &
                           (nonzero_x >= win_xright_low) &
                           (nonzero_x < win_xright_high)).nonzero()[0]
        ## good_right_inds 설명 : 오른쪽 사각형 범위 안에 x값의 0이 아닌 값 추출

        left_lane_inds.append(good_left_inds)
        ## left_lane_inds 설명 : good_left_inds 값을 left_lane_inds에 저장
        right_lane_inds.append(good_right_inds)
        ## right_lane_inds 설명 : good_right_inds 값을 right_lane_inds 저장

        leftx_base_step.append(leftx_current)
        rightx_base_step.append(rightx_current)

        if len(good_left_inds) > minpix:
            leftx_current = int(np.mean(nonzero_x[good_left_inds]))
            ## leftx_current 설명 : nonzero_x 안에 good_left_inds 인덱스 값의 평균 값
            pass
        elif len(good_left_inds) == 0:
            leftx_current = int(np.mean(leftx_base_step[-10:]))
            pass
        if len(good_right_inds) > minpix:
            rightx_current = int(np.mean(nonzero_x[good_right_inds]))
            ## rightx_current 설명 : nonzero_x 안에 good_right_inds 인덱스 값의 평균 값
            pass
        elif len(good_right_inds) == 0:
            rightx_current = int(np.mean(rightx_base_step[-10:]))
            pass
        # 네모 상자 그리기
    # 네모 상자 그리기

    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    ## left_lane_inds, right_lane_inds 설명 : 2차 배열을 1차 배열로 합침


    leftx = nonzero_x[left_lane_inds]
    ## leftx 설명 : nonzero_x 안에 left_lane_inds 인덱스 값
    lefty = nonzero_y[left_lane_inds]
    ## lefty 설명 : nonzero_y 안에 left_lane_inds 인덱스 값
    rightx = nonzero_x[right_lane_inds]
    ## rightx 설명 : nonzero_x 안에 right_lane_inds 인덱스 값
    righty = nonzero_y[right_lane_inds]
    ## righty 설명 : nonzero_y 안에 right_lane_inds 인덱스 값

    leftx_.append(leftx)
    lefty_.append(lefty)
    rightx_.append(rightx)
    righty_.append(righty)
    if len(left_lane_inds) == 0:

        for i in range(len(leftx_)):
            if len(leftx_[-(i + 1)]) != 0:

                leftx = leftx_[-(i + 1)]

                lefty = lefty_[-(i + 1)]


                print("왼쪽 에러")
                break
                pass
            pass
        pass
    if len(right_lane_inds) == 0:
        for i in range(len(rightx_)):
            if len(rightx_[-(i + 1)]) != 0:
                rightx = rightx_[-(i + 1)]
                righty = righty_[-(i + 1)]
                print("오른쪽 에러")
                break
                pass
            pass
        pass

    left_fit = np.polyfit(lefty, leftx, 2)
    ## left_fit 설명 : np.polyfit를 통해 lefty, leftx 값에 대한 차수가 2인 값을 반환 (곡선)
    right_fit = np.polyfit(righty, rightx, 2)
    ## right_fit 설명 : np.polyfit를 통해 righty, rightx 값에 대한 차수가 2인 값을 반환 (곡선)

    left_a.append(left_fit[0])
    left_b.append(left_fit[1])
    left_c.append(left_fit[2])
    right_a.append(right_fit[0])
    right_b.append(right_fit[1])
    right_c.append(right_fit[2])

    left_fit_[0] = np.mean(left_a[-10:])
    left_fit_[1] = np.mean(left_b[-10:])
    left_fit_[2] = np.mean(left_c[-10:])
    right_fit_[0] = np.mean(right_a[-10:])
    right_fit_[1] = np.mean(right_b[-10:])
    right_fit_[2] = np.mean(right_c[-10:])


    ploty = np.linspace(0, height - 1, height)
    ## ploty 설명 : np.linspace 통해 0 ~ 이미지 height 값 까지 값을 순서대로 배열 생성

    left_fitx = left_fit_[0] * ploty ** 2 + left_fit_[1] * ploty + left_fit_[2]
    ## left_fitx 설명 : X = a[0] * y^2 + a[1] * y + a[2] // 2차 다항식 회귀 공식
    right_fitx = right_fit_[0] * ploty ** 2 + right_fit_[1] * ploty + right_fit_[2]
    ## right_fitx 설명 : X = a[0] * y^2 + a[1] * y + a[2] // 2차 다항식 회귀 공식

    mid_fitx = (left_fitx + right_fitx) // 2

    temp[nonzero_y[left_lane_inds], nonzero_x[left_lane_inds]] = [255, 0, 100]
    temp[nonzero_y[right_lane_inds], nonzero_x[right_lane_inds]] = [0, 100, 255]

    left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    mid = np.array([np.transpose(np.vstack([mid_fitx, ploty]))])
    points = np.hstack((left, right))

    left_center = np.int_(np.mean(left_fitx))
    right_center = np.int_(np.mean(right_fitx))
    mid_center = np.int_(np.mean(mid_fitx))
    road_center = width // 2

    road_pixel = road_width / (right_center - left_center)
    error = mid_center - road_center
    error_pixel = error * road_pixel



    cv2.polylines(temp, np.int_(points), False, (0, 255, 255), 10)


    cv2.line(temp, (left_center + 10, height // 2 + line_len), (left_center - 10, height // 2 - line_len), (255, 0, 255), 30)
    cv2.line(temp, (right_center + 10, height // 2 + line_len), (right_center - 10, height // 2 - line_len), (255, 0, 255), 30)
    cv2.line(temp, (mid_center + 10, height // 2 + line_len), (mid_center - 10, height // 2 - line_len), (255, 0, 255), 30)
    cv2.line(temp, (width // 2 - 10, height // 2), (width // 2 + 20, height), (255, 0, 255), 10)
    cv2.line(temp, (mid_center, height // 2), (width // 2 - 10, height // 2), (255, 255, 255), 30)

    temp = cut_img(temp, dst, src)


    img_result = cv2.addWeighted(img, 1., temp, 0.4, 0)
    if error > 0:
        cv2.putText(img_result, f'right : {abs(error_pixel):.2f}m', (width // 2 - 50, height // 2 + 200), cv2.FONT_ITALIC, 0.5,
                    (0, 0, 255), 2)
        if error_pixel>0.7:
            cv2.putText(img_result, f'WARRING!', (width // 2 - 50,20),cv2.FONT_ITALIC, 0.5,(0, 0, 255), 2)
    elif error < 0:
        cv2.putText(img_result, f'left : {abs(error_pixel):.2f}m', (width // 2 - 50, height // 2 + 200), cv2.FONT_ITALIC, 0.5,
                    (0, 0, 255), 2)
        if error_pixel<-0.7:
            cv2.putText(img_result, f'WARRING!', (width // 2 - 50, 20), cv2.FONT_ITALIC, 0.5, (0, 0, 255), 2)

    elif error == 0:
        cv2.putText(img_result, f'center', (width // 2 - 50, height // 2 + 200), cv2.FONT_ITALIC, 0.5,
                    (0, 0, 255), 2)


    _, img_result = my.imgsize(img_result, 1920)

    my.imgshow("result", img_result, 1000)

    key = cv2.waitKey(3)
    end=time.time()
    # print(f'end={end-start}')


    if key == 32:#spacebar
        key=cv2.waitKey(0)
    elif key == 27:
        Video.release()
        cv2.destroyAllWindows()
        break
        pass
    pass
Video.release()
cv2.destroyAllWindows()
pass
