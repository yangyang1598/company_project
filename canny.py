import cv2
import numpy as np

def Perspective(img, Target_location, img_location):
    height, width = img.shape[:2]
    dst_size = (width, height)
    src = Target_location * np.float32(dst_size)
    dst = img_location * np.float32(dst_size)
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, dst_size)
    pass

capture = cv2.VideoCapture("video/WIN_20220101_12_23_10_Pro.mp4")

while True:
    ret, img_frame = capture.read()

    img_frame = cv2.pyrDown(img_frame)

    height, width = img_frame.shape[:2]

    img_gray = cv2.cvtColor(img_frame, cv2.COLOR_BGR2GRAY)

    img_canny = cv2.Canny(img_gray, 60, 120)


    TopLeft = (.45, .55)
    TopRight = (.55, .55)
    BottomLeft = (.2, .85)
    BottomRight = (.8, .85)

    img_target_location = np.float32([TopLeft, TopRight, BottomLeft, BottomRight])
    # img_target_location = np.float32([(0, 0), (1, 0), (0., 1.0), (1., 1.)])
    img_location = np.float32([(.0, .0), (1., .0), (.0, 1.), (1., 1.)])

    img_p = Perspective(img_canny, img_target_location, img_location)

    img_canny = cv2.merge([img_canny, img_canny, img_canny])

    Perspective_line = np.int_(np.float32([TopLeft, TopRight, BottomRight, BottomLeft]) * (width, height))
    cv2.polylines(img_canny, [Perspective_line], True, (0, 255, 0), 2)  ## 도로 오른쪽

    cv2.imshow("2", img_canny)
    cv2.imshow("1", img_p)

    key = cv2.waitKey(1)

    if key == 27:
        break
    pass

capture.release()
cv2.destroyAllWindows()