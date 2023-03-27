import cv2
import imutils
import math
from PIL import Image
import numpy as np
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract.exe"
# Đọc ảnh vào biến image
image = cv2.imread("C:/Users/locph/OneDrive/Desktop/picture/car_4.jpg")
(height, width) = image.shape[:2]
if width < 750:
    ratio = 700 / width
    image = cv2.resize(image, (int(width * ratio), int(height * ratio)))

# Chuyển đổi ảnh thành ảnh xám
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Làm mịn hình ảnh để loại bỏ nhiễu
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
kernel=np.ones((3,3), np.uint8)
tophat=cv2.morphologyEx(blurred, cv2.MORPH_TOPHAT, kernel, iterations=100)
blackhat=cv2.morphologyEx(blurred, cv2.MORPH_BLACKHAT, kernel, iterations=100)
combined= cv2.subtract(tophat, blackhat)
# Phát hiện cạnh trên ảnh đã được xử lý
edged = cv2.Canny(combined, 50, 150,apertureSize=3)
thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

# Tìm các contours trong ảnh đã phát hiện cạnh
contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
#contours = imutils.grab_contours(contours)
# Lọc các contours không phải là tấm số đăng ký
# plate_contour = None
# for contour in contours:
#     area = cv2.contourArea(contour)
#     if area > 5000:
#         plate_contour = contour
#         break

# Lọc các contours hình chữ nhật có 4 góc gần bằng 90 độ
plate_contour = []
for contour in contours:
    # Xác định đường bao của contour
    perimeter = cv2.arcLength(contour, True)

    # Xác định đa giác xấp xỉ của contour
    approx = cv2.approxPolyDP(contour, 0.025 * perimeter, True)

    # Nếu contour có 4 đỉnh
    if len(approx) == 4:
        # Tính góc của các cạnh của contour
        angles = []
        for i in range(4):
            j = (i + 1) % 4
            k = (i + 2) % 4
            angle = np.degrees(
                np.arctan2(
                    approx[j][0][1] - approx[i][0][1],
                    approx[j][0][0] - approx[i][0][0],
                )
                - np.arctan2(
                    approx[k][0][1] - approx[j][0][1],
                    approx[k][0][0] - approx[j][0][0],
                )
            )
            angle = np.abs(angle - 180) if angle > 90 else angle
            angles.append(angle)

        # Nếu các cạnh có góc gần bằng 90 độ thì contour đó là hình chữ nhật
        if np.all(np.array(angles) < 95):
            plate_contour.append(approx)


if plate_contour is not None:
    # Vẽ tấm số đăng ký lên ảnh gốc
    cv2.drawContours(image, plate_contour, -1, (0, 255, 0), 3)

    # Cắt ảnh tấm số đăng ký
    mask = np.zeros(gray.shape, np.uint8)
    new_image = cv2.drawContours(mask, plate_contour, 0, 255, -1, )
    new_image = cv2.bitwise_and(image, image, mask=mask)
    (x, y) = np.where(mask == 255)
    (topx, topy) = (np.min(x)-1, np.min(y)-1)
    (bottomx, bottomy) = (np.max(x)+1, np.max(y)+1)
    crop = image[topx:bottomx +1, topy:bottomy +1]
    gray_crop = blurred[topx -5:bottomx +5, topy:bottomy ]
    edged_crop =edged[topx:bottomx +5, topy:bottomy ]
    thresh_crop =thresh[topx -5:bottomx +5, topy :bottomy ]
    #new_image = new_image[topx -5:bottomx +5, topy :bottomy ]

    # Chuyển đảo ngược màu sáng tối của ảnh tấm số đăng ký
    inverted_crop = cv2.bitwise_not(gray_crop)
    inverted_crop = cv2.equalizeHist(inverted_crop)
    
    
    data1 = pytesseract.image_to_string(new_image, lang='eng', config='--psm 6')
    print('Case 1:',data1)
    data2 = pytesseract.image_to_string(thresh_crop, lang='eng', config='--psm 6')
    print('Case 2:',data2)
    data3 = pytesseract.image_to_string(inverted_crop, lang='eng', config='--psm 6')
    print('Case 3:', data3)
    
    # Nhận dạng biển số thuộc tỉnh nào
    with open("C:/Users/locph/OneDrive/Desktop/biensoxe.txt", 'r', encoding='utf-8') as file:
        content = file.read().splitlines()

    def func(data, flag):
        for line in content:
            if line[:2]==data[:2]:
                if not flag[0]:
                    print('Biển số xe này của: ' + line[3:])
                    flag[0] = True
        return flag
    flag = [False]
    func(data1, flag)
    func(data2, flag)
    func(data3, flag)
            

    # Hiển thị ảnh đã được xử lý
    #cv2.imshow('Edge image', edged)
    #cv2.imshow('Gray image', gray)
    cv2.imshow('image', image)
    cv2.imshow('Inverted Cropped', inverted_crop)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Không tìm thấy tấm số đăng ký trong ảnh.")
    cv2.imshow('Edge image', edged)
    cv2.imshow('Gray image', gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()