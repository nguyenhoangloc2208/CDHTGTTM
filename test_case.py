from PIL import Image, ImageTk
import cv2
import numpy as np
import pytesseract
import strProcessing
pytesseract.pytesseract.tesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract.exe"
MIN_PLATE_ANGLE = 95
MORPH_KERNEL = np.ones((3, 3), np.uint8)
MORPH_ITERATIONS = 100
CANNY_THRESH_1 = 50
CANNY_THRESH_2 = 150
CANNY_THRESH_3 = 250
CANNY_THRESH_4 = 255
OTSU_THRESH = 0
n = 1

while True:
    try:
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # xử lý khung hình ở đây

            cv2.imshow('frame', frame)
            if cv2.waitKey(1) == ord('q'):
                break

        def extra_plate(frame):
            print('Chạy thành công')
            frame_temp = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            imagePIL = Image.fromarray(frame_temp)
            width = 1000
            height = int(width * imagePIL.size[1] / imagePIL.size[0])
            imagePIL = imagePIL.resize((width, height))

            # Chuyển đổi đối tượng image_PIL thành một mảng numpy
            image = np.array(imagePIL)
            # Chuyển đổi từ không gian màu RBG sang BGR
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            ################# Xử lí ảnh###############
            # Chuyển đổi sang ảnh xám
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Áp dụng bộ lọc GaussianBlur để làm mờ ảnh
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            # Sử dụng hàm morphologyEx tạo ảnh TOPHAT
            tophat = cv2.morphologyEx(
                blurred, cv2.MORPH_TOPHAT, MORPH_KERNEL, iterations=100)
            # Sử dụng hàm morphologyEx tạo ảnh BLACKHAT
            blackhat = cv2.morphologyEx(
                blurred, cv2.MORPH_BLACKHAT, MORPH_KERNEL, iterations=100)
            # Combined ảnh được xử lý bằng hàm morphologyEx
            combined = cv2.subtract(tophat, blackhat)
            reCombined = cv2.subtract(blackhat, tophat)
            reCombined = cv2.bitwise_not(reCombined)
            # Tạo ảnh canny từ ảnh combined với thông số 50,150
            edged = cv2.Canny(combined, CANNY_THRESH_1,
                              CANNY_THRESH_2, apertureSize=3)
            # Tạo ảnh canny với thông số 250,255
            Edged = cv2.Canny(blurred, CANNY_THRESH_3, CANNY_THRESH_4)
            thresh = cv2.threshold(blurred, OTSU_THRESH, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[
                1]  # Tạo ảnh đen và trắng
            contours, _ = cv2.findContours(
                edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)  # Tìm contour từ ảnh edge
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[
                :10]  # Lọc ra top 10 contour
            ############### Lọc contour################

            plate_contour = []  # Danh sách các contour của tấm biển số
            for contour in contours:
                perimeter = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
                if len(approx) == 4:
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
                    if np.all(np.array(angles) < MIN_PLATE_ANGLE):
                        plate_contour.append(approx)

            ################# Nhận diện biển số xe################

            if plate_contour is not None:
                # Vẽ đường viền xung quanh biển số vào ảnh image, -1 cho biết vẽ tất cả các đường viền, 2 là độ dày
                cv2.drawContours(image, plate_contour, -1, (0, 255, 0), 2)
                # Tạo ma trận mask với kích thước ảnh gray
                mask = np.zeros(gray.shape, np.uint8)
                # Chỉ định vẽ contour đầu tiên, 0 là đầu tiên, -1  cho biết vẽ đầy đủ contour
                new_image = cv2.drawContours(mask, plate_contour, 0, 255, -1, )
                # Tạo ảnh nhị phân chỉ chứa tấm biển số bằng phép toán AND bit
                new_image = cv2.bitwise_and(gray, gray, mask=mask)
                # Tìm vị trí của biển số xe
                (x, y) = np.where(mask == 255)
                if len(x) > 0 and len(y) > 0:
                    (topx, topy) = (np.min(x), np.min(y))
                    (bottomx, bottomy) = (np.max(x), np.max(y))
                    # topx trên, topy trái, botx dưới, boty phải
                    # Cắt biển số trên từng trường hợp
                    crop = image[topx:bottomx+3, topy+3:bottomy-3]
                    new_image = new_image[topx:bottomx+3, topy+3:bottomy-3]
                    gray_crop = blurred[topx:bottomx+3, topy+3:bottomy-3]
                    thresh_crop = thresh[topx:bottomx+3, topy+3:bottomy-3]
                    reCombinedCrop = reCombined[topx:bottomx +
                                                3, topy+3:bottomy-3]
                    thresh_baw, blackandwhite = cv2.threshold(
                        gray_crop, 85, 255, cv2.THRESH_BINARY)
                    inverted_crop = cv2.bitwise_not(gray_crop)
                    inverted_crop = cv2.bitwise_not(inverted_crop)
                    inverted_crop = cv2.equalizeHist(inverted_crop)

                    # Đọc biển số xe
                    case1 = pytesseract.image_to_string(
                        blackandwhite, lang='eng', config='--psm 6')
                    case2 = pytesseract.image_to_string(
                        thresh_crop, lang='eng', config='--psm 6')
                    case3 = pytesseract.image_to_string(
                        gray_crop, lang='eng', config='--psm 6')
                    case4 = pytesseract.image_to_string(
                        crop, lang='eng', config='--psm 6')
                    case5 = pytesseract.image_to_string(
                        new_image, lang='eng', config='--psm 6')
                    case6 = pytesseract.image_to_string(
                        inverted_crop, lang='eng', config='--psm 6')
                    case7 = pytesseract.image_to_string(
                        reCombinedCrop, lang='eng', config='--psm 6')

                    flag = [False]
                    place = None
                    strProcessing.str_processing(case2, flag, place)
                    strProcessing.str_processing(case3, flag, place)
                    strProcessing.str_processing(case1, flag, place)
                    strProcessing.str_processing(case5, flag, place)
                    strProcessing.str_processing(case4, flag, place)
                    strProcessing.str_processing(case6, flag, place)
                    strProcessing.str_processing(case7, flag, place)
                    return crop
                else:
                    print('Cắt biển số thất bại')
                    return None
            else:
                print("Không tìm thấy tấm số đăng ký trong ảnh.")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                return None

        result_image = extra_plate(frame)
        if result_image is not None and result_image.any():
            cv2.imshow('result', result_image)
        else:
            pass

    except:
        pass
