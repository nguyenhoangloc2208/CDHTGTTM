import tkinter as tk
from tkinter.filedialog import askopenfilename
from tkinter import filedialog
from PIL import Image, ImageTk
import random
import cv2
import numpy as np
import pytesseract
win = tk.Tk()
win.geometry("500x400")

pytesseract.pytesseract.tesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract.exe"
MIN_PLATE_ANGLE = 95
MORPH_KERNEL = np.ones((3, 3), np.uint8)
MORPH_ITERATIONS = 100
CANNY_THRESH_1 = 50
CANNY_THRESH_2 = 150
CANNY_THRESH_3 = 250
CANNY_THRESH_4 = 255
OTSU_THRESH = 0

def extra_plate():
    "Mở file để có thể chỉnh sửa"
    filepath = tk.filedialog.askopenfilename(initialdir="/", title="Select An Image",
        filetypes=(("Text Files", "*.jpg"), ("All Files", "*.*"))
    )
    image_PIL=Image.open(filepath)
    image=np.array(image_PIL)
    image=cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    (height, width) = image.shape[:2]
    if width < 750:
        ratio = 700 / width
        image = cv2.resize(image, (int(width * ratio), int(height * ratio)))
    elif width>1000:
        ratio = 800 / width
        image = cv2.resize(image, (int(width * ratio), int(height * ratio)))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    tophat=cv2.morphologyEx(blurred, cv2.MORPH_TOPHAT, MORPH_KERNEL, iterations=100)
    blackhat=cv2.morphologyEx(blurred, cv2.MORPH_BLACKHAT, MORPH_KERNEL, iterations=100)
    combined= cv2.subtract(tophat, blackhat)
    edged = cv2.Canny(combined, CANNY_THRESH_1, CANNY_THRESH_2,apertureSize=3)
    Edged = cv2.Canny(blurred, CANNY_THRESH_3, CANNY_THRESH_4)
    thresh = cv2.threshold(blurred, OTSU_THRESH, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    plate_contour = []
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.025 * perimeter, True)
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
    if plate_contour is not None:
        cv2.drawContours(image, plate_contour, -1, (0, 255, 0), 3)
        mask = np.zeros(gray.shape, np.uint8)
        new_image = cv2.drawContours(mask, plate_contour, 0, 255, -1, )
        new_image = cv2.bitwise_and(image, image, mask=mask)
        (x, y) = np.where(mask == 255)
        (topx, topy) = (np.min(x)-1, np.min(y)-1)
        (bottomx, bottomy) = (np.max(x)+1, np.max(y)+1)
        crop = image[topx-5:bottomx, topy-5:bottomy+5 ]
        gray_crop = blurred[topx-5:bottomx, topy-5:bottomy ]
        edged_crop =edged[topx-5:bottomx, topy-5:bottomy ]
        Edged_crop =edged[topx-5:bottomx, topy-5:bottomy ]
        thresh_crop =thresh[topx-5:bottomx, topy-5:bottomy ]
        thresh_baw, blackandwhite = cv2.threshold(gray_crop, 85,255,cv2.THRESH_BINARY)
        inverted_crop = cv2.bitwise_not(gray_crop)
        inverted_crop = cv2.equalizeHist(inverted_crop)
        case1 = pytesseract.image_to_string(new_image, lang='eng', config='--psm 6')
        case2 = pytesseract.image_to_string(Edged_crop, lang='eng', config='--psm 6')
        case3 = pytesseract.image_to_string(thresh_crop, lang='eng', config='--psm 6')
        case4 = pytesseract.image_to_string(inverted_crop, lang='eng', config='--psm 6')
        case5 = pytesseract.image_to_string(blackandwhite, lang='eng', config='--psm 6')
        case6 = pytesseract.image_to_string(crop, lang='eng', config='--psm 6')
        with open("C:/Users/locph/OneDrive/Desktop/biensoxe.txt", 'r', encoding='utf-8') as file:
            content = file.read().splitlines()
        plate = None
        (Height, Width) = crop.shape[:2]
        if 2*Height<Width:
            plate = 0
        else:
            plate = 1    
        def func(data, flag, place, plate):
            data = data.translate(str.maketrans("", "", " ;|"))
            for line in content:
                    if line[:2]==data[:2] and len(data)>4:
                        if plate == 0:
                            if data[2] == '1':
                                data = data[:2] + 'I' + data[3:]
                            elif data[2] == '8':
                                data = data[:2] + 'B' + data[3:]
                            elif data[2] == '5':
                                data = data[:2] + 'S' + data[3:]
                            elif data[2] == '2':
                                data = data[:2] + 'Z' + data[3:]
                            elif data[2] == '6':
                                data = data[:2] + 'G' + data[3:]
                            elif data[2] == '4':
                                data = data[:2] + 'A' + data[3:]
                            print('data' + data)
                            if data[2].isupper() and not data[2].isdigit():
                                if not flag[0]:
                                    print(data)
                                    print('Biển số xe này của: ' + line[3:])
                                    place=line[3:]
                                    flag[0] = True
                        elif plate ==1:
                            if data[3] == '1':
                                data = data[:3] + 'T' + data[4:]
                            if data[0] == 'A':
                                data = '4' + data[1:]
                            if data[3:].count("G") > 0:
                                data = data.replace("G", "6", 1)
                            if not flag[0]:
                                print(data)
                                print('Biển số xe này của: ' + line[3:])
                                place=line[3:]
                                flag[0] = True
            return flag,place
        flag = [False]
        place = None
        func(case6, flag, place, plate)
        func(case5, flag, place, plate)
        func(case4, flag, place, plate)
        func(case3, flag, place, plate)
        func(case2, flag, place, plate)
        func(case1, flag, place, plate)
        cv2.imshow('image', image)
        cv2.imshow('Inverted Cropped', inverted_crop)
    else:
        print("Không tìm thấy tấm số đăng ký trong ảnh.")
        cv2.imshow('Edge image', edged)
        cv2.imshow('Gray image', gray)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return crop, place

img_var = None

def update_image():
    global img_var
    img_cv2, place = extra_plate()
    (h,w)=img_cv2.shape[:2]
    if 2*h <w:
        img_cv2=cv2.resize(img_cv2, (500,(int(h*(500/w)))), interpolation=cv2.INTER_LINEAR)
    else:
        img_cv2=cv2.resize(img_cv2, (500,300), interpolation=cv2.INTER_LINEAR)
    img= Image.fromarray(cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB))
    img_var = ImageTk.PhotoImage(img)
    display_image(canvas, img_var)
    label = tk.Label(win,text=place)
    label.pack()
    
    

button=tk.Button(win, text="Button", command=update_image)


#img_path="C:/Users/locph/OneDrive/Desktop/picture/car_1.jpg"

def display_image(canvas, photo):
    # Hiển thị hình ảnh trên Canvas
    canvas.create_image(0, 0, image=photo, anchor=tk.NW)
    
canvas = tk.Canvas(win, width=500, height=300, bg="grey")
canvas.pack()
button.pack()


# frame=tk.Frame(win)
# frame.config(width=200,height=100,background="black")
# frame.pack()



win.mainloop()