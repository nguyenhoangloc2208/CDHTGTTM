import tkinter as tk
from tkinter.filedialog import askopenfilename
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import pytesseract
import strProcessing
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
    #Mở file để có thể chỉnh sửa
    filepath = tk.filedialog.askopenfilename(initialdir="/", title="Select An Image",
        filetypes=(("Text Files", "*.jpg"), ("All Files", "*.*"))
    )
    #Mở ảnh bằng thư viện Pillow với nguồn là filepath
    image_PIL=Image.open(filepath)
    #Resize lại ảnh
    width = 1000
    height = int(width * image_PIL.size[1] / image_PIL.size[0])
    image_PIL=image_PIL.resize((width, height))
    #Chuyển đổi đối tượng image_PIL thành một mảng numpy
    image=np.array(image_PIL)
    #Chuyển đổi từ không gian màu RBG sang BGR
    image=cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    #################Xử lí ảnh###############
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #Chuyển đổi sang ảnh xám
    blurred = cv2.GaussianBlur(gray, (5, 5), 0) #Áp dụng bộ lọc GaussianBlur để làm mờ ảnh
    tophat=cv2.morphologyEx(blurred, cv2.MORPH_TOPHAT, MORPH_KERNEL, iterations=100)#Sử dụng hàm morphologyEx tạo ảnh TOPHAT
    blackhat=cv2.morphologyEx(blurred, cv2.MORPH_BLACKHAT, MORPH_KERNEL, iterations=100)#Sử dụng hàm morphologyEx tạo ảnh BLACKHAT
    combined= cv2.subtract(tophat, blackhat) #Combined ảnh được xử lý bằng hàm morphologyEx
    edged = cv2.Canny(combined, CANNY_THRESH_1, CANNY_THRESH_2,apertureSize=3)#Tạo ảnh canny từ ảnh combined với thông số 50,150
    Edged = cv2.Canny(blurred, CANNY_THRESH_3, CANNY_THRESH_4)#Tạo ảnh canny với thông số 250,255
    thresh = cv2.threshold(blurred, OTSU_THRESH, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]#Tạo ảnh đen và trắng
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)#Tìm contour từ ảnh edge
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]#Lọc ra top 10 contour
    
    ###############Lọc contour################
    
    plate_contour = [] #Danh sách các contour của tấm biển số
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
    
    #################Nhận diện biển số xe################
    
    if plate_contour is not None:
        cv2.drawContours(image, plate_contour, -1, (0, 255, 0), 2)#Vẽ đường viền xung quanh biển số vào ảnh gray, -1 cho biết vẽ tất cả các đường viền, 2 là độ dày
        mask = np.zeros(gray.shape, np.uint8)#Tạo ma trận mask với kích thước ảnh gray
        new_image = cv2.drawContours(mask, plate_contour, 0, 255, -1, )#Chỉ định vẽ contour đầu tiên, 0 là đầu tiên, -1  cho biết vẽ đầy đủ contour
        new_image = cv2.bitwise_and(gray, gray, mask=mask)#Tạo ảnh nhị phân chỉ chứa tấm biển số bằng phép toán AND bit
        #Tìm vị trí của biển số xe
        (x, y) = np.where(mask == 255)
        (topx, topy) = (np.min(x), np.min(y))
        (bottomx, bottomy) = (np.max(x), np.max(y))
        
        #topx trên, topy trái, botx dưới, boty phải
        #Cắt biển số trên từng trường hợp
        crop = image[topx:bottomx+3, topy+3:bottomy-3 ]
        new_image = new_image[topx:bottomx+3, topy+3:bottomy-3 ]
        gray_crop = blurred[topx:bottomx+3, topy+3:bottomy-3 ]
        thresh_crop =thresh[topx:bottomx+3, topy+3:bottomy-3 ]
        thresh_baw, blackandwhite = cv2.threshold(gray_crop, 85,255,cv2.THRESH_BINARY)
        inverted_crop = cv2.bitwise_not(gray_crop)
        inverted_crop = cv2.bitwise_not(inverted_crop)
        inverted_crop = cv2.equalizeHist(inverted_crop)
        
        #Đọc biển số xe
        case1 = pytesseract.image_to_string(blackandwhite, lang='eng', config='--psm 6')
        case2 = pytesseract.image_to_string(thresh_crop, lang='eng', config='--psm 6')
        case3 = pytesseract.image_to_string(gray_crop, lang='eng', config='--psm 6')
        case4 = pytesseract.image_to_string(crop, lang='eng', config='--psm 6')
        case5 = pytesseract.image_to_string(new_image, lang='eng', config='--psm 6')
        case6 = pytesseract.image_to_string(inverted_crop, lang='eng', config='--psm 6')
        
        
        flag = [False]
        place = None 
        
        # print("Case1 = ", case1)
        # print("Case2 = ", case2)
        # print("Case3 = ", case3)
        # print("Case4 = ", case4)
        # print("Case5 = ", case5)
        # print("Case6 = ", case6)
        
        strProcessing.str_processing(case2, flag, place)
        strProcessing.str_processing(case3, flag, place)
        strProcessing.str_processing(case1, flag, place)
        strProcessing.str_processing(case5, flag, place)
        strProcessing.str_processing(case4, flag, place)
        strProcessing.str_processing(case6, flag, place)
        
        # cv2.imshow('new_image', new_image)
        # cv2.imshow('Edged_crop', Edged_crop)
        # cv2.imshow('thresh_crop', thresh_crop)
        # cv2.imshow('inverted_crop', inverted_crop)
        # cv2.imshow('crop', crop)
        # cv2.imshow('blackandwhite', blackandwhite)
        # cv2.imshow('edged_crop', edged_crop)
        # cv2.imshow('gray_crop', gray_crop)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
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



def display_image(canvas, photo):
    # Hiển thị hình ảnh trên Canvas
    canvas.create_image(0, 0, image=photo, anchor=tk.NW)
    
canvas = tk.Canvas(win, width=500, height=300, bg="grey")
canvas.pack()
button.pack()


win.mainloop()
