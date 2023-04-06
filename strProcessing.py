def str_processing(data, flag, place):
    #Đọc file biensoxe.txt để tiến hành lọc dữ liệu
    with open("C:/Users/locph/OneDrive/Desktop/biensoxe.txt", 'r', encoding='utf-8') as file:
        content = file.read().splitlines()
        
    #Lọc dữ liệu
    data = data.translate(str.maketrans("", "", " }{\/;°|-,()"))
    if len(data) > 0:
        if data[0]=='L':
            data = '4' + data[1:]
        elif data[0] == 'A':
            data = '4' + data[1:]
        for line in content:
                if line[:2]==data[:2] and len(data)>6:
                        if data[2] == '1':
                            data = data[:2] + 'T' + data[3:]
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
                        if data[3:].count("a") > 0:
                            data = data.replace("a", "8", 1)    
                        if data[2] == '1':
                            data = data[:3] + 'T' + data[4:]
                        elif data[2]=='¥':
                            data = data[:3] + 'V' + data[4:]
                        if data[3:].count("G") > 0:
                            data = data.replace("G", "6", 1)
                        if data[2].isupper() and not data[2].isdigit():
                            if not flag[0]:
                                print(data)
                                print('Biển số xe này của: ' + line[3:])
                                place=line[3:]
                                flag[0] = True
                        
    return flag,place