import cv2
import datetime
import csv
from collections import Counter
tm = datetime.datetime.now()
date = tm.strftime("%a") + tm.strftime("%d") + tm.strftime("%b")


def column(matrix, i):
    return [row[i] for row in matrix]

def writecsv(data):
    with open("report_csv/"+date + '.csv', mode='a') as (employee_file):
        employee_writer = csv.writer(employee_file, lineterminator='\n')
        employee_writer.writerow(data)


def writenewcsv():
    with open("report_csv/"+date + '.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["TotalNotSafe", "Tanggal", "Kamera", "Foto"])
        writer.writerow(["0", "0", "0", "0"])


def readcsv():
    linecount = 0
    x = []
    try:
        with open("report_csv/"+date + '.csv', 'r') as readFile:
            reader = csv.reader(readFile, lineterminator='\n')
            lines = list(reader)
            x = column(lines,0)
            return x
    except:
        writenewcsv()

        with open("report_csv/"+date + '.csv', 'r') as readFile:
            reader = csv.reader(readFile, lineterminator='\n')
            lines = list(reader)
            x = column(lines,0)
            return x

font = cv2.FONT_HERSHEY_SIMPLEX

def delid(listid,val):
    for i in listid[:]:
        if i == val:
            listid.remove(val)

def visual(kordinat,img,threshold,nframe,wt,path,A):
    coorworker = []
    coorhelm = []
    coorvest = []
    rect = []
    tem = Counter()
    timestamp = datetime.datetime.now()
    for box in kordinat:
        xmin = box[0]
        xmax = box[1]
        ymin = box[2]
        ymax = box[3]
        st = box[4]
        typ = st[0]

        if typ == 'person':
            coorworker.append([xmin,xmax,ymin,ymax])
        elif typ == "helm":
            coorhelm.append([xmin,xmax,ymin,ymax])
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 255, 255), 1)
        else:
            coorvest.append([xmin,xmax,ymin,ymax])
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 255, 255), 1)

    for coor in coorworker:
        flaghelm = 0
        flagvest = 0
        for box in coorhelm:
            if (coor[0] - 15 < box[0] and coor[1] + 15 > box[1]):
                flaghelm = 1
        for ind in coorvest:
            if (coor[0] - 15 < ind[0] and coor[1] + 15 > ind[1]):
                flagvest = 1

        if flaghelm == 1 and flagvest == 1:
            pass
        elif flaghelm == 1 and flagvest == 0:
            pass
        elif flaghelm == 0 and flagvest == 1:
            boxx = (coor[0], coor[2], coor[1], coor[3])
            rect.append(boxx)
        else:
            boxx = (coor[0], coor[2], coor[1], coor[3])
            rect.append(boxx)
        color = [0, 255, 0]
        cv2.rectangle(img, (coor[0], coor[2]), (coor[1], coor[3]), color, 1)
        cv2.putText(img, "worker ", (coor[0], coor[2] - 5), font, 0.5, color, 2)

        objects = wt.update(rect)
        for (objectID, centroid) in objects.items():
            A.append(objectID)


    tem.update(A)

    b = list(tem.values())
    c = list(tem.keys())

    for i in range (len(b)):
        try:
            if b[i] > threshold:
                #print(c[i],"values = ",b[i])
                lastid = readcsv()
                last = c[i]
                print(str(last)+"."+str(nframe))
                if str(last)+"."+str(nframe) not in str(lastid):
                    timenow = timestamp.strftime("%H") + "-" + timestamp.strftime("%M") + "-" + timestamp.strftime("%S")
                    idfoto = str(last) + "-" + timenow + "-" + str(i) + '.jpg'
                    cv2.imwrite("report_foto/" + idfoto, img)
                    cwd = path + idfoto
                    data = [str(last)+'.'+str(nframe), timestamp.strftime("%a") + timestamp.strftime("%d") + timestamp.strftime("%b"),nframe, cwd]
                    writecsv(data)
                delid(A,c[i])
        except:
            pass

