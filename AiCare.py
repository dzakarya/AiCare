from flask import Flask, render_template, flash, redirect, url_for, session, logging, request, Response, g
import cv2
import os
import sys
from tensorflow.keras.models import load_model
import tensorflow as tf
from apdlib.apddetector import classifier
from functools import wraps
import mysql.connector as sqlconnect
import numpy as np
import imutils
from imutils.video import FileVideoStream,VideoStream
import threading
import time
import datetime
from collections import OrderedDict
import imgconvert

import glob
import os
from flask_mysqldb import MySQL
sys.path.append('..')
cd = os.getcwd()

CWD_PATH = os.getcwd()




app = Flask(__name__)
"""
app.config['MYSQL_HOST'] = 'mydb.cthd89lrpxfp.ap-southeast-1.rds.amazonaws.com'
app.config['MYSQL_USER'] = 'jone'
app.config['MYSQL_PASSWORD'] = 'jonetechnology'
app.config['MYSQL_DB'] = 'persondb'
mysql = MySQL(app)

connection = sqlconnect.connect(host='mydb.cthd89lrpxfp.ap-southeast-1.rds.amazonaws.com',
                                database='persondb',
                                user='jone',
                                password='jonetechnology')
"""
app.config['MYSQL_HOST'] = '127.0.0.1'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'jonetechnology'
app.config['MYSQL_DB'] = 'persondb'
mysql = MySQL(app)

connection = sqlconnect.connect(host='127.0.0.1',
                                database='persondb',
                                user='root',
                                password='jonetechnology')


def load_face(num_faces):
    label = []
    face = []
    X = []
    Y = []
    cursor = mysql.connection.cursor()
    sql_fetch_blob_query = "SELECT id, face1, face2, face3, face4, face5, " \
                           "face6, face7, face8, face9, face10 from people"

    cursor.execute(sql_fetch_blob_query)
    record = cursor.fetchall()
    for row in record:
        for i in range(num_faces):
            label.append(row[0])
            img = imgconvert.BinToImg(row[i + 1])
            face.append(img)
    print("loaded face")
    cursor.close()
    return face, label


def init():
    global lock,outputFrame,stat,mskcls,url_cam,stop_event,stop_event_regis,num_faces
    num_faces = 10
    tolerance = 1
    stop_event = threading.Event()
    stop_event_regis = threading.Event()
    outputFrame = [None,None,None,None]
    lock = threading.Lock()
    url_cam = ['abi1.MOV','abi3.MOV']
    clasmodel = load_model("resnetapd.h5")
    mskcls = classifier(clasmodel)

def init_classfier():
    global mskcls
    #facetrainx,facetrainy = load_face(10)
   #mskcls.recognizer.train(facetrainx,facetrainy)

def store_detected(object_worker):
    cur = connection.cursor()

    for x in range(len(object_worker)):
        num_notsafe = object_worker[x]['id_worker']
        img = object_worker[x]['image_worker']
        date = object_worker[x]['date']
        ts = object_worker[x]['time_stamp']
        querys = "INSERT INTO apd_table(num_notsafe, img, date, ts)" \
             " VALUES (%s,%s,%s,%s)"
        cur.execute(querys, (num_notsafe, img, date, ts))
    connection.commit()
    cur.close()

def get_info(idvisitor):
    cur = connection.cursor()
    querys = ("SELECT name, gender, age from people where id = %s")

    try:
        cur.execute(querys, (idvisitor,))
        record = cur.fetchall()
        for result in record:
            name = result[0]
            gender = result[1]
            age = result[2]
        cur.close()
        return name,gender,age
    except:
        print("failed to connect database")
        return "","",""

def create_blank(width, height, rgb_color=(0, 0, 0)):
    """Create new image(numpy array) filled with certain color in RGB"""
    # Create black blank image
    image = np.zeros((height, width, 3), np.uint8)

    # Since OpenCV uses BGR, convert the color first
    color = tuple(reversed(rgb_color))
    # Fill image with color
    image[:] = color

    return image


"""def regis(idcam,stop_event):
    global outputFrame, lock
    vs = VideoStream(url_cam[idcam]).start()
    while not stop_event.is_set():
        images = vs.read()
        faceimage = mskcls.regis_face(images)
        outputFrame[idcam+1] = images
        outputFrame[idcam+2] = faceimage"""



def runclassifier(idcam,stop_event):
    global outputFrame, lock,stat,mskcls,url_cam
    time = datetime.datetime.now()
    lasttime = time.strftime("%H")
    vs = FileVideoStream(url_cam[idcam]).start()
    while not stop_event.is_set():
        image = vs.read()
        mskcls.procbody(image)
        time = datetime.datetime.now()
        cttime = time.strftime("%H")
        if cttime != lasttime:
            store_detected(mskcls.tempworker)
            mskcls.tempworker.clear()
            lasttime = cttime
        outputFrame[idcam] = image




@app.route('/')
def index():
    return render_template('index.html')




@app.route('/report', methods=['GET', 'POST'])
def report(chart1ID = 'chart1_ID', chart1_type = 'line', chart1_height = 500,
    chart2ID = 'chart2_ID', chart2_type = 'pie', chart2_height = 500):
    path = os.path.join(CWD_PATH, 'static', 'temp_img')
    try:
        for file in os.listdir(path + '/'):
            os.remove(path + '/' + file)
    except:
        print(" ")
    timestamp = datetime.datetime.now()
    timed = []
    date = []
    csvdata = []
    label1 = []
    dataset = []
    dataset2 = []
    cur = mysql.connection.cursor()
    sql_fetch_blob_query = "SELECT num_notsafe,date,ts,img from apd_table where date = %s"

    if request.args.get("format") == 'perweek':
        for i in range(28):
            timed.append(timestamp - datetime.timedelta(days=i))
            date.append((timed[i].strftime("%Y-%m-%d")))
            cur.execute(sql_fetch_blob_query, (date[i],))
            record = cur.fetchall()
            try:
                num_notsafe = int(record[-1][0])
            except:
                num_notsafe = 0
            csvdata.append(num_notsafe)
        label1=['four_weeks_ago','three_weeks_ago','two_weeks_ago','this_week']

        for i in range(4):
            dataset.append(sum(csvdata[(3-i)*7:(3-i)*7+7]))
            if sum(dataset) == 0:
                total_data = 1
            else:
                total_data = sum(dataset)
            piedata = {
                'name': label1[i] + " " + str(dataset[i] * 100 / total_data) + "%",
                'y': dataset[i] * 100 / total_data
            }
            dataset2.append(piedata)

    else:
        for i in range(7):
            timed.append(timestamp - datetime.timedelta(days=i))
            date.append((timed[i].strftime("%Y-%m-%d")))
            cur.execute(sql_fetch_blob_query, (date[i],))
            record = cur.fetchall()
            try:
                num_notsafe = int(record[-1][0])
            except:
                num_notsafe = 0
            csvdata.append(num_notsafe)

        for i in range(7):
            dataset.append(csvdata[6 - i])
            label1.append(date[6 - i])
            if sum(csvdata) == 0:
                total_data = 1
            else:
                total_data = sum(csvdata)
            piedata = {
                'name': date[6 - i] + " " + str(csvdata[6 - i] * 100 / total_data) + "%",
                'y': csvdata[6 - i] * 100 / total_data
            }
            dataset2.append(piedata)

    data_event = []
    if request.method == 'POST':
        data_event.clear()
        try:
            cur.execute(sql_fetch_blob_query, (request.form['date_show'],))
            record = cur.fetchall()
            for x in record:
                event = {
                    'event_id': x[0],
                    'date': x[1],
                    'time': x[2],
                    'img': "temp_img/" + str(x[0]) +str(x[1]) + ".jpg"
                }
                cv2.imwrite("static/temp_img/" + str(x[0]) +str(x[1])  + ".jpg",
                            cv2.resize(imgconvert.BinToImg(x[3]), (200, 200)))
                data_event.append(event)

            print(request.form['date_show'])
        except Exception as e:
            print(e)


    mysql.connection.commit()
    cur.close()
    pageType = 'graph'
    chart1 = {"renderTo": chart1ID, "type": chart1_type, "height": chart1_height, }
    series1 = [{"name": "Number of notsafe Status","data": dataset}]
    title1 = {"text": 'Worker Foul Detected'}
    xAxis1 = {"categories": label1}
    yAxis1 = {"title": {"text": 'Total Number'}}

    chart2 = {"renderTo": chart2ID, "type": chart2_type, "height": chart2_height, }
    series2 = [{"name": 'notsafe status in %', "data":dataset2}]
    title2 = {"text": 'Worker Foul Distribution in Percentage'}
    xAxis2 = {"categories": ['xAxis Data1', 'xAxis Data2', 'xAxis Data3']}
    yAxis2 = {"title": {"text": 'yAxis Label'}}
    return render_template('/report.html', chart1ID=chart1ID, chart1=chart1, series1=series1, title1=title1,
                           xAxis1=xAxis1, yAxis1=yAxis1, chart2ID=chart2ID, chart2=chart2, series2=series2,
                           title2=title2, xAxis2=xAxis2, yAxis2=yAxis2,data_event=data_event)

@app.route('/login', methods=['GET', 'POST'])
def login():
    stop_event_regis.set()
    error = None
    if request.method == 'POST':
        if request.form['username'] != 'admin' or request.form['password'] !='admin':
            error = 'Wrong Username or Password'
        else:
            session['logged_in'] = True
            return redirect(url_for('streamvid'))
    return render_template('login.html', error=error)

def is_logged_in(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        if 'logged_in' in session:
            return f(*args, **kwargs)
        else:
            flash('Unauthorized, Please login', 'danger')
            return redirect(url_for('login'))
    return wrap


@app.route('/setting', methods=['GET', 'POST'])
def setting():
    cur = mysql.connection.cursor()
    querys = 'UPDATE parametertb SET face_tolerance = %s,face_threshold = %s,mask_threshold = %s WHERE id=1'
    if request.method == 'POST':
        f_tolerance = request.form['tolerance']
        f_threshold = request.form['frthreshold']
        m_threshold = request.form['mdthreshold']
        cur.execute(querys, (f_tolerance, f_threshold, m_threshold))
        mysql.connection.commit()
        cur.close()
        return redirect(url_for('login'))
    return render_template('setting.html')

@app.route('/logout')
def logout():
    session.clear()
    stop_event.set()
    flash('Your now Logged Out', 'success')
    return redirect(url_for('login'))

def gen(cam):
    """Video streaming generator function."""
    global outputFrame, lock
    # loop over frames from the output stream
    while True:
        # wait until the lock is acquired
        if outputFrame[cam] is None:
            continue

        # encode the frame in JPEG format
        (flag, encodedImage) = cv2.imencode(".jpg", outputFrame[cam])

        # ensure the frame was successfully encoded
        if not flag:
            continue

        # yield the output frame in the byte format
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
               bytearray(encodedImage) + b'\r\n')

@app.route('/aicare')
@is_logged_in
def streamvid():
    init_classfier()
    return render_template('streamvid.html')

@app.route('/video_feed')
@is_logged_in
def video_feed():
    stop_event.clear()
    t = threading.Thread(target=runclassifier,args=(0,stop_event))
    t.daemon = True
    t.start()
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(0),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/cam2')
@is_logged_in
def cam2():
    stop_event.clear()
    t = threading.Thread(target=runclassifier,args=(1,stop_event))
    t.daemon = True
    t.start()
    return Response(gen(1),mimetype='multipart/x-mixed-replace; boundary=frame')

"""
@app.route('/regisface')
@is_logged_in
def regisface():
    stop_event_regis.clear()
    t = threading.Thread(target=regis,args=(0,stop_event_regis))
    t.daemon = True
    t.start()
    return Response(gen(1),mimetype='multipart/x-mixed-replace; boundary=frame')"""

if __name__ == "__main__":
    # start the flask app

    init()
    app.secret_key='secret123'
    app.run(host='0.0.0.0', debug=False,port=5000,
            threaded=True, use_reloader=False)
