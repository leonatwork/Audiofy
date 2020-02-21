from django.shortcuts import render

import urllib.request
import cv2
import numpy as np
import time
import pytesseract

def home(request):
    return render(request, 'home.html', {})


def background():
    org_time = time.time()
    while True:
        imgResp=urllib.request.urlopen('http://192.168.43.243:8080/shot.jpg')
        imgNp=np.array(bytearray(imgResp.read()),dtype=np.uint8)
        img=cv2.imdecode(imgNp,-1)
        cur_time = time.time()
        if cur_time-org_time >= 5:
            org_time = cur_time
            print('\n****************** Pre-processing image ******************\n')
            config = ('-l eng --oem 1 --psm 3')
            text = pytesseract.image_to_string(img, config=config)
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            analyzer = SentimentIntensityAnalyzer()
            vs = analyzer.polarity_scores(text)
            if(vs['compound']>=0.5):
                emotion = 'happy'
                tag = '&#128522;'
            elif(vs['compound']>-0.5):
                emotion = 'neutral'
                tag = '&#128528;'
            else:
                emotion = 'sad'
                tag = '&#128532;'
            print(emotion)
            f = open("static/audiofy/emot.html", "w")
            f.write('<div style="margin-top: -12px;margin-bottom: 0px;padding: 0px;font-size:100px">'+tag+'</div>')
            f.close()

            f = open("static/audiofy/extractedEmotion.txt", "w")
            f.write(emotion)
            f.close()

            f = open("static/audiofy/textext.html", "w")
            f.write("<pre>"+text+"</pre>")
            f.close()

def camVid(request):
    url = 'http://' + request.POST['camurl']
    import threading
    t = threading.Thread(target=background, args=(), kwargs={})
    t.setDaemon(True)
    t.start()
    return render(request, 'camera.html', {'camurl': url + '/jsfs.html'})
