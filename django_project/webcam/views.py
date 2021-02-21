from django.shortcuts import render

import urllib.request
import cv2
import numpy as np
import time
import pytesseract
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import nltk
from nltk.corpus import stopwords
from expertai.nlapi.cloud.client import ExpertAiClient
import os
import threading
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import redirect
from django.http import JsonResponse
from django import forms
try:
    from PIL import Image
except ImportError:
    import Image

os.environ["EAI_USERNAME"] = ''
os.environ["EAI_PASSWORD"] = ''

client = ExpertAiClient()

stop_thread = False

model = load_model("model_lstm_new_ds_2.h5")

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

labels = ["joy", "sadness", "anger", "love", "surprise", "fear"]


def process(x):
    article = str(re.sub('[^a-zA-Z]', ' ', x))
    article = article.lower()
    article = article.split()
    article = [word for word in article if word not in set(
        stopwords.words('english'))]
    article = " ".join(article)
    return article


def get_sentiment_score(text):
    output = client.specific_resource_analysis(
        body={"document": {"text": text}},
        params={'language': 'en', 'resource': 'sentiment'})
    print(f"Sentiment score: {output.sentiment.overall}")
    return output.sentiment.overall


def predict(text):
    article = process(text)
    print(article)
    sentiment_score = get_sentiment_score(article)

    if sentiment_score > 50.0:
        category = "joy"
    elif sentiment_score < -50.0:
        category = "sadness"
    else:
        q = tokenizer.texts_to_sequences([article])
        q = pad_sequences(q, maxlen=35)
        output = model.predict(q)
        idx = np.argmax(output[0])
        category = labels[idx]
    return category


def home(request):
    global stop_thread
    stop_thread = True
    return render(request, 'home.html', {})


def cam(request):
    global stop_thread
    stop_thread = True
    return render(request, 'cam.html', {})


def background(url):
    org_time = time.time()
    while True:
        global stop_thread
        if stop_thread:
            break
        cur_time = time.time()
        if cur_time-org_time >= 5:
            print(stop_thread)
            org_time = cur_time
            imgResp = urllib.request.urlopen(url+'/shot.jpg')
            # print('imgResp1')
            # print(imgResp.read())
            # print(type(imgResp.read()))
            imgNp = np.array(bytearray(imgResp.read()), dtype=np.uint8)
            # print('imgNp')
            # print(type(imgNp))
            img = cv2.imdecode(imgNp, -1)
            # print('img')
            # print(type(img))
            config = ('-l eng --oem 1 --psm 3')
            text = pytesseract.image_to_string(img, config=config)
            emotion = predict(text)
            print(f"emotion : {emotion}")
            f = open("static/audiofy/output.txt", "w")
            f.write(text)
            f.close()
            f = open("static/audiofy/emotion.txt", "w")
            f.write(emotion)
            f.close()


def camVid(request):
    global stop_thread
    stop_thread = True
    url = request.POST['camurl']
    if url != '':
        url = 'http://' + request.POST['camurl']
        stop_thread = False
        thread = threading.Thread(target=background, args=(url,), kwargs={})
        thread.setDaemon(True)
        thread.start()
        return render(request, 'camera.html', {'camurl': url + '/jsfs.html'})
    return render(request, 'error.html')


def renderEbook(request):
    global stop_thread
    stop_thread = True
    return render(request, 'ebook.html')


@csrf_exempt
def screenShot(request):
    print('s4 response')
    if request.method == 'POST':
        myform = forms.Form(request.POST, request.FILES)
        if myform.is_valid():
            print(myform.cleaned_data.keys())
            print("abcd")
            print(myform.files['screenshot'])
            print(myform.files['screenshot'].read())
            # imgNp = np.fromstring(
            #     myform.files['screenshot'].read().decode(), np.uint8)
            # print('imgNp')
            # print(type(imgNp))
            # print(len(imgNp))
            # img = cv2.imdecode(imgNp, -1)
            # print('imgtest')
            # img = cv2.imdecode(np.frombuffer(
            #     myform.files['screenshot'].read(), dtype=np.uint8), cv2.IMREAD_COLOR)
            # print(type(img))
            config = ('-l eng --oem 1 --psm 3')
            text = pytesseract.image_to_string(Image.open(
                myform.files['screenshot']), config=config)
            emotion = predict(text)
            print(f"text : {text}")
            print(f"emotion : {emotion}")
            return JsonResponse({"success": True}, status=200)
