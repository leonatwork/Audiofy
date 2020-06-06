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
from nltk.stem.porter import PorterStemmer

model = load_model("model_lstm_new_ds_2.h5")

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

ps = PorterStemmer()
labels = ["joy","sadness","anger","love","surprise","fear"]


def process(x):
    article = str(re.sub('[^a-zA-Z]',' ',x))
    article = article.lower()
    article = article.split()
    article = [ ps.stem(word) for word in article if word not in set(stopwords.words('english'))]
    article = " ".join(article)
    return article


def predict(text):
    article = process(text)
    q = tokenizer.texts_to_sequences([article])
    q = pad_sequences(q, maxlen=35)
    output = model.predict(q)
    idx = np.argmax(output[0])
    category = labels[idx]
    return category


def home(request):
    return render(request, 'home.html', {})


def background(url):
    org_time = time.time()
    while True:
        cur_time = time.time()
        if cur_time-org_time >= 5:
            org_time = cur_time
            imgResp=urllib.request.urlopen(url+'/shot.jpg')
            imgNp=np.array(bytearray(imgResp.read()),dtype=np.uint8)
            img=cv2.imdecode(imgNp,-1)
            config = ('-l eng --oem 1 --psm 3')
            text = pytesseract.image_to_string(img, config=config)
            emotion = predict(text)
            f = open("static/audiofy/output.txt", "w")
            f.write(text)
            f.close()
            f = open("static/audiofy/emotion.txt", "w")
            f.write(emotion)
            f.close()


def camVid(request):
    url = 'http://' + request.POST['camurl']
    import threading
    t = threading.Thread(target=background, args=(url,), kwargs={})
    t.setDaemon(True)
    t.start()
    return render(request, 'camera.html', {'camurl': url + '/jsfs.html'})

