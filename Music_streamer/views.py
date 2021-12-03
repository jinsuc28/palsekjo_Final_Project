from django import template
from django.contrib.auth.decorators import login_required
from django.http import HttpResponse, HttpResponseRedirect
from django.template import loader
from django.urls import reverse
from django.shortcuts import render, redirect
from django.http import HttpResponse
from keras.preprocessing import image
import time
import pandas as pd

from webdriver_manager.chrome import ChromeDriverManager
from selenium import webdriver

from keras.models import model_from_json
from django.contrib.auth import authenticate, login
from .forms import LoginForm, SignUpForm
from mtcnn import MTCNN

import cv2
import numpy as np
from django.http import StreamingHttpResponse
from django.views.decorators import gzip
import threading

from Music_streamer.models import UserSong
from random import *
import pyglet


#################################

def home(request):

    Song_list = UserSong.objects.all()
    count = Song_list.count()-1
    index_num = randrange(count)

    youtubeUrl_2 = str(Song_list[int(index_num)].url)[32:]
    youtubeUrl_2 = f"https://www.youtube.com/embed/{youtubeUrl_2}"
    return render(request, "home/index.html", {'youtubeUrl_2': youtubeUrl_2})


class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        (self.grabbed, self.frame) = self.video.read()
        threading.Thread(target=self.update, args=()).start()

        self.mtcnn_detector = MTCNN()

    def __del__(self):
        self.video.release()

    def get_frame(self):
        img = self.frame

        detections = self.mtcnn_detector.detect_faces(img)
        # 얼굴 인식 -> detections 를 얻어내고
        # Song_list = pd.read_excel('./Song_list.xlsx')

        model = model_from_json(open("./model/facial_expression_model_structure.json", "r").read())
        model.load_weights('./model/facial_expression_model_weights.h5')  # load weights

        emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')

        # 인식된 얼굴 잘라서 감정 추출 -> cropped_image, emotions
        emotion_list = []

        for detection in detections:
            confidence_score = str(round(100 * detection["confidence"], 2)) + "%"
            x, y, w, h = detection["box"]

            cv2.putText(img, confidence_score, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 1)  # highlight detected face

            detected_face = img[int(y):int(y + h), int(x):int(x + w)]  # crop detected face
            detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY)  # transform to gray scale
            detected_face = cv2.resize(detected_face, (48, 48))  # resize to 48x48

            img_pixels = image.img_to_array(detected_face)
            img_pixels = np.expand_dims(img_pixels, axis=0)

            img_pixels /= 255  # pixels are in scale of [0, 255]. normalize all pixels in scale of [0, 1]

            # ------------------------------

            predictions = model.predict(img_pixels)  # store probabilities of 7 expressions
            max_index = np.argmax(predictions[0])

            # background of expression list
            overlay = img.copy()
            opacity = 0.4
            cv2.rectangle(img, (x + w + 10, y - 25), (x + w + 150, y + 115), (64, 64, 64), cv2.FILLED)
            cv2.addWeighted(overlay, opacity, img, 1 - opacity, 0, img)

            # connect face and expressions
            cv2.line(img, (int((x + x + w) / 2), y + 15), (x + w, y - 20), (255, 255, 255), 1)
            cv2.line(img, (x + w, y - 20), (x + w + 10, y - 20), (255, 255, 255), 1)

            # cropped_image 와 emotions를 이용해서 원본 image에 그리기 -> output_image
            for i in range(len(predictions[0])):
                emotion = "%s %s%s" % (emotions[i], round(predictions[0][i] * 100, 2), '%')
                j = np.argmax(predictions[0])
                k = emotions[j]
                emotion_list.append(k)
                """if i != max_index:
                    color = (255,0,0)"""

                color = (255, 255, 255)

                cv2.putText(img, emotion, (int(x + w + 15), int(y - 12 + i * 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            color, 1)

        _, jpeg = cv2.imencode('.jpg', img)
        return jpeg.tobytes()

    def update(self):
        while True:
            (self.grabbed, self.frame) = self.video.read()




def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@gzip.gzip_page
def detectme(request):
    try:
        cam = VideoCamera()
        return StreamingHttpResponse(gen(cam), content_type="multipart/x-mixed-replace;boundary=frame")
    except:  # This is bad! replace it with proper handling
        print("에러입니다...")
        pass


#################################

def login_view(request):
    form = LoginForm(request.POST or None)

    msg = None

    if request.method == "POST":

        if form.is_valid():
            username = form.cleaned_data.get("username")
            password = form.cleaned_data.get("password")
            user = authenticate(username=username, password=password)
            if user is not None:
                login(request, user)
                return redirect("/")
            else:
                msg = 'Invalid credentials'
        else:
            msg = 'Error validating the form'

    return render(request, "accounts/login.html", {"form": form, "msg": msg})


def register_user(request):
    msg = None
    success = False

    if request.method == "POST":
        form = SignUpForm(request.POST)
        if form.is_valid():
            form.save()
            username = form.cleaned_data.get("username")
            raw_password = form.cleaned_data.get("password1")
            user = authenticate(username=username, password=raw_password)

            msg = 'User created - please <a href="/login">login</a>.'
            success = True

            # return redirect("/login/")

        else:
            msg = 'Form is not valid'
    else:
        form = SignUpForm()

    return render(request, "accounts/register.html", {"form": form, "msg": msg, "success": success})


@login_required(login_url="/login/")
def index(request):
    context = {'segment': 'index'}

    html_template = loader.get_template('home/index.html')
    return HttpResponse(html_template.render(context, request))


@login_required(login_url="/login/")
def pages(request):
    context = {}
    # All resource paths end in .html.
    # Pick out the html file name from the url. And load that template.
    try:

        load_template = request.path.split('/')[-1]

        if load_template == 'admin':
            return HttpResponseRedirect(reverse('admin:index'))
        context['segment'] = load_template

        html_template = loader.get_template('home/' + load_template)
        return HttpResponse(html_template.render(context, request))

    except template.TemplateDoesNotExist:

        html_template = loader.get_template('home/page-404.html')
        return HttpResponse(html_template.render(context, request))

    except:
        html_template = loader.get_template('home/page-500.html')
        return HttpResponse(html_template.render(context, request))


def new(request):
    return render(request, 'Music_streamer/new.html')


def start_rc(request):
    context = {}
    return render(request, 'Music_streamer/start_rc.html', context)


##################### new 추가 admin 추가
def create(request):
    usersong = UserSong()  # 빈 객체 생성
    usersong.emotion = request.GET['emotion']
    usersong.song = request.GET['song']

    song_name = request.GET['song']

    ## Webdirver option 설정
    options = webdriver.ChromeOptions()
    options.add_argument('headless')  # 크롬 띄우기창 없애기
    driver = webdriver.Chrome(ChromeDriverManager().install(), chrome_options=options)
    time.sleep(2)
    youtubeUrl = (f"https://www.youtube.com/results?search_query={song_name}")
    driver.get(youtubeUrl)

    box_list = driver.find_elements_by_css_selector("#dismissible")
    title = box_list[0].find_element_by_css_selector('#video-title')
    youtubeUrl_2 = title.get_attribute("href")[32:]
    youtubeUrl_2 = f"https://www.youtube.com/embed/{youtubeUrl_2}"
    usersong.url = youtubeUrl_2
    usersong.save()
    time.sleep(2)

    return render(request, 'home/index.html', {'youtubeUrl_2': youtubeUrl_2})



def streamer(request):

    # Song_list = pd.read_excel('./Song_list.xlsx')

    # -----------------------------
    # opencv initialization

    mtcnn_detector = MTCNN()

    # -----------------------------
    # face expression recognizer initialization

    model = model_from_json(open("./model/facial_expression_model_structure.json", "r").read())
    model.load_weights('./model/facial_expression_model_weights.h5')  # load weights
    # -----------------------------

    emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')

    emotion_list = []

    cap = cv2.VideoCapture(0)  # process real time web-cam

    frame = 0



    while (True):
        ret, img = cap.read()

        original_size = img.shape

        cv2.putText(img, 'mtcnn', (70, 120), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)

        #img = dlib.load_rgb_image("img1.jpg")

        detections = mtcnn_detector.detect_faces(img)

        for detection in detections:
            confidence_score = str(round(100*detection["confidence"], 2))+"%"
            x, y, w, h = detection["box"]

            cv2.putText(img, confidence_score, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.rectangle(img, (x, y), (x+w, y+h),(255,255,255), 1) #highlight detected face

            detected_face = img[int(y):int(y + h), int(x):int(x + w)]  # crop detected face
            detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY)  # transform to gray scale
            detected_face = cv2.resize(detected_face, (48, 48))  # resize to 48x48

            img_pixels = image.img_to_array(detected_face)
            img_pixels = np.expand_dims(img_pixels, axis=0)

            img_pixels /= 255  # pixels are in scale of [0, 255]. normalize all pixels in scale of [0, 1]

            # ------------------------------

            predictions = model.predict(img_pixels)  # store probabilities of 7 expressions
            max_index = np.argmax(predictions[0])

            # background of expression list
            overlay = img.copy()
            opacity = 0.4
            cv2.rectangle(img, (x + w + 10, y - 25), (x + w + 150, y + 115), (64, 64, 64), cv2.FILLED)
            cv2.addWeighted(overlay, opacity, img, 1 - opacity, 0, img)

            # connect face and expressions
            cv2.line(img, (int((x + x + w) / 2), y + 15), (x + w, y - 20), (255, 255, 255), 1)
            cv2.line(img, (x + w, y - 20), (x + w + 10, y - 20), (255, 255, 255), 1)

            emotion = ""

            for i in range(len(predictions[0])):
                emotion = "%s %s%s" % (emotions[i], round(predictions[0][i] * 100, 2), '%')
                j = np.argmax(predictions[0])
                k = emotions[j]
                emotion_list.append(k)
                """if i != max_index:
                    color = (255,0,0)"""

                color = (255, 255, 255)

                cv2.putText(img, emotion, (int(x + w + 15), int(y - 12 + i * 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            color, 1)

                # -------------------------

        # cv2.imshow('img', img)


        frame = frame + 1
        # print(frame)

        # ---------------------------------
        # time.sleep(10)
        if frame > 20:
            break
        #
        # if cv2.waitKey(1) & 0xFF == ord('q'):  # press q to quit
        #     break

    # kill open cv things
    # cv2.imwrite('/static/assets/img/yourimage.jpg', img)

    cap.release()
    cv2.destroyAllWindows()

    if emotion_list ==[]:
        return render(request, 'Music_streamer/testcopy.html')

    else:
        b_emotion = pd.value_counts(emotion_list).head(1).index[0]
        Song_list = UserSong.objects.filter(emotion=f'{b_emotion}')
        count = Song_list.count()-1
        # index_num = randrange(count)

        # youtubeUrl = (f"https://www.youtube.com/results?search_query={str(Song_list[int(index_num)].song)}")
        # c_emotion = k
        # a = Song_list[c_emotion]
        # a = a.dropna()
        pyglet.options['search_local_libs'] = True
        for i in range(4):
            # name = Song_list[c_emotion][i]
            # media = pyglet.media.load(f"./music/{c_emotion}/{name}.mp3")
            index_num = randrange(count)
            media = pyglet.media.load(f"./music/{str(Song_list[index_num].song)}.mp3")

            player = media.play()
            time.sleep(10)

            mtcnn_detector = MTCNN()

            # -----------------------------
            # face expression recognizer initialization

            model = model_from_json(open("./model/facial_expression_model_structure.json", "r").read())
            model.load_weights('./model/facial_expression_model_weights.h5')  # load weights
            # -----------------------------

            emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')

            cap = cv2.VideoCapture(0)  # process real time web-cam

            frame = 0

            emotion_list = []

            while (True):
                ret, img = cap.read()

                original_size = img.shape

                cv2.putText(img, 'mtcnn', (70, 120), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)

                #img = dlib.load_rgb_image("img1.jpg")

                detections = mtcnn_detector.detect_faces(img)

                for detection in detections:
                    confidence_score = str(round(100*detection["confidence"], 2))+"%"
                    x, y, w, h = detection["box"]

                    cv2.putText(img, confidence_score, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                    cv2.rectangle(img, (x, y), (x+w, y+h),(255,255,255), 1) #highlight detected face

                    detected_face = img[int(y):int(y + h), int(x):int(x + w)]  # crop detected face
                    detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY)  # transform to gray scale
                    detected_face = cv2.resize(detected_face, (48, 48))  # resize to 48x48

                    img_pixels = image.img_to_array(detected_face)
                    img_pixels = np.expand_dims(img_pixels, axis=0)

                    img_pixels /= 255  # pixels are in scale of [0, 255]. normalize all pixels in scale of [0, 1]

                    # ------------------------------

                    predictions = model.predict(img_pixels)  # store probabilities of 7 expressions
                    max_index = np.argmax(predictions[0])

                    # background of expression list
                    overlay = img.copy()
                    opacity = 0.4
                    cv2.rectangle(img, (x + w + 10, y - 25), (x + w + 150, y + 115), (64, 64, 64), cv2.FILLED)
                    cv2.addWeighted(overlay, opacity, img, 1 - opacity, 0, img)

                    # connect face and expressions
                    cv2.line(img, (int((x + x + w) / 2), y + 15), (x + w, y - 20), (255, 255, 255), 1)
                    cv2.line(img, (x + w, y - 20), (x + w + 10, y - 20), (255, 255, 255), 1)

                    emotion = ""

                    for i in range(len(predictions[0])):
                        emotion = "%s %s%s" % (emotions[i], round(predictions[0][i] * 100, 2), '%')
                        j = np.argmax(predictions[0])
                        k = emotions[j]
                        emotion_list.append(k)
                        """if i != max_index:
                            color = (255,0,0)"""

                        color = (255, 255, 255)

                        cv2.putText(img, emotion, (int(x + w + 15), int(y - 12 + i * 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    color, 1)

                        # -------------------------

                # cv2.imshow('img', img)

                frame = frame + 1
                # print(frame)

                # ---------------------------------
                # time.sleep(10)
                if frame > 20:
                    break

                # if cv2.waitKey(1) & 0xFF == ord('q'):  # press q to quit
                #     break

            # kill open cv things
            cap.release()
            cv2.destroyAllWindows()

            if emotion_list ==[]:
                return render(request, 'Music_streamer/testcopy.html')

            k = pd.value_counts(emotion_list).head(1).index[0]
            if k == 'happy':
                # youtubeUrl_2 = driver.current_url[32:]
                return render(request, 'Music_streamer/test.html')
            elif k != 'happy':
                player.pause()
            elif i == 4:
                player.pause()

        return render(request, 'Music_streamer/create.html')

def Song_list(request):
    Song_lists = UserSong.objects.all().values()

    return render(request, 'Music_streamer/Song_list.html', {'Song_lists':Song_lists})