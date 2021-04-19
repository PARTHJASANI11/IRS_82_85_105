from django.shortcuts import render
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import regex
import numpy as np
import pandas as pd
import speech_recognition as sr
#from nudenet import NudeClassifier,NudeDetector
#import cv2
import math
import argparse
#import matplotlib.pyplot as plt
#import matplotlib.image as img


# Create your views here.

def home(request):
    return render(request, 'home.html')

def text(request):
    return render(request, 'text.html')

def speech(request):
    return render(request, 'speech.html')

def image(request):
    return render(request, 'image.html')

def text1(request):
    a = pd.read_csv('irs.csv')
    l = []

    for i in range(len(a['jigaboo'])):
        l.append(a['jigaboo'][i])

    d=''
    if request.method == 'POST':
        d = request.FILES['text_submit'].readlines()
    d=d[0].decode('utf-8')
    print(d)

    tokens = word_tokenize(d.lower())
    tot_len=len(tokens)
    print('After tokenization, the words are:')
    print()
    print(tokens)

    # List of stop words
    stop_words = stopwords.words('english')

    # Stop word removal step
    new_tokens = []
    for i in range(len(tokens)):
        if tokens[i] not in stop_words and tokens[i].isalnum():
            new_tokens.append(tokens[i])
    print('After stop word removal, the words are:')
    print()
    print(new_tokens)

    # Stemming of the words
    terms = []
    temp_terms = []
    stemmer = PorterStemmer()

    for i in new_tokens:
        terms.append(stemmer.stem(i))

    terms.sort()
    terms_list = terms

    for i in terms:
        if i not in temp_terms:
            temp_terms.append(i)

    print('Ater stemming, the words are:')
    print()
    # print(terms_list)

    # Stemming of the words
    terms = []
    temp_terms = []
    stemmer = PorterStemmer()

    for i in l:
        terms.append(stemmer.stem(i))

    terms.sort()
    terms_list_all = terms

    """for i in terms:
        if i not in temp_terms:
            temp_terms.append(i)"""

    print('Ater stemming, the words are:')
    print()
    # print(terms_list_all)

    final = []
    for i in terms_list:
        if (i in terms_list_all):
            final.append(i)

    # final

    print(len(final))
    print(len(terms_list))
    p = len(final) / tot_len

    return render(request,'text.html', {'probability':p,'words':tot_len,'adults':len(final)})


def speech1(request):
    data = request.POST.get('record')
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print('speak ')
        audio = r.listen(source)
        text = ''
        try:
            text = r.recognize_google(audio)
            print('You said:', text)
        except:
            print('Sorry could not recognize your voice')


    a = pd.read_csv('irs.csv')
    l = []

    for i in range(len(a['jigaboo'])):
        l.append(a['jigaboo'][i])

    d = text
    alr_adu = 0
    if (d != None):
        tokens = d.lower().split()
        tot = len(tokens)
        print('After tokenization, the words are:')
        print()
        print(tokens)

        new_tokens = []

        for i in tokens:
            if '*' in i:
                alr_adu = alr_adu + 1
            else:
                new_tokens.append(i)

        tokens = new_tokens

        # List of stop words
        stop_words = stopwords.words('english')

        # Stop word removal step
        new_tokens = []
        for i in range(len(tokens)):
            if tokens[i] not in stop_words and tokens[i].isalnum():
                new_tokens.append(tokens[i])
        print('After stop word removal, the words are:')
        print()
        print(new_tokens)

        # Stemming of the words
        terms = []
        temp_terms = []
        stemmer = PorterStemmer()

        for i in new_tokens:
            terms.append(stemmer.stem(i))

        terms.sort()
        terms_list = terms

        """for i in terms:
            if i not in temp_terms:
                temp_terms.append(i)"""

        print('Ater stemming, the words are:')
        print()
        # print(terms_list)

        # Stemming of the words
        terms = []
        temp_terms = []
        stemmer = PorterStemmer()

        for i in l:
            terms.append(stemmer.stem(i))

        terms.sort()
        terms_list_all = terms

        """for i in terms:
            if i not in temp_terms:
                temp_terms.append(i)"""

        print('Ater stemming, the words are:')
        print()
        # print(terms_list_all)

        final = []
        for i in terms_list:
            if (i in terms_list_all):
                final.append(i)

        # final
        adults=(len(final) + alr_adu)
        print('Alr_adu: ', alr_adu)
        print(len(final))
        print(len(terms_list))
        p = (len(final) + alr_adu) / len(terms_list)
        print('adult content ratio after stop words removal is : ', p)
        p1 = adults / tot
        print('adult content ratio before stop words removal is : ', p1)
    else:
        print('Your voice not recognised')
    return render(request, 'speech.html',{'probability':p1,'adults': adults , 'words':tot})

'''def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn=frame.copy()
    frameHeight=frameOpencvDnn.shape[0]
    frameWidth=frameOpencvDnn.shape[1]
    blob=cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections=net.forward()
    faceBoxes=[]
    for i in range(detections.shape[2]):
        confidence=detections[0,0,i,2]
        if confidence>conf_threshold:
            x1=int(detections[0,0,i,3]*frameWidth)
            y1=int(detections[0,0,i,4]*frameHeight)
            x2=int(detections[0,0,i,5]*frameWidth)
            y2=int(detections[0,0,i,6]*frameHeight)
            faceBoxes.append([x1,y1,x2,y2])
            #cv2.rectangle(frameOpencvDnn, (x1,y1), (x2,y2), (0,255,0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn,faceBoxes

def age_detect():
    parser=argparse.ArgumentParser()
    parser.add_argument('--image')

    args=parser.parse_args()

    faceProto="opencv_face_detector.pbtxt"
    faceModel="opencv_face_detector_uint8.pb"
    ageProto="age_deploy.prototxt"
    ageModel="age_net.caffemodel"
    genderProto="gender_deploy.prototxt"
    genderModel="gender_net.caffemodel"

    MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
    ageList=['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
    mapping={'(0-2)':0, '(4-6)':1, '(8-12)':2, '(15-20)':3, '(25-32)':4, '(38-43)':5, '(48-53)':6, '(60-100)':7}
    genderList=['Male','Female']

    faceNet=cv2.dnn.readNet(faceModel,faceProto)
    ageNet=cv2.dnn.readNet(ageProto,ageModel)
    genderNet=cv2.dnn.readNet(genderModel,genderProto)

    cam=cv2.VideoCapture(0)
    padding=20
    i=0
    hasFrame=''
    frame=''

    while True:
        hasFrame,frame=cam.read()
        if not hasFrame:
            cv2.waitKey()
            break
        break

    resultImg,faceBoxes=highlightFace(faceNet,frame)

    if not faceBoxes:
        print("No face detected")

    for faceBox in faceBoxes:
        face=frame[max(0,faceBox[1]-padding):
                   min(faceBox[3]+padding,frame.shape[0]-1),max(0,faceBox[0]-padding)
                   :min(faceBox[2]+padding, frame.shape[1]-1)]

        blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)
        genderPreds=genderNet.forward()
        gender=genderList[genderPreds[0].argmax()]
        print(f'Gender: {gender}')

        ageNet.setInput(blob)
        agePreds=ageNet.forward()
        age=ageList[agePreds[0].argmax()]
        print(f'Age: {age[1:-1]} years')

        #cv2.putText(resultImg, f'{gender}, {age}', (faceBox[0], faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)
        #cv2.imshow("Detecting age and gender", resultImg)
        #key = cv2.waitKey(1) & 0xFF
        # if the `q` key was pressed, break from the loop
        #if key == ord("q"):
        #   break

        if(mapping[age]<=3):
            return False
        else:
            return True
'''

def image1(request):

    if request.method == 'POST':
        d = request.FILES['text_submit'].readlines()
    '''img1 = 'girl2.jpeg'
    testImage = img.imread(img1)

    plt.imshow(testImage)
    # plt.show()

    classifier = NudeClassifier()

    # A. Classify single image
    di_out = classifier.classify(testImage)
    print(di_out)
    threshold = 0.5

    text=''
    if di_out[0]['unsafe'] < threshold:
        plt.imshow(testImage)
        # plt.show()
    else:
        age = age_detect()
        if age == True:
            #text='You can view the adult image'
            plt.imshow(testImage)
            # plt.show()
        else:
            text="Sorry! You can't view the adult image"'''
    return render(request,'image.html')
