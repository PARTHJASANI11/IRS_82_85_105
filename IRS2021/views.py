from django.shortcuts import render
import speech_recognition as sr
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import regex
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

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

    d = 'f** f* f* por** positive stay'
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
        print('Alr_adu: ', alr_adu)
        print(len(final))
        print(len(terms_list))
        p = (len(final) + alr_adu) / len(terms_list)
        print('adult content ratio after stop words removal is : ', p)
        p1 = (len(final) + alr_adu) / tot
        print('adult content ratio before stop words removal is : ', p1)
    else:
        print('Your voice not recognised')
    return render(request, 'text.html')

def speech1(request):
    return render(request, 'text.html')

def image1(request):
    return render(request, 'text.html')