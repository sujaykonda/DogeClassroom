from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect
import pathlib
import pandas as pd
import os
from .forms import FeedbackForm

parent_dir = str(pathlib.Path().parent.absolute())
f = open(parent_dir + "/model/tokenizer.json", "r")
tokenizer = tokenizer_from_json(f.read())
f.close()

#model = load_model(str(pathlib.Path().parent.absolute()) + "/model/model.h5")


def feedback(request):
    # if this is a POST request we need to process the form data
    if request.method == 'POST':
        # create a form instance and populate it with data from the request:
        form = FeedbackForm(request.POST)
        # check whether it's valid:
        if form.is_valid():
            fb = form.data['feedback']
            sequences = tokenizer.texts_to_sequences([fb])
            padded_sequences = pad_sequences(sequences, maxlen=100)

            rating = model.predict(padded_sequences)[0][0]

            data = pd.read_csv(parent_dir + "/data/data.csv", index_col="Id")
            print(data)
            data = data.append(pd.DataFrame({"Feedback": [fb], "Rating": [rating]}), ignore_index=True)
            print(data)
            data.to_csv(parent_dir + "/data/data.csv", index=True, index_label="Id")
            # process the data in form.cleaned_data as required
            # ...
            # redirect to a new URL:
            return HttpResponseRedirect('/student/thanks/')

    # if a GET (or any other method) we'll create a blank form
    else:
        form = FeedbackForm()

    return render(request, 'student/index.html', {'form': form})


def thanks(request):
    return render(request, 'student/thanks.html')

