from django import forms


class FeedbackForm(forms.Form):
    feedback = forms.CharField(max_length=300, widget=forms.Textarea)
