# ReviewsScoring
**ReviewsScoring** is a system that can score and filter reviews from booking-hotels/booking-flats services based on **ML** and **DL** approaches.
# How does it work?
Well, in this repository I have three types of models. First one is a Spam Filter that based on *TfIdf* and *Logistic Regression*. It's can help to choose the most relevant reviews. Second one is reviews scorer that based on *Convolutional 1D* and *Bidirectional GRU Neural Networks*. And the third one is a full system combined together - scorer and filter. Before processing by models texts must be preprocessed. For this task we use *NLTK* and *spacy*. Via *NLTK* we just get stopwords and via *spacy* we do main preprocessing (*lemmatization* and so on).
# Installation
After you clone this repository onto your PC go to directory of repository and run following command in terminal:

```pip install -r requirements.txt```

After all packages have installed you have to install language model for spaCy by the following command:

`python -m spacy download en_core_web_sm`

# Usage Examples
In repository you can find usage examples and models training. If you want to see how models was trained look at **reviews_scoring.ipynb** and **spam-filter-fitting.ipynb**, but if you want only look at usage examples see **review-example.ipynb** and **system_test.ipynb**.
