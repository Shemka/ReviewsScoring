from scripts.sfilter import SFilterTransformer
from scripts.review_scoring import ReviewScorer

class SystemScorer:
    def __init__(self, sfilter_path, scorer_path, tokenizer_path):
        self.scorer = ReviewScorer(scorer_path, tokenizer_path)
        self.sfilter = SFilterTransformer(sfilter_path)
    
    # Delete spam reviews and score non-spam
    def delete_predict(self, texts):
        print('Count of texts before filtering:', len(texts))
        texts = self.sfilter.transform(texts)
        print('Count of texts after filtering:', len(texts))
        scores = self.scorer.predict(texts)
        return scores