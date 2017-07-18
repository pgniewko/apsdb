import nltk
from newspaper import Article
from .utils import remove_non_ascii


class Extractor(object):
    def __init__(self, text=None, url=None):
        if not text and not url:
            raise Exception('text or url is required')

        self.text = text
        self.url = url
        self.places = []
    
    def set_text(self):
        if not self.text and self.url:
            a = Article(self.url)
            a.download()
            a.parse()
            self.text = a.text


    def find_entities(self):
        self.set_text()

        text = nltk.word_tokenize(remove_non_ascii(self.text))
        nes = nltk.ne_chunk(nltk.pos_tag(text))

        for ne in nes:
            if len(ne) == 1:
#                if (ne.label() == 'GPE' or ne.label() == 'PERSON') and ne[0][1] == 'NNP': # PG, 06/11/17; this version is for Linux
                if (ne.node == 'GPE' or ne.node == 'PERSON') and ne[0][1] == 'NNP':    # this version is for Mac OS
                    self.places.append(ne[0][0])
