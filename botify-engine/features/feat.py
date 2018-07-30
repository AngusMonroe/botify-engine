
import pickle

class feat_extractor(object):

    def __init__(self):
        self.feats = []

    def add(self, feat):
        self.feats.append(feat)
        return self

    def sanitize(self):
        for feat in self.feats:
            feat.sanitize()

    def unsanitize(self):
        for feat in self.feats:
            feat.unsanitize()
