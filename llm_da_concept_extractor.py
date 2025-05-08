import MeCab
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import dill
from llm_concept_extractor import get_weather_info_from_utterance
# from crf_util import word2features, sent2features, sent2labels
import re
import ipadic

# 発話文から対話行為タイプとコンセプトを抽出するクラス
class DA_Concept:

    def __init__(self):
        # MeCabの初期化
        self.mecab = MeCab.Tagger(ipadic.MECAB_ARGS)
        self.mecab.parse('')

        # SVMモデルの読み込み
        with open("svc.model","rb") as f:
            self.vectorizer = dill.load(f)
            self.label_encoder = dill.load(f)
            self.svc = dill.load(f)

    # 発話文から対話行為タイプをコンセプトを抽出
    def process(self,utt):
        lis = []
        for line in self.mecab.parse(utt).splitlines():
            if line == "EOS":
                break
            else:
                word, feature_str = line.split("\t")
                features = feature_str.split(',')
                postag = features[0]
                lis.append([word, postag, "O"])

        words = [x[0] for x in lis]
        tokens_str = " ".join(words)
        X = self.vectorizer.transform([tokens_str])
        Y = self.svc.predict(X)
        # 数値を対応するラベルに戻す
        da = self.label_encoder.inverse_transform(Y)[0]

      
        conceptdic = get_weather_info_from_utterance(utt)

        return da, conceptdic

if __name__ ==  '__main__':
    da_concept = DA_Concept()
    da, conceptdic = da_concept.process("東京の天気は？")
    print(da, conceptdic)
