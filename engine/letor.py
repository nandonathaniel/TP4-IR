import re
import random

import lightgbm as lgb
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from scipy.spatial.distance import cosine
from gensim.models import LsiModel
from gensim.corpora import Dictionary
import nltk

class LambdaMart:
    NUM_NEGATIVES = 1
    NUM_LATENT_TOPICS = 200

    def __init__(self):

        self.documents = {}
        self.queries = {}
        self.val_queries = {}

        self.q_docs_rel = {}
        self.val_q_docs_rel = {}

        self.group_qid_count = []
        self.val_group_qid_count = []

        self.dataset = []
        self.val_dataset = []

        self.dictionary = Dictionary()
        self.ranker = lgb.LGBMRanker(
            objective="lambdarank",
            boosting_type = "gbdt",
            n_estimators = 500,
            importance_type = "gain",
            metric = "ndcg",
            num_leaves = 50,
            learning_rate = 0.01,
            max_depth = -1,
        )

        nltk.download('punkt')
        self.stemmer = PorterStemmer()

        nltk.download('stopwords')
        self.stop_words_set = set(stopwords.words('english'))

        # train
        self.load_documents('engine/qrels-folder/train_docs.txt')
        self.load_queries('engine/qrels-folder/train_queries.txt')
        self.load_qrels('engine/qrels-folder/train_qrels.txt')
        self.construct_dataset()

        # validation
        self.load_val_queries('engine/qrels-folder/val_queries.txt')
        self.load_val_qrels('engine/qrels-folder/val_qrels.txt')
        self.construct_val_dataset()

        self.build_lsi_model()
        self.fit_dataset()
    
    def _preprocess_line(self, line: str):

        tokens = re.findall(r'\w+', line)

        stemmed_tokens = [
            self.stemmer.stem(token) if token else ''
            for token in tokens
        ]

        removed_stop_words = [
            token
            for token in stemmed_tokens
            if token not in self.stop_words_set
        ]
        
        return removed_stop_words

    def load_documents(self, file_path: str):
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file.readlines():
                doc_id, content = line.strip().split(' ', 1)
                self.documents[doc_id] = self._preprocess_line(content)
    
    def load_queries(self, file_path: str):
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file.readlines():
                q_id, content = line.strip().split(' ', 1)
                self.queries[q_id] = self._preprocess_line(content)
    
    def load_val_queries(self, file_path: str):
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file.readlines():
                q_id, content = line.strip().split(' ', 1)
                self.val_queries[q_id] = self._preprocess_line(content)
    
    def load_qrels(self, train_qrel_path: str):
        with open(train_qrel_path, 'r', encoding='utf-8') as file:
            for line in file.readlines():
                q_id, doc_id, rel = line.strip().split()
                if (q_id in self.queries) and (doc_id in self.documents):
                    if q_id not in self.q_docs_rel:
                        self.q_docs_rel[q_id] = []
                    self.q_docs_rel[q_id].append((doc_id, int(rel)))

    def load_val_qrels(self, val_qrels_path: str):
        with open(val_qrels_path, 'r', encoding='utf-8') as file:
            for line in file.readlines():
                q_id, doc_id, rel = line.strip().split()
                if (q_id in self.val_queries) and (doc_id in self.documents):
                    if q_id not in self.val_q_docs_rel:
                        self.val_q_docs_rel[q_id] = []
                    self.val_q_docs_rel[q_id].append((doc_id, int(rel)))
        
    def construct_dataset(self):
        for q_id in self.q_docs_rel:
            docs_rels = self.q_docs_rel[q_id]
            self.group_qid_count.append(len(docs_rels) + LambdaMart.NUM_NEGATIVES)
            for doc_id, rel in docs_rels:
                self.dataset.append((self.queries[q_id], self.documents[doc_id], rel))

            # tambahkan negative sebanyak NUM_NEGATIVES (random sampling saja dari documents)
            for _ in range(LambdaMart.NUM_NEGATIVES):
                self.dataset.append((self.queries[q_id], random.choice(list(self.documents.values())), 0))
    
    def construct_val_dataset(self):
        for q_id in self.val_q_docs_rel:
            docs_rels = self.val_q_docs_rel[q_id]
            self.val_group_qid_count.append(len(docs_rels) + LambdaMart.NUM_NEGATIVES)
            for doc_id, rel in docs_rels:
                self.val_dataset.append((self.val_queries[q_id], self.documents[doc_id], rel))
            
            # tambahkan negative sebanyak NUM_NEGATIVES (random sampling saja dari documents)
            for _ in range(LambdaMart.NUM_NEGATIVES):
                self.val_dataset.append((self.val_queries[q_id], random.choice(list(self.documents.values())), 0))
    
    def build_lsi_model(self):

        bow_corpus = [self.dictionary.doc2bow(doc, allow_update=True) for doc in self.documents.values()]
        self.model = LsiModel(bow_corpus, num_topics=LambdaMart.NUM_LATENT_TOPICS)

    def _vector_rep(self, text: str):
        rep = [topic_value for (_, topic_value) in self.model[self.dictionary.doc2bow(text)]]
        return rep if len(rep) == LambdaMart.NUM_LATENT_TOPICS else [0.] * LambdaMart.NUM_LATENT_TOPICS

    def features(self, query: list[str], doc: list[str]):

        v_q = self._vector_rep(query)
        v_d = self._vector_rep(doc)
        q = set(query)
        d = set(doc)
        cosine_dist = cosine(v_q, v_d)
        jaccard = len(q & d) / len(q | d)

        return v_q + v_d + [jaccard] + [cosine_dist]
    
    def fit_dataset(self):
        X = []
        Y = []
        for (query, doc, rel) in self.dataset:
            X.append(self.features(query, doc))
            Y.append(rel)
        
        # ubah X dan Y ke format numpy array
        X = np.array(X)
        Y = np.array(Y)

        # validation
        X_val = []
        Y_val = []
        for (query, doc, rel) in self.val_dataset:
            X_val.append(self.features(query, doc))
            Y_val.append(rel)
        
        X_val = np.array(X_val)
        Y_val = np.array(Y_val)

        self.ranker.fit(X, Y, group=self.group_qid_count, eval_set=[(X_val, Y_val)], eval_group=[self.val_group_qid_count], eval_metric='ndcg')
        print(self.ranker.best_score_)
    
    def predict(self, X: np.ndarray):
        return self.ranker.predict(X)
    
    def rerank_letor(self, query, doc_path):
        if not doc_path:
            return []
        
        X_unseen = []
        for doc in doc_path:
            with open(doc, 'r', encoding='utf-8') as f:
                X_unseen.append(self.features(self._preprocess_line(query), self._preprocess_line(f.readline())))

        X_unseen = np.array(X_unseen)
        scores = self.predict(X_unseen)

        did_scores = [x for x in zip(scores, doc_path)]
        sorted_did_scores = sorted(did_scores, key = lambda tup: tup[0], reverse = True)

        return sorted_did_scores