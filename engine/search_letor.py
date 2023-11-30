from bsbi import BSBIIndex
from compression import VBEPostings
from letor import LambdaMart
import time

class SearchLetor:
    def __init__(self):
        self.BSBI_instance = BSBIIndex(data_dir='collections',
                                postings_encoding=VBEPostings,
                                output_dir='index')
        self.BSBI_instance.load()
        self.letor = LambdaMart()
    
    def ranking(self, query, k=10):
        print("Query  : ", query)
        print("Results:")

        start_time = time.time()
        tf_idf_result = self.BSBI_instance.retrieve_tfidf(query, k=50)
        reranked_with_letor = self.letor.rerank_letor(query, [t[1] for t in tf_idf_result])
        for (score, doc) in reranked_with_letor[:k]:
            print(f"{doc:30} {score:>.3f}")
        print()
        end_time = time.time()

        elapsed_time = end_time - start_time
        elapsed_time_ms = elapsed_time * 1000
        print(f"Elapsed time: {elapsed_time_ms:>.3f} ms")

if __name__ == '__main__':
    sl = SearchLetor()
    while True:
        s = input()
        sl.ranking(s)