from engine.bsbi import BSBIIndex
from engine.compression import VBEPostings
from engine.letor import LambdaMart
import time

class SearchLetor:
    def __init__(self):
        self.BSBI_instance = BSBIIndex(data_dir='engine/collections',
                                postings_encoding=VBEPostings,
                                output_dir='engine/index')
        self.BSBI_instance.load()
        # self.letor = LambdaMart()
    
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

    def rankingReturn(self, query, k=10):
        start_time = time.time()
        reranked_with_letor = self.BSBI_instance.retrieve_tfidf(query, k=20)
        # reranked_with_letor = self.letor.rerank_letor(query, [t[1] for t in tf_idf_result])

        results_list = []
        for (score, doc) in reranked_with_letor[:k]:
            results_list.append({"doc": doc, "score": score})

        end_time = time.time()
        elapsed_time = end_time - start_time
        elapsed_time_ms = elapsed_time * 1000

        return {"time": elapsed_time_ms, "results": results_list}



if __name__ == '__main__':
    sl = SearchLetor()
    while True:
        s = input()
        sl.ranking(s)