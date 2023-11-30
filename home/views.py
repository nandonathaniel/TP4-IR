from django.shortcuts import render
import os
import sys
from engine.search_letor import SearchLetor

class SearchLetorSingleton:
    _instance = None

    def __new__(cls):
        if not cls._instance:
            cls._instance = super(SearchLetorSingleton, cls).__new__(cls)
            cls._instance.sl = SearchLetor()
        return cls._instance

# Create your views here.
def index(request):
    return render(request, 'index.html')

def search(request):
    if request.method == 'POST':
        engine_path = os.path.abspath('./engine/')
        sys.path.append(engine_path)
        search = request.POST['search']

        # Use the SearchLetor instance from the singleton
        sl_singleton = SearchLetorSingleton()
        context = sl_singleton.sl.rankingReturn(search)
        context["query"] = search

        return render(request, 'search.html', context)

    else:
        return render(request, 'search.html')