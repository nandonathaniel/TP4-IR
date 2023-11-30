from django.http import HttpResponse
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

        for kamus in context['results']:
            file_path = os.path.abspath('./' + kamus['doc'])
            folder_id, file_name = file_path.split(os.sep)[-2:]
            kamus['folder_id'] = folder_id
            kamus['file_name'] = file_name
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
                if len(text) > 500:
                    text = text[:500] + ' ...'
                kamus['text'] = text

        return render(request, 'search.html', context)

    else:
        return render(request, 'search.html')

def get_file(request, folder_id, file_name):
    file_path = os.path.abspath('./engine/collections/' + f'{folder_id}/{file_name}')
    with open(file_path, 'r', encoding='utf-8') as file:
        return HttpResponse(file.read(), content_type='text/plain')
    