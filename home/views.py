from django.shortcuts import render
import requests

from engine.search_letor import SearchLetor
# Create your views here.
def index(request):
    return render(request, 'index.html')

def search(request):
    if request.method == 'POST':
        search = request.POST['search']

        sl = SearchLetor()
        context = sl.rankingReturn(search)

        return render(request, 'search.html', context)

    else:
        return render(request, 'search.html')