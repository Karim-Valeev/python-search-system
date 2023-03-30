from django.views import View
from django.http import HttpRequest
from django.shortcuts import render
from search.logic import search_files


class SearchView(View):

    def get(self, request: HttpRequest):
        query = request.GET.get("q", None)
        files = []
        if query:
            files = search_files(query)
        return render(
            request,
            "search/main.html",
            context={
                "files": files,
                "query": query if query else "",
            },
        )
