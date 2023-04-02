from django.views import View
from django.http import HttpRequest
from django.shortcuts import render
from search.logic import search_file_links


class SearchView(View):

    def get(self, request: HttpRequest):
        query = request.GET.get("search", None)
        links = []
        if query:
            links = search_file_links(query)
        return render(
            request,
            "search/main.html",
            context={
                "links": links,
                "search": query if query else "",
            },
        )
