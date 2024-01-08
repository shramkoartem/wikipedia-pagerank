"""
Module containing Crawler classes for extracting links structure
of pages within selected wikipedia category
"""

import json
from abc import abstractmethod, ABC
from typing import List, Dict
from dataclasses import dataclass

import requests
import numpy as np
from bs4 import BeautifulSoup
from bs4 import BeautifulSoup as bs
from tqdm import tqdm


@dataclass
class Link:
    """
    Dataclass for link
    """
    title: str
    url: str

class Crawler(ABC):
    """
    Crawler base class
    """
    def __init__(
            self,
            root: str,
            queue: List[str],
            index: Dict[str, str],
            adj_list: Dict[str, List[str]]
        ):

        self.url_base: str = "https://de.wikipedia.org/"
        self.root: str = root

        # Tracking pages to visit
        self.queue: List[str] = queue

        # Link - Title page index
        self.index: Dict[str, str] = index

        # Links graph adjacency list
        self.adj_list: Dict[str, List[str]] = adj_list
        if not adj_list.get("root"):
            self.adj_list[root] = []

    @staticmethod
    def get_soup(url: str) -> BeautifulSoup:
        """
        Utility function to get page urls
        """
        res = requests.get(url)
        html = res.text
        soup = bs(html)
        return soup
    
    @abstractmethod
    def run(self) -> None:
        """
        Execute crawling job
        """


class PageCrawler(Crawler):
    """
    Specialised crawler for collecting links
    within wikipedia pages
    """

    def run(self) -> None:
        """
        1. Get all links of the page
        2. Filter for links that are indexed
        3. Update the adjecency list
        """

        # Get page source
        url: str = self.url_base + self.root
        soup: BeautifulSoup = self.get_soup(url)

        # Grab all links
        links = soup.find_all(name="a")

        # Filter for links that are in index
        urls = [link.attrs.get("href") for link in links
                 if link.attrs.get("href") in self.index.keys()] 

        # Update the adjacency matrix
        self.adj_list[self.root] += urls
        return
   

   class CategoryCrawler(Crawler):
    """
    Specialised crawler for collecting hyperlinks
    to subpages and subcategories of the category page
    """
    
    @staticmethod
    def get_subcategories(soup: BeautifulSoup) -> List[Link]:
        """
        Extract all subcategories of the category page
        """
        subcategories = soup.find(name="div", attrs={"id":"mw-subcategories"})
        assert subcategories, "Not a category page!"

        links = subcategories.find_all(name="a")

        return [
            Link(title=link.attrs["title"], url=link.attrs["href"])
            for link in links     
        ]
    
    @staticmethod
    def get_subpages(soup: BeautifulSoup) -> List[Link]:
        """
        Extract all subpages of the subcategory page
        """
        subpages = soup.find(name="div", attrs={"id": "mw-pages"})
        assert subpages, "Does not contain subpages!"

        links = subpages.find(attrs={"class": "mw-category"})\
                        .find_all(name="a")

        return [
            Link(title=link.attrs["title"], url=link.attrs["href"])
            for link in links     
        ]
    
    def register_links(
            self, links: List[Link], add_to_queue: bool = False
        ) -> None:
        """
        Add discovered links to the page index.
        Add links connection to the adjacency list.
        """
        for link in links:
            # Add discovered links to the page index.
            self.index[link.url] = link.title

            # Add links connection to the adjacency list.
            self.adj_list[self.root].append(link.url)

            # Add to queue
            if add_to_queue and not link.url in self.queue:
                self.queue.append(link.url)

        return

    def run(self) -> None:
        """
        Main method
        """
        # Get page source
        url: str = self.url_base + self.root
        soup: BeautifulSoup = self.get_soup(url)

        # Register subcategories
        # Some cateogory pages are leaf pages in the category taxonomy
        # We still want to capture their subpages, thus we skip the assertion error
        try:
            subcat_links: List[Link] = self.get_subcategories(soup)
            self.register_links(subcat_links, add_to_queue=True)
        except AssertionError:
            print(f"Category {self.root} does not contain subcategories.")

        # Register subpages
        subpage_links: List[Link] = self.get_subpages(soup)
        self.register_links(subpage_links)

        return