"""
Crawling controller
"""

from typing import Dict, List
from crawlers import PageCrawler, CategoryCrawler

from tqdm import tqdm


class Controller:
    """
    Controls crawling flow.
    Registers new crawlers
    """

    def __init__(self, root: str = "wiki/Kategorie:Stochastik"):
        # Root
        self.root: str = root

        # Tracking pages to visit
        self.queue: List[str] = [root]

        # Link - Title page index
        self.index: Dict[str, str] = {}

        # Links graph adjacency list
        self.adj_list: Dict[str, List[str]] = {}

    def main(self) -> None:
        """
        Main loop
        ----------
        1. Index all subcategory pages and their content child pages
        2. Parse all child pages
        3. For each child page, only keep links that are in the index
        """
        self.errors = []

        # Parsing category pages
        print("Parsing subcategories...")
        while len(self.queue) > 0:
            url: str = self.queue.pop(0)
            try:
                crawler: CategoryCrawler = CategoryCrawler(
                    root=url, queue=self.queue, index=self.index, adj_list=self.adj_list
                )
                crawler.run()
            except AssertionError:
                self.errors.append(url)
        print(
            f"Complete. Found {len(self.adj_list.keys())} subcategories and {len(self.index.keys())} pages."
        )

        # Parse pages
        page_links = [link for link in self.index if "Kategorie" not in link]

        for url in tqdm(page_links):
            try:
                crawler: PageCrawler = PageCrawler(
                    root=url, queue=None, index=self.index, adj_list=self.adj_list
                )
                crawler.run()
            except AssertionError:
                self.errors.append(url)
        print("Done.")

        return
