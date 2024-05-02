import feedparser
import requests
from bs4 import BeautifulSoup
from transformers import pipeline, BartTokenizer
import markdown
from tqdm import tqdm
import sys, os
import concurrent.futures
import pandas as pd
from datetime import datetime

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")

MARKDOWN_FILE = "summary_output.md"


def get_current_date():
    return datetime.now().strftime("%m/%d/%Y")


md_header = f"""# TL;DR News

**Project Title:** RSS-Summarizer

**Author:** [Tucker Craig](https://tuckercraig.com)

**Source:** [BBC World News](http://feeds.bbci.co.uk/news/world/rss.xml)

**Script ran:** `{get_current_date()}`  

---\n"""


class Article:
    def __init__(
        self, title, link, summary, publication_date, image_url=None, description=None
    ):
        self.title = title
        self.link = link
        self.image_url = image_url
        self.description = description
        self.summary = summary
        self.publication_date = publication_date
        self.relevance = self.parse_date(publication_date)
        self.brief = f"{self.title} - {self.summary[:20]}..."

    @staticmethod
    def parse_date(date_str):
        return pd.to_datetime(date_str)

    def __repr__(self):
        return (
            f"### [{self.title}]({self.link})\n\n"
            + (f"![Image]({self.image_url})\n\n" if self.image_url else "")
            + f"{self.summary}\n\n"
            + f"_Published on: {self.publication_date}_\n\n"
        )


def fetch_and_summarize(entry):
    try:
        response = requests.get(entry.link)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        content = soup.find_all(["p", "h1", "h2", "h3"])
        text = " ".join([elem.get_text() for elem in content])

        published = None
        if "published" in entry:
            published = entry.published
        image_url = None
        if "media_thumbnail" in entry:
            media_contents = entry["media_thumbnail"]
            if media_contents and len(media_contents) > 0:
                image_url = media_contents[0]["url"]
        description = None
        if "summary" in entry:
            description = f"{entry.summary}\n\n"

        summary = summarize_text(text)
        return Article(
            entry.title, entry.link, summary, published, image_url, description
        )
    except Exception as e:
        print(f"Error processing {entry.title}: {str(e)}")
        return None


def summarize_text(text):
    word_count = len(text.split())
    user_min_len = max(30, int(word_count * 0.1))
    user_max_len = max(130, int(word_count * 0.2))
    inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = summarizer.model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=user_max_len,
        min_length=user_min_len,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.9,
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


def process_feed(url):
    feed = feedparser.parse(url)
    articles = []
    num_threads = os.cpu_count() * 4

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [
            executor.submit(fetch_and_summarize, entry) for entry in feed.entries
        ]
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Processing Articles",
        ):
            article = future.result()
            if article:
                articles.append(article)

    articles.sort(key=lambda x: x.relevance, reverse=True)

    with open(MARKDOWN_FILE, "w", encoding="utf-8") as md_file:
        md_file.write(md_header)
        for article in articles:
            md_file.write(str(article))


def write_markdown_html():
    """Reads the Markdown file and converts it to HTML."""
    if os.path.exists(MARKDOWN_FILE):
        with open(MARKDOWN_FILE, "r", encoding="utf-8") as file:
            text = file.read()
            content = markdown.markdown(text)
        with open("index.html", "w", encoding="utf-8") as file:
            html = f"""
            <!doctype html>
            <html lang="en">
            <head>
                <meta charset="utf-8">
                <title>Summary Output</title>
                <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css">
            </head>
            <body>
                <div class="container mt-5">
                    {content}
                </div>
            </body>
            </html>
            """
            file.write(html)


if __name__ == "__main__":
    rss_url = "http://feeds.bbci.co.uk/news/world/rss.xml"
    process_feed(rss_url)
    write_markdown_html()
