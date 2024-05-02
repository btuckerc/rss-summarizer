import feedparser
import requests
from bs4 import BeautifulSoup
from transformers import pipeline, BartTokenizer
import markdown
from tqdm import tqdm
import sys, os, re, json, pytz
import concurrent.futures
import pandas as pd
from datetime import datetime
from collections import Counter
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")

MARKDOWN_FILE = "summary_output.md"


def get_current_date():
    # Get the current date and time in UTC, then convert to EST
    utc_now = datetime.now(pytz.utc)
    est = pytz.timezone("America/New_York")
    return utc_now.astimezone(est)


md_header = f"""# TL;DR News

**Project Title:** RSS-Summarizer

**Author:** [Tucker Craig](https://tuckercraig.com)

**Source:** [BBC World News](http://feeds.bbci.co.uk/news/world/rss.xml)

**Script ran:** `{get_current_date().strftime("%m/%d/%Y")}`  

---\n"""


def calculate_relevance(article, keyword_list, current_date):
    # Ensure current_date is a datetime object with timezone
    current_date = pd.to_datetime(current_date)

    # Basic keyword relevance scoring
    words = article.content.lower().split()
    word_freq = Counter(words)
    keyword_relevance = sum(word_freq[key] for key in keyword_list if key in word_freq)

    # Sentiment analysis
    sia = SentimentIntensityAnalyzer()
    sentiment_score = sia.polarity_scores(article.content)["compound"]

    # Convert publication date to EST
    est = pytz.timezone("America/New_York")
    if article.publication_date:
        pub_date_est = (
            pd.to_datetime(article.publication_date).tz_localize("GMT").tz_convert(est)
        )
    else:
        pub_date_est = None

    # Recency scoring using minutes
    if pub_date_est and isinstance(pub_date_est, datetime):
        minutes_since_pub = (
            current_date - pub_date_est
        ).total_seconds() / 60  # Convert seconds to minutes
        # Define a recency threshold, e.g., 180 minutes (3 hours)
        threshold_minutes = 3600
        if minutes_since_pub <= threshold_minutes:
            recency_score = (
                (threshold_minutes - minutes_since_pub) / threshold_minutes * 100
            )
        else:
            # Apply a decreasing score for older articles
            recency_score = max(50 - (minutes_since_pub - threshold_minutes) / 10, 0)
    else:
        recency_score = 0  # Default to 0 if date is invalid

    # Composite score
    relevance = keyword_relevance * 0.1 + sentiment_score * 0.2 + recency_score * 0.7
    return relevance


class Article:

    def __init__(
        self, title, link, content, publication_date, image_url=None, description=None
    ):
        self.title = title
        self.link = link
        self.image_url = image_url
        self.description = description
        self.summary = None
        self.content = content
        self.publication_date = publication_date
        self.relevance = calculate_relevance(self, [], get_current_date())
        self.brief = f"{self.title} - {self.content[:20]}..."

    @staticmethod
    def parse_date(date_str):
        return (
            pd.to_datetime(date_str).tz_localize("GMT").tz_convert("America/New_York")
        )

    def summarize(self):
        self.summary = summarize_text(self.content)
        return self.summary

    def __repr__(self):
        return (
            f"### [{self.title}]({self.link})\n\n"
            + (f"![Image]({self.image_url})\n\n" if self.image_url else "")
            + f"{self.summary}\n\n"
            + f"_Published on: {self.publication_date}_\n\n"
        )


def find_keys(node, key, accum):
    if isinstance(node, list):
        for i in node:
            find_keys(i, key, accum)
    elif isinstance(node, dict):
        if key in node:
            accum.append(node[key])
        for j in node.values():
            find_keys(j, key, accum)


def extract_json_data(soup):
    script = soup.find("script", {"id": "__NEXT_DATA__"})
    if script:
        json_data = json.loads(script.string)
        return json_data
    return None


def extract_text_from_json(json_data):
    texts = []
    find_keys(json_data, "text", texts)
    return texts


def fetch_article(entry):
    try:
        response = requests.get(entry.link)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        content = soup.find_all(["p", "h1", "h2", "h3"])
        text = " ".join([elem.get_text() for elem in content])
        if len(text.split()) < 1:  # if there aren't any words in the article
            json_data = extract_json_data(soup)
            hidden_article_text = extract_text_from_json(json_data)
            text = " ".join(hidden_article_text)

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
            description = entry.summary

        return Article(entry.title, entry.link, text, published, image_url, description)
    except Exception as e:
        print(f"Error processing {entry.title}: {str(e)}")
        return None


def summarize_text(text):
    word_count = len(text.split())
    if word_count < 30:
        return "No summary available."
    user_min_len = max(40, int(word_count * 0.05))
    user_max_len = max(100, int(word_count * 0.1))
    inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = summarizer.model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=user_max_len,
        min_length=user_min_len,
        do_sample=False,  # More deterministic outputs
        num_beams=6,
        length_penalty=2.5,
        early_stopping=True,
        no_repeat_ngram_size=2,  # Prevents repetitive ngrams for better quality text
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


def summarize_article(article):
    return summarize_text(article.content)


def process_feed(url):
    feed = feedparser.parse(url)
    articles = []

    # Fetch articles using ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=os.cpu_count() * 4
    ) as executor:
        future_to_article = {
            executor.submit(fetch_article, entry): entry for entry in feed.entries
        }
        for future in tqdm(
            concurrent.futures.as_completed(future_to_article),
            total=len(future_to_article),
            desc="Fetching articles",
        ):
            article = future.result()
            if article:
                articles.append(article)

    # Summarize articles using ProcessPoolExecutor
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Using executor.map to ensure the order is preserved
        summaries = list(
            tqdm(
                executor.map(summarize_article, articles),
                total=len(articles),
                desc="Summarizing articles",
            )
        )

    for article, summary in zip(articles, summaries):
        article.summary = summary  # Update each article with its summary

    # Sort articles by relevance
    articles.sort(key=lambda x: x.relevance, reverse=True)

    # Write to markdown file
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
    nltk.download("vader_lexicon")
    rss_url = "http://feeds.bbci.co.uk/news/world/rss.xml"
    # rss_url = "http://feeds.arstechnica.com/arstechnica/technology-lab"  # timezone broken here
    # rss_url = "http://rss.cnn.com/rss/cnn_us.rss" # date missing for some
    process_feed(rss_url)
    write_markdown_html()
