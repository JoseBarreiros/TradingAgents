import json
import requests
import os
from bs4 import BeautifulSoup
from datetime import datetime
import time
import random
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    retry_if_result,
)


def is_rate_limited(response):
    """Check if the response indicates rate limiting (status code 429)"""
    return response.status_code == 429


@retry(
    retry=(retry_if_result(is_rate_limited)),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    stop=stop_after_attempt(5),
)
def make_request(url, headers):
    """Make a request with retry logic for rate limiting"""
    # Random delay before each request to avoid detection
    time.sleep(random.uniform(2, 6))
    response = requests.get(url, headers=headers)
    return response


def getNewsData(query, start_date, end_date):
    """
    Scrape Google News search results for a given query and date range.
    query: str - search query
    start_date: str - start date in the format yyyy-mm-dd or mm/dd/yyyy
    end_date: str - end date in the format yyyy-mm-dd or mm/dd/yyyy
    """
    if "-" in start_date:
        start_date = datetime.strptime(start_date, "%Y-%m-%d")
        start_date = start_date.strftime("%m/%d/%Y")
    if "-" in end_date:
        end_date = datetime.strptime(end_date, "%Y-%m-%d")
        end_date = end_date.strftime("%m/%d/%Y")
    assert start_date < end_date, "Start date must be before end date"
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query must be a non-empty string.")

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/101.0.4951.54 Safari/537.36"
        )
    }

    news_results = []
    page = 0
    while True:
        offset = page * 10
        url = (
            f"https://www.google.com/search?q={query}"
            f"&tbs=cdr:1,cd_min:{start_date},cd_max:{end_date}"
            f"&tbm=nws&start={offset}"
        )

        try:
            response = make_request(url, headers)
            soup = BeautifulSoup(response.content, "html.parser")
            results_on_page = soup.select("div.SoaBEf")

            if not results_on_page:
                break  # No more results found

            for el in results_on_page:
                try:
                    link = el.find("a")["href"]
                    title = el.select_one("div.MBeuO").get_text()
                    snippet = el.select_one(".GI74Re").get_text()
                    date = el.select_one(".LfVVr").get_text()
                    source = el.select_one(".NUnG9d span").get_text()
                    news_results.append(
                        {
                            "link": link,
                            "title": title,
                            "snippet": snippet,
                            "date": date,
                            "source": source,
                        }
                    )
                except Exception as e:
                    print(f"Error processing result: {e}")
                    # If one of the fields is not found, skip this result
                    continue

            # Update the progress bar with the current count of results scraped

            # Check for the "Next" link (pagination)
            next_link = soup.find("a", id="pnnext")
            if not next_link:
                break

            page += 1

        except Exception as e:
            print(f"Failed after multiple retries: {e}")
            break

    return news_results


def getNewsData_api(query, start_date, end_date):
    """
    Fetch news articles for a given query and date range using NewsAPI.
    Args:
        query: str - search query
        start_date: str - start date in the format yyyy-mm-dd
        end_date: str - end date in the format yyyy-mm-dd
    Returns:
        list of dicts: Each dict contains 'title', 'description', 'url', 'publishedAt', 'source'
    """
    api_key = os.environ.get("NEWSAPI_KEY")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query must be a non-empty string.")
    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    except ValueError:
        raise ValueError("Dates must be in yyyy-mm-dd format.")
    assert start_dt < end_dt, "Start date must be before end date"

    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "from": start_date,
        "to": end_date,
        "sortBy": "relevancy",
        "language": "en",
        "pageSize": 100,  # max per page
        "apiKey": api_key,
    }

    news_results = []
    page = 1
    while True:
        params["page"] = page
        response = requests.get(url, params=params)
        if response.status_code != 200:
            print(f"API error: {response.status_code} - {response.text}")
            break
        data = response.json()
        articles = data.get("articles", [])
        if not articles:
            break
        for article in articles:
            news_results.append(
                {
                    "title": article.get("title"),
                    "snippet": article.get("description"),
                    "link": article.get("url"),
                    "date": article.get("publishedAt"),
                    "source": article.get("source", {}).get("name"),
                }
            )
        if len(articles) < params["pageSize"]:
            break  # No more pages
        page += 1

    return news_results
