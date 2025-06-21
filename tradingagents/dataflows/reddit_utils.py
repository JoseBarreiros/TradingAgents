import requests
import time
import json
from datetime import datetime, timedelta
from contextlib import contextmanager
from typing import Annotated, List, Dict
import os
import re
import praw

ticker_to_company = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "GOOGL": "Google",
    "AMZN": "Amazon",
    "TSLA": "Tesla",
    "NVDA": "Nvidia",
    "TSM": "Taiwan Semiconductor Manufacturing Company OR TSMC",
    "JPM": "JPMorgan Chase OR JP Morgan",
    "JNJ": "Johnson & Johnson OR JNJ",
    "V": "Visa",
    "WMT": "Walmart",
    "META": "Meta OR Facebook",
    "AMD": "AMD",
    "INTC": "Intel",
    "QCOM": "Qualcomm",
    "BABA": "Alibaba",
    "ADBE": "Adobe",
    "NFLX": "Netflix",
    "CRM": "Salesforce",
    "PYPL": "PayPal",
    "PLTR": "Palantir",
    "MU": "Micron",
    "SQ": "Block OR Square",
    "ZM": "Zoom",
    "CSCO": "Cisco",
    "SHOP": "Shopify",
    "ORCL": "Oracle",
    "X": "Twitter OR X",
    "SPOT": "Spotify",
    "AVGO": "Broadcom",
    "ASML": "ASML ",
    "TWLO": "Twilio",
    "SNAP": "Snap Inc.",
    "TEAM": "Atlassian",
    "SQSP": "Squarespace",
    "UBER": "Uber",
    "ROKU": "Roku",
    "PINS": "Pinterest",
}


def fetch_top_from_category(
    category: Annotated[
        str, "Category to fetch top post from. Collection of subreddits."
    ],
    date: Annotated[str, "Date to fetch top posts from."],
    max_limit: Annotated[int, "Maximum number of posts to fetch."],
    query: Annotated[str, "Optional query to search for in the subreddit."] = None,
    data_path: Annotated[
        str,
        "Path to the data folder. Default is 'reddit_data'.",
    ] = "reddit_data",
):
    # Validate category
    if not isinstance(category, str) or not category.strip():
        raise ValueError("Category must be a non-empty string.")
    try:
        datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        raise ValueError(f"Date '{date}' must be in 'YYYY-MM-DD' format.")

    # Validate max_limit
    if not isinstance(max_limit, int) or max_limit <= 0:
        raise ValueError("max_limit must be a positive integer.")

    base_path = data_path

    all_content = []

    if max_limit < len(os.listdir(os.path.join(base_path, category))):
        raise ValueError(
            "REDDIT FETCHING ERROR: max limit is less than the number of files in the category. Will not be able to fetch any posts"
        )

    limit_per_subreddit = max_limit // len(
        os.listdir(os.path.join(base_path, category))
    )

    for data_file in os.listdir(os.path.join(base_path, category)):
        # check if data_file is a .jsonl file
        if not data_file.endswith(".jsonl"):
            continue

        all_content_curr_subreddit = []

        with open(os.path.join(base_path, category, data_file), "rb") as f:
            for i, line in enumerate(f):
                # skip empty lines
                if not line.strip():
                    continue

                parsed_line = json.loads(line)

                # select only lines that are from the date
                post_date = datetime.utcfromtimestamp(
                    parsed_line["created_utc"]
                ).strftime("%Y-%m-%d")
                if post_date != date:
                    continue

                # if is company_news, check that the title or the content has the company's name (query) mentioned
                if "company" in category and query:
                    search_terms = []
                    if query not in ticker_to_company:
                        company_name = query
                    else:
                        company_name = ticker_to_company[query]
                    if "OR" in company_name:
                        search_terms = company_name.split(" OR ")
                    else:
                        search_terms = [company_name]

                    search_terms.append(query)

                    found = False
                    for term in search_terms:
                        if re.search(
                            term, parsed_line["title"], re.IGNORECASE
                        ) or re.search(term, parsed_line["selftext"], re.IGNORECASE):
                            found = True
                            break

                    if not found:
                        continue

                post = {
                    "title": parsed_line["title"],
                    "content": parsed_line["selftext"],
                    "url": parsed_line["url"],
                    "upvotes": parsed_line["ups"],
                    "posted_date": post_date,
                }

                all_content_curr_subreddit.append(post)

        # sort all_content_curr_subreddit by upvote_ratio in descending order
        all_content_curr_subreddit.sort(key=lambda x: x["upvotes"], reverse=True)

        all_content.extend(all_content_curr_subreddit[:limit_per_subreddit])

    return all_content


def fetch_top_from_category_online(
    category: Annotated[str, "Comma-separated list of subreddits or a category name."],
    date: Annotated[str, "Date to fetch top posts from."],
    max_limit: Annotated[int, "Maximum number of posts to fetch."],
    query: Annotated[str, "Optional query to search for in the subreddit."] = None,
    subreddit_map: Dict[str, List[str]] = None,  # Map category to list of subreddits
) -> List[Dict]:
    """
    Fetch top posts from Reddit online for a given category and date using Reddit API (PRAW).
    """
    reddit_client = praw.Reddit(
        client_id=os.environ["REDDIT_CLIENT_ID"],
        client_secret=os.environ["REDDIT_CLIENT_SECRET"],
        user_agent="script:trading_agents:v1.0 (by u/SpiritQueasy3662)",
    )
    if reddit_client is None:
        raise ValueError("A PRAW Reddit client instance must be provided.")

    # Validate category
    if not isinstance(category, str) or not category.strip():
        raise ValueError("Category must be a non-empty string.")
    try:
        date_dt = datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        raise ValueError(f"Date '{date}' must be in 'YYYY-MM-DD' format.")

    if not isinstance(max_limit, int) or max_limit <= 0:
        raise ValueError("max_limit must be a positive integer.")

    # Map category to subreddits if needed
    if subreddit_map and category in subreddit_map:
        subreddits = subreddit_map[category]
    else:
        subreddits = [s.strip() for s in category.split(",") if s.strip()]

    all_content = []
    limit_per_subreddit = max_limit // len(subreddits) if subreddits else max_limit

    # Calculate start and end timestamps for the date
    start_epoch = int(datetime(date_dt.year, date_dt.month, date_dt.day).timestamp())
    end_epoch = int((date_dt + timedelta(days=1)).timestamp())

    for subreddit_name in subreddits:
        subreddit = reddit_client.subreddit(subreddit_name)
        count = 0
        for submission in subreddit.top(
            time_filter="day", limit=100
        ):  # Fetch top 100 for the day
            created_utc = int(submission.created_utc)
            if not (start_epoch <= created_utc < end_epoch):
                continue
            if query:
                # Check if query is in title or selftext
                if not (
                    re.search(query, submission.title, re.IGNORECASE)
                    or re.search(
                        query, getattr(submission, "selftext", ""), re.IGNORECASE
                    )
                ):
                    continue
            post_obj = {
                "title": submission.title,
                "content": getattr(submission, "selftext", ""),
                "url": submission.url,
                "upvotes": submission.score,
                "posted_date": datetime.utcfromtimestamp(created_utc).strftime(
                    "%Y-%m-%d"
                ),
            }
            all_content.append(post_obj)
            count += 1
            if count >= limit_per_subreddit:
                break

    all_content.sort(key=lambda x: x["upvotes"], reverse=True)
    return all_content[:max_limit]
