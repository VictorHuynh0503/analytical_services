import requests
import json
from typing import List, Dict, Any
import pandas as pd

DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/1404056069080219698/EhiyLSwoR-yS_PO9TDVcw83sPjGwXNOvLVpYGYlBJqiiPb1QHhsAHrX_-A6dXDYdTInk"  # Replace

def send_discord_message(content: str):
    """Send a plain text message to Discord."""
    payload = {"content": content}
    response = requests.post(DISCORD_WEBHOOK_URL, json=payload)
    if not response.ok:
        raise Exception(f"Discord API error: {response.status_code} {response.text}")

def send_table(headers: List[str], rows: List[List[Any]], title: str = "📊 Table Data"):
    """Send a table-formatted message to Discord."""
    col_widths = [max(len(str(cell)) for cell in col) for col in zip(headers, *rows)]
    header_line = " | ".join(f"{headers[i]:<{col_widths[i]}}" for i in range(len(headers)))
    separator = "-+-".join("-" * col_widths[i] for i in range(len(headers)))
    row_lines = "\n".join(
        " | ".join(f"{str(row[i]):<{col_widths[i]}}" for i in range(len(headers))) for row in rows
    )

    table_text = f"```\n{header_line}\n{separator}\n{row_lines}\n```"
    send_discord_message(f"**{title}**\n{table_text}")

def send_timeseries(data: Dict[str, float], title: str = "📈 Time Series"):
    """Send time series data to Discord."""
    lines = "\n".join(f"{timestamp}: {value}" for timestamp, value in data.items())
    formatted = f"```\n{lines}\n```"
    send_discord_message(f"**{title}**\n{formatted}")

def send_news(headline: str, body: str, source: str = None):
    """Send a news-style alert to Discord."""
    message = f"📰 **{headline}**\n{body}"
    if source:
        message += f"\n_Source: {source}_"
    send_discord_message(message)


MAX_LENGTH = 2000 - 8  # For code block markers: ```\n + \n```

def escape_discord_markdown(text: str) -> str:
    """Escape characters that may cause formatting issues in Discord Markdown."""
    escape_chars = ['`', '*', '_', '~', '|', '>']
    for char in escape_chars:
        text = text.replace(char, f"\\{char}")
    return text

def format_dataframe_as_table(df: pd.DataFrame) -> str:
    """Format DataFrame as a Markdown table."""
    headers = list(df.columns)
    col_widths = [max(len(str(cell)) for cell in [headers] + df[col].astype(str).tolist()) for col in headers]

    header_line = " | ".join(f"{headers[i]:<{col_widths[i]}}" for i in range(len(headers)))
    separator = "-+-".join("-" * col_widths[i] for i in range(len(headers)))
    row_lines = "\n".join(
        " | ".join(f"{str(row[i]):<{col_widths[i]}}" for i in range(len(headers)))
        for row in df.itertuples(index=False, name=None)
    )

    return escape_discord_markdown(f"{header_line}\n{separator}\n{row_lines}")

def truncate_dataframe(df: pd.DataFrame, max_length: int) -> str:
    """Trim DataFrame to fit Discord's message limit."""
    text = format_dataframe_as_table(df)
    if len(text) <= max_length:
        return text

    rows = df.shape[0]
    while rows > 0:
        truncated_df = df.iloc[:rows]
        text = format_dataframe_as_table(truncated_df)
        if len(text) <= max_length:
            return text
        rows -= 1

    return text

def split_text_into_chunks(text: str, max_length: int) -> list:
    """Split text into chunks without breaking lines."""
    lines = text.split("\n")
    chunks = []
    current_chunk = ""

    for line in lines:
        if len(current_chunk) + len(line) + 1 > max_length:
            chunks.append(current_chunk)
            current_chunk = line
        else:
            if current_chunk:
                current_chunk += "\n" + line
            else:
                current_chunk = line

    if current_chunk:
        chunks.append(current_chunk)

    return chunks

def send_discord_message(content: str):
    """Send a raw message to Discord."""
    payload = {"content": content}
    response = requests.post(DISCORD_WEBHOOK_URL, json=payload)
    if not response.ok:
        raise Exception(f"Failed to send message: {response.status_code}, {response.text}")

def send_dataframe_to_discord(df: pd.DataFrame, title: str = "📊 Data Table"):
    """Send a DataFrame to Discord in one or multiple messages."""
    formatted_table = truncate_dataframe(df, MAX_LENGTH)
    chunks = split_text_into_chunks(f"```\n{formatted_table}\n```", MAX_LENGTH)

    for chunk in chunks:
        send_discord_message(f"**{title}**\n{chunk}")