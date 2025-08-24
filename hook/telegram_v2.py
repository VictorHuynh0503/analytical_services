import telebot
import pandas as pd
import requests
from io import StringIO

# Your Telegram bot token and chat ID
BOT_TOKEN = '1200942736:AAEG8y9qyJ7aHefUm4vt_xKqkNBxfKd3qCc'
CHAT_ID = '@vincent_signal'

# Define the maximum length for Telegram messages
MAX_LENGTH = 4096 - 6  # 6 characters for Markdown code block (```\n```\n)

# Function to escape Markdown special characters
def escape_markdown(text):
    escape_chars = ['_', '*', '[', ']', '(', ')', '~', '`', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!']
    for char in escape_chars:
        text = text.replace(char, '\\' + char)
    return text

# Function to format DataFrame as a table with columns separated by "|"
def format_dataframe_as_table(df):
    header = ' | '.join(df.columns)
    rows = df.apply(lambda row: ' | '.join(row.astype(str)), axis=1)
    table = f"{header}\n{'-' * len(header)}\n" + '\n'.join(rows)
    return escape_markdown(table)

# Function to truncate DataFrame
def truncate_dataframe(df, max_length):
    text = format_dataframe_as_table(df)
    if len(text) <= max_length:
        return text

    # Determine how many rows can fit within the max_length
    rows = df.shape[0]
    while rows > 0:
        truncated_df = df.iloc[:rows]
        text = format_dataframe_as_table(truncated_df)
        if len(text) <= max_length:
            return text
        rows -= 1
    
    return text  # Return the last attempt, even if it's too large

# Function to split text into chunks, ensuring words are not cut off
def split_text_into_chunks(text, max_length):
    lines = text.split('\n')
    chunks = []
    current_chunk = ""

    for line in lines:
        if len(current_chunk) + len(line) + 1 > max_length:
            chunks.append(current_chunk)
            current_chunk = line
        else:
            if current_chunk:
                current_chunk += '\n' + line
            else:
                current_chunk = line

    if current_chunk:
        chunks.append(current_chunk)

    return chunks

# Function to send a message via Telegram Bot API
def call_api(token, chat_id, message):
    url = f'https://api.telegram.org/bot{token}/sendMessage'
    payload = {
        'chat_id': chat_id,
        'text': message,
        'parse_mode': 'MarkdownV2'
    }
    response = requests.post(url, json=payload)
    return response


def send_telegram_message(df, token, chat_id):
    
    # Format the DataFrame as a table
    formatted_table = truncate_dataframe(df, MAX_LENGTH)

    # Split the formatted table into chunks if necessary
    chunks = split_text_into_chunks(f"```\n{formatted_table}\n```", MAX_LENGTH)

    # Send each chunk as a separate message
    for chunk in chunks:
        response = call_api(token, chat_id, chunk)
        print(response.json())


bot = telebot.TeleBot(BOT_TOKEN)

# Function to generate sample football match data
def get_sample_data():
    data = {
        "Date": ["2024-02-01", "2024-01-28", "2024-01-25"],
        "Opponent": ["Team A", "Team B", "Team C"],
        "Home Score": [2, 1, 3],
        "Away Score": [1, 2, 0],
        "BTTS": [True, True, False]
    }
    df = pd.DataFrame(data)
    return df

@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    bot.reply_to(message, "Welcome! Use /history All to get all matches or /history <team_name> to get specific results.")

@bot.message_handler(commands=['history'])
def send_history(message):
    args = message.text.split()[1:]
    df = get_sample_data()
    
    if args:
        query = ' '.join(args)
        if query.lower() != "all":
            df = df[df['Opponent'].str.contains(query, case=False, na=False)]
    
    if df.empty:
        bot.reply_to(message, "No results found for your query.")
    else:
        send_telegram_message(df, BOT_TOKEN, CHAT_ID)

if __name__ == "__main__":
    print("Bot is running...")
    bot.polling(none_stop=True)