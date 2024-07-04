# Legal documents QA Telegram bot

A QA Telegram bot in Polish based on the legal information from parsed websites

## Usage

### Parse websites

Parse websites and save information to text files

`python parse_websites.py`

### Create Vector DB

Create and save Chroma Vector DB using OpenAI Embeddings and Langchain

`python create_database.py`

### Run Telegram bot

Run QA Telegram bot that is able to answer questions in Polish based on that data stored in the Vector DB

`python telegram_bot.py`
