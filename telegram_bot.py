import os
import logging
import openai
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)

# Load environment variables. Assumes that project contains .env file with API keys
load_dotenv()
openai.api_key = os.environ['OPENAI_API_KEY']
TELEGRAM_BOT_TOKEN = os.environ['TELEGRAM_BOT_TOKEN']

CHROMA_PATH = "chroma"
OPENAI_EMBEDDINGS = OpenAIEmbeddings(model='text-embedding-3-small')
# MIN_SIMILARITY_SCORE = 0.7

PROMPT_TEMPLATE_PL = """
Odpowiedz na pytanie wyłącznie w oparciu o następujący kontekst.
Jeśli na podstawie kontekstu nie można udzielić odpowiedzi, powiedz „nie wiem”:

{context}

---

Odpowiedz na pytanie w oparciu o powyższy kontekst: {question}
"""

DATABASE = None

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        """
        Cześć! Jestem botem, który pomoże Ci odpowiedzieć na Twoje pytania prawne. Proszę poczekać chwilę, ładuję bazę danych...
        """
    )
    global DATABASE
    DATABASE = Chroma(persist_directory=CHROMA_PATH, embedding_function=OPENAI_EMBEDDINGS)
    await update.message.reply_text(
        """Baza danych została załadowana! Jakie jest Twoje pytanie?"""
    )


async def query(update: Update, context: ContextTypes.DEFAULT_TYPE):
    results = DATABASE.similarity_search_with_relevance_scores(update.message.text, k=3)
    # if len(results) == 0 or results[0][1] < MIN_SIMILARITY_SCORE:
        # await update.message.reply_text("Unable to find an answer to your question")

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE_PL)
    prompt = prompt_template.format(context=context_text, question=update.message.text)
    # await update.message.reply_text(prompt)

    model = ChatOpenAI()
    response_text = model.predict(prompt)

    sources = [doc.metadata.get("source", None).split('/')[-1].replace("__", "/")[:-4] for doc, _score in results]
    formatted_response = f"Odpowiedź: {response_text}\n\nŹródła: {sources}\n\nChcesz zadać kolejne pytanie?"
    await update.message.reply_text(formatted_response)

if __name__ == "__main__":
    application = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    application.add_handler(CommandHandler('start', start))
    application.add_handler(MessageHandler(filters.TEXT, query))
    # Run the bot until the user presses Ctrl-C
    application.run_polling(allowed_updates=Update.ALL_TYPES)
