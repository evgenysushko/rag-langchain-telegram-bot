import os
import argparse
import openai
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

# Load environment variables. Assumes that project contains .env file with API keys
load_dotenv()
openai.api_key = os.environ['OPENAI_API_KEY']

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


def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text

    # Prepare the DB.
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=OPENAI_EMBEDDINGS)

    # Search the DB.
    results = db.similarity_search_with_relevance_scores(query_text, k=3)

    # if len(results) == 0 or results[0][1] < MIN_SIMILARITY_SCORE:
    #     print(f"Unable to find matching results.")
    #     return

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE_PL)
    prompt = prompt_template.format(context=context_text, question=query_text)
    # print(prompt)

    model = ChatOpenAI()
    response_text = model.predict(prompt)

    sources = [doc.metadata.get("source", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)


if __name__ == "__main__":
    main()
