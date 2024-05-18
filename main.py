import logging
import os
import telebot

from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore

logging.basicConfig(level=logging.INFO)


def load_api_credentials():
    load_dotenv()

    openai_api_key = os.environ.get("OPENAI_API_KEY")
    telegram_bot_token = os.environ.get("BOT_TOKEN")
    pinecone_index = os.environ.get("PINECONE_INDEX_NAME")

    if not openai_api_key:
        logging.error("OPENAI_API_KEY is not set in environment variables")
        raise SystemExit(1)

    if not telegram_bot_token:
        logging.error("TELEGRAM_BOT_TOKEN is not set in environment variables")
        raise SystemExit(1)

    if not pinecone_index:
        logging.error("PINECONE_INDEX is not set in environment variables")
        raise SystemExit(1)

    return openai_api_key, telegram_bot_token, pinecone_index


def format_document_content(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def create_telegram_bot(bot_token):
    try:
        bot = telebot.TeleBot(bot_token)
        return bot
    except Exception as e:
        logging.error(f"Error creating Telegram bot instance: {e}")
        raise SystemExit(1)

def generate_rag_chain(retriever, llm):
    prompt_template = """Ты - Света, ты являешься ассистентом на крипто-хакатоне компании Latoken.
    Для начала, определи, относится ли последнее сообщение к компании хакатону, или смежной теме, компании Латокен и т.д
    Если вопрос относится относится, тогда используй приведенные ниже фрагменты извлеченного контекста,
    чтобы ответить на вопрос.
    Если вопрос относится к хакатону, или Латокен, и ты не знаешь ответа, 
    скажи что не уверен в ответе на вопрос, но вы можете обратиться к организаторам хакатона для получения информации.
    Если вопрос не относится, тогда отвечай этим символом: ⠀.

    \n
    Вопрос: {question} \n
    Контекст: {context}  \n
    Ответ: """

    prompt = ChatPromptTemplate.from_template(prompt_template)

    rag_chain = (
            {"context": retriever | format_document_content, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )

    return rag_chain

def load_training_data():
    with open('training_data.txt', 'r') as file:
        contents = file.read()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_text(contents)

    return texts


def create_embeddings(texts, pinecone_index_name):
    embeddings = OpenAIEmbeddings()
    vectorstore = PineconeVectorStore.from_texts(texts, embeddings, index_name=pinecone_index_name)
    return vectorstore

def setup_bot(openai_api_key, telegram_bot_token, pinecone_index):
    data = load_training_data()
    vectorstore = create_embeddings(data, pinecone_index)
    bot = create_telegram_bot(telegram_bot_token)

    @bot.message_handler(func=lambda message: True)
    def echo_hackaton_related(message):
        rag_chain = generate_rag_chain(retriever=vectorstore.as_retriever(),
                                       llm=ChatOpenAI(model="gpt-3.5-turbo-0125", api_key=openai_api_key))
        response = rag_chain.invoke(message.text)
        print(response)

        if "⠀" != response:
            bot.reply_to(message, response)
        else:
            print("Not related to Hackaton or Latoken question. Ignoring..")

    return bot


def main():
    credentials = load_api_credentials()
    bot = setup_bot(*credentials)

    bot.infinity_polling()


if __name__ == "__main__":
    main()
