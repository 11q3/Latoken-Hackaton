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
    pinecone_api_key = os.environ.get("PINECONE_API_KEY")
    pinecone_index = os.environ.get("PINECONE_INDEX_NAME")

    if not openai_api_key:
        logging.error("OPENAI_API_KEY is not set in environment variables")
        raise SystemExit(1)

    if not telegram_bot_token:
        logging.error("TELEGRAM_BOT_TOKEN is not set in environment variables")
        raise SystemExit(1)

    if not pinecone_api_key:
        logging.error("PINECONE_BOT_TOKEN is not set in environment variables")
        raise SystemExit(1)

    if not pinecone_index:
        logging.error("PINECONE_INDEX is not set in environment variables")
        raise SystemExit(1)

    return openai_api_key, telegram_bot_token, pinecone_api_key, pinecone_index


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
    prompt_template = """Ты являешься ассистентом на крипто-хакатоне компании Latoken.
    Используй приведенные ниже фрагменты извлеченного контекста, чтобы ответить на вопрос.
    Если вы не знаете ответа, предоставьте необходимую информацию о хакатоне.
    Ты не должен упоминать наличие у себя контекста в своем ответе. 
    Пользователь  не должен знать, что у тебя есть контекст.\n
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


def create_embeddings(texts):
    embeddings = OpenAIEmbeddings()
    vectorstore = PineconeVectorStore.from_texts(texts, embeddings, index_name=os.environ.get('PINECONE_INDEX_NAME'))
    return vectorstore


def setup_bot():
    openai_api_key, telegram_bot_token, pinecone_api_key, pinecone_index = load_api_credentials()
    data = load_training_data()
    vectorstore = create_embeddings(data)
    bot = create_telegram_bot(telegram_bot_token)

    @bot.message_handler(func=lambda message: True)
    def echo_hackaton_related(message):
        rag_chain = generate_rag_chain(retriever=vectorstore.as_retriever(search_type="similarity",
                                                                          search_kwargs={"k": 3}),
                                       llm=ChatOpenAI(model="gpt-3.5-turbo-0125", api_key=openai_api_key))
        response = rag_chain.invoke(message.text)
        print(response)
        bot.reply_to(message, response)

    return bot


def main():
    bot = setup_bot()
    bot.polling()


if __name__ == "__main__":
    main()
