import logging
import os
import langchain_community
import telebot
from dotenv import load_dotenv
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from openai import OpenAI

from langchain.chains import RetrievalQA

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


def create_telegram_bot(bot_token):
    try:
        bot = telebot.TeleBot(bot_token)
        return bot
    except Exception as e:
        logging.error(f"Error creating Telegram bot instance: {e}")
        raise SystemExit(1)


#def handle_message(update, context):
#    message = update.message.text
#    prompt = f'Answer the following question related to the hackathon: {message}'
#    response = call_gpt4_api(prompt)
#    context.bot.send_message(chat_id=update.effective_chat.id, text=response)


def call_gpt4_api(prompt):
    client = OpenAI(api_key=load_api_credentials()[0])
    try:
        response = client.completions.create(
            model='gpt-3.5-turbo-instruct',
            prompt=prompt,
        )

        return response.choices[0].text.strip()
    except Exception as e:
        logging.error(f"Error calling GPT-4 API: {e}")
        return "Sorry, I'm unable to answer that question."


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


def main():
    openai_api_key, telegram_bot_token, pinecone_api_key, pinecone_index = load_api_credentials()

    data = load_training_data()

    #vectorstore = create_embeddings(data)

    embeddings = OpenAIEmbeddings()
    vectorstore = PineconeVectorStore(
        index_name=os.environ.get('PINECONE_INDEX_NAME'), embedding=embeddings
    )

    chat = ChatOpenAI(verbose=True, temperature=0, model_name="gpt-3.5-turbo")
    qa = RetrievalQA.from_chain_type(
        llm=chat, chain_type="stuff", retriever=vectorstore.as_retriever(),
    )

    res = qa.invoke("Что такое хакатон латокен?")
    print(res)

    res = qa.invoke("Какие есть за и против работы в латокен?")
    print(res)

    #bot = create_telegram_bot(telegram_bot_token)

    #client = OpenAI()
    #completion = client.chat.completions.create(
    #    model="gpt-3.5-turbo",
    #    messages=[{"role": "system",
    #               "content": "You are a poetic assistant, "
    #                          "skilled in explaining complex programming concepts with creative flair."
    #               },
    #              {
    #                  "role": "user",
    #                  "content": "Compose a poem that explains the concept of recursion in programming."}])
    #print(completion.choices[0].message)
    #
    #related_words = ['latoken', 'hackaton']

    #    @bot.message_handler(func=lambda message: any(word.lower() in message.text.lower() for word in related_words))

    #@bot.message_handler(func=lambda message: True)
    #def echo_hackaton_related(message):
    #    bot.reply_to(message, call_gpt4_api(prompt=message.text))

    #bot.polling()


if __name__ == "__main__":
    main()
