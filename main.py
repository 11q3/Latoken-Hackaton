import logging
import os
import telebot

from dotenv import load_dotenv
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.chains.llm import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.prompts import PromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
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


#    def handle_message(update, context):
#    message = update.message.text
#    prompt = f'Answer the following question related to the hackathon: {message}'
#    response = call_gpt4_api(prompt)
#    context.bot.send_message(chat_id=update.effective_chat.id, text=response)

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

    vectorstore = create_embeddings(data)

    llm = ChatOpenAI(verbose=True, temperature=0, model_name="gpt-3.5-turbo")

    condense_question_prompt = PromptTemplate.from_template(template)

    qa_system_prompt = """Ты - дружелюбный чатбот ассистент в телеграмм группе компании "Латокен". 

        Используй данный тебе контекст, чтобы максимально помочь пользователям этой телеграмм группы,
        получить ответы на их вопросы, связанные с компанией Латокен, или Хакатоном который она проводит.
        
        Контекст:
        {context}
        \"""
        Вопрос:
        \"""
        Твой ответ:"""

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retriieva

    question_generator = LLMChain(
        llm=llm,
        prompt=condense_question_prompt
    )

    chat_history = []

    qa_prompt = HumanMessagePromptTemplate.from_template(template)

    chat_prompt = ChatPromptTemplate.from_messages([qa_prompt])

    doc_chain = load_qa_chain(
        llm=llm,
        chain_type="stuff",
    )

    qa = ConversationalRetrievalChain(
        retriever=vectorstore.as_retriever(),
        combine_docs_chain=doc_chain(),
        question_generator=question_generator()
    )

    bot = create_telegram_bot(telegram_bot_token)

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

    #related_words = ['latoken', 'hackaton']
    #@bot.message_handler(func=lambda message: any(word.lower() in message.text.lower() for word in related_words))


    @bot.message_handler(func=lambda message: True)
    def echo_hackaton_related(message):

        result = qa(
            {'inputs': message, 'chat_history': chat_history}
        )

        print(result)

        history = (result['query'], result['result'])
        chat_history.append(history)

        if 'result' in result and result['result']:
            bot.reply_to(message, result['result'])
        else:
            bot.reply_to(message, "Я не могу ответить на этот вопрос..")

    bot.polling()


if __name__ == "__main__":
    main()
