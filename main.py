import logging
import os
import telebot
from dotenv import load_dotenv
from openai import OpenAI

logging.basicConfig(level=logging.INFO)


def load_api_credentials():
    load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY")
    bot_token = os.environ.get("BOT_TOKEN")

    if not api_key or not bot_token:
        logging.error("OPENAI_API_KEY or BOT_TOKEN not set in environment variables")
        raise SystemExit(1)

    return api_key, bot_token


def create_telegram_bot(bot_token):
    try:
        bot = telebot.TeleBot(bot_token)
        return bot
    except Exception as e:
        logging.error(f"Error creating Telegram bot instance: {e}")
        raise SystemExit(1)


def handle_message(update, context):
    message = update.message.text
    prompt = f'Answer the following question related to the hackathon: {message}'
    response = call_gpt4_api(prompt)
    context.bot.send_message(chat_id=update.effective_chat.id, text=response)


def call_gpt4_api(prompt): 
    client = OpenAI(api_key=load_api_credentials()[0])
    try:
        response = client.completions.create(
            prompt=prompt,
            model='gpt-3.5-turbo-instruct'
        )
        print('call_gpt4_api exit')

        return response.choices[0].text.strip()
    except Exception as e:
        logging.error(f"Error calling GPT-4 API: {e}")
        return "Sorry, I'm unable to answer that question."


api_key, bot_token = load_api_credentials()

bot = create_telegram_bot(bot_token)
#client = OpenAI()


#completion = client.chat.completions.create(
#    model="gpt-3.5-turbo",
#    messages=[
#        {"role": "system", "content": "You are a poetic assistant, skilled in explaining complex programming concepts with creative flair."},
#        {"role": "user", "content": "Compose a poem that explains the concept of recursion in programming."}
#    ]
#)

#print(completion.choices[0].message)

@bot.message_handler(func=lambda message: True)
def echo_all(message):
    print('answered')
    bot.reply_to(message, message.text)


bot.polling()

