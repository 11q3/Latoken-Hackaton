import logging
import os
import telebot
from dotenv import load_dotenv
import openai
from openai import OpenAI
import requests.exceptions
import time

# Set up logging
logging.basicConfig(level=logging.INFO)

# Define constants
HACKATHON_KEYWORDS = ['hackathon', 'latoken', 'deliver', 'about']
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 5

# Load API credentials from environment variables
def load_api_credentials():
    print('load_api_credentials')
    load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY")
    bot_token = os.environ.get("BOT_TOKEN")

    if not api_key or not bot_token:
        logging.error("OPENAI_API_KEY or BOT_TOKEN not set in environment variables")
        raise SystemExit(1)

    print('load_api_credentials exit')
    return api_key, bot_token

# Create Telegram bot instance
def create_telegram_bot(bot_token):
    try:
        bot = telebot.TeleBot(bot_token)
        print('create_telegram_bot exit')
        return bot
    except Exception as e:
        logging.error(f"Error creating Telegram bot instance: {e}")
        raise SystemExit(1)


# Handle incoming messages
def handle_message(update, context):
    print('handle_message')
    message = update.message.text
    #if any(keyword in message.lower() for keyword in HACKATHON_KEYWORDS):
        # Generate a prompt for GPT-4
    prompt = f'Answer the following question related to the hackathon: {message}'
        # Call GPT-4 API to generate a response
    response = call_gpt4_api(prompt)
        # Send response back to the user
    context.bot.send_message(chat_id=update.effective_chat.id, text=response)
    print('handle_message exit')

# Call GPT-4 API to generate a response
def call_gpt4_api(prompt):
    print('call_gpt4_api')
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

# Main function
def main():
    api_key, bot_token = load_api_credentials()
    bot = create_telegram_bot(bot_token)

    # Set up message handler
    print('++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    @bot.message_handler(func=lambda message: True)
    def handle_text_message(message):
        print('------------------------------------------------------------------------------------------------------------------------------')
        handle_message(message, bot)
    print('------------------------------------------------------')

    # Start polling
    bot.polling(timeout=1)

if __name__ == "__main__":
    main()