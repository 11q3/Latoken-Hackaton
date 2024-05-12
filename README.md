# LATOKEN Hackathon Telegram Bot

This Telegram bot is designed to respond to questions related to the LATOKEN hackathon using AI. It utilizes the GPT-4 model from OpenAI to generate responses.

## Instructions

### Setting Up Environment

1. Clone this repository.
2. Navigate to the project directory.
3. Create a Python virtual environment:
    ```bash
    python3 -m venv env
    ```
4. Activate the virtual environment:
    ```bash
    source env/bin/activate  # On Linux/Mac
    env\Scripts\activate  # On Windows
    ```
5. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Running the Bot

1. Obtain a Telegram Bot Token from [@BotFather](https://t.me/BotFather) on Telegram.
2. Replace `"YOUR_TELEGRAM_BOT_TOKEN"` in the script with your actual bot token.
3. Set up your OpenAI API key and replace `"YOUR_OPENAI_API_KEY"` in the script.
4. Run the bot script:
    ```bash
    python bot.py
    ```

### Usage

1. Add the bot to your Telegram group.
2. Start typing your questions related to the LATOKEN hackathon.
3. The bot will respond with AI-generated answers.
