# LATOKEN Hackathon Telegram Bot
## Description
The bot is built using the Langchain library and the OpenAI API for language processing. 
The bot is trained on a set of data (training_data.txt) that is loaded and processed using the Langchain text splitter. 
The processed text is then embedded into vectors using the OpenAIEmbeddings module and 
stored in a Pinecone vector database.

The bot uses a Retrieval-Augmented Generation (RAG) chain to generate responses to user messages. 
The RAG chain consists of a retriever, a language model, and a prompt template.
The retriever retrieves relevant context from the vector database based on the user's message. 
The language model generates a response using the retrieved context and the prompt template.

The bot is designed to handle messages related to the cryptohackathon or the company Latoken. 
If the bot is unable to answer a question, it will inform the user that it is not sure of the answer but can refer the
user to the organizers of the hackathon for more information.


### Prerequisites
- Python 3.8 or higher
- OpenAI API key
- Pinecone account and index
- Telegram bot token
- VPN, if you are trying to use a bot from a country that is not available for OpenAI

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
5. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

### Running the Bot

1. Obtain a Telegram Bot Token from [@BotFather](https://t.me/BotFather) on Telegram.
2. Obtain a Pinecone API key and create an index:
   - Create an account at https://www.pinecone.io/
   - Find your API key in your Pinecone Console
   - Create a Pinecone Index (You can choose embedding model "text-embedding-ada-002"") for example.
   - Remember the chosen name, you will need it later
3. Update the variables in the .env file located at the project root directory with your credentials:
   - Add your Telegram bot token after BOT_TOKEN=
   - Add your OpenAI API key after OPENAI_API_KEY=
   - Add your Pinecone API key after PINECONE_API_KEY=
   - Add your Pinecone Index name after PINECONE_INDEX_NAME=
4. Run the bot script:
    ```bash
    python main.py
    ```

The bot will start and wait for messages in the specified Telegram chat. 
When a user sends a message, the bot will process the message, search the training data using Pinecone, and 
generate a response using Langchain and OpenAI.

### Usage

1. Add the bot to your Telegram group.
2. Start typing your questions related to the LATOKEN hackathon.
3. The bot will respond with AI-generated answers.
