import os

from dotenv import load_dotenv
from groq import Groq
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes

# Load environment variables from the .env file
load_dotenv()

# Initialize the Groq client with the API key
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY is not set in the environment variables.")

groq_client = Groq(api_key=groq_api_key)


# Function to call the Groq API
async def get_groq_response(prompt: str) -> str:
    print("get_groq_response")
    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            # model="llama3-8b-8192",
            model="gemma2-9b-it",
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Error fetching response from Groq: {e}"


# Function to handle messages where the bot is mentioned
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    print("handle_message")
    message = update.message
    if message and message.text:
        bot_username = context.bot.username

        # Check if the bot was mentioned
        if f"@{bot_username}" in message.text:
            # Extract the text after the bot mention
            prompt = message.text.replace(f"@{bot_username}", "").strip()

            # Call Groq API with the extracted prompt
            response = await get_groq_response(prompt)

            # Send response back to the group
            await message.reply_text(response)


# Function to start the bot
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    print("Bot is started")
    await update.message.reply_text("Hello! Mention me in a group to ask something!")


def main() -> None:
    print("main()")
    # Get the bot token from environment variable
    telegram_bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not telegram_bot_token:
        raise ValueError("TELEGRAM_BOT_TOKEN is not set in the environment variables.")

    # Set up the application with your bot token
    application = ApplicationBuilder().token(telegram_bot_token).build()

    # Register start command handler
    application.add_handler(CommandHandler("start", start))

    # Register message handler for mentions
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Start the bot
    application.run_polling()


if __name__ == '__main__':
    main()
