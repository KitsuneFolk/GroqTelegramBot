import os
import logging
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters
from groq import Groq

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Set up Groq client
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Telegram bot token
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")


async def start(update: Update, context):
    logger.info(f"Start command received from user {update.effective_user.id}")
    await update.message.reply_text("Hello! Mention me in a message and I'll respond using the Groq LLM.")


async def handle_message(update: Update, context):
    logger.debug(f"Message received: {update.message.text}")

    bot_username = context.bot.username
    if bot_username and bot_username.lower() in update.message.text.lower():
        logger.info(f"Bot mentioned in message: {update.message.text}")

        # Extract the message content without the mention
        message_text = update.message.text.lower().replace(f"@{bot_username.lower()}", "").strip()

        # Send the message to Groq API
        try:
            logger.info(f"Sending message to Groq API: {message_text}")
            chat_completion = groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": message_text,
                    }
                ],
                model="llama3-8b-8192",
            )

            # Get the response from Groq
            response = chat_completion.choices[0].message.content
            logger.info(f"Received response from Groq API: {response}")

            # Send the response back to the Telegram chat
            await update.message.reply_text(response)
        except Exception as e:
            logger.error(f"An error occurred: {str(e)}")
            await update.message.reply_text(f"An error occurred: {str(e)}")
    else:
        logger.debug(f"Message received but bot not mentioned: {update.message.text}")


def main():
    logger.info("Starting bot application...")
    application = Application.builder().token(TOKEN).build()

    logger.info("Adding handlers...")
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT, handle_message))

    try:
        logger.info("Starting bot...")
        application.run_polling(drop_pending_updates=True)
    except Exception as e:
        logger.error(f"Error running bot: {str(e)}")


if __name__ == "__main__":
    main()