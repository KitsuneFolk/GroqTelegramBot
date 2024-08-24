import ast
import os

from dotenv import load_dotenv
from groq import Groq
from telegram import Update, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.ext import ApplicationBuilder, CommandHandler, CallbackQueryHandler, MessageHandler, filters, ContextTypes

# Load environment variables from the .env file
load_dotenv()

# Initialize the Groq client with the API key
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY is not set in the environment variables.")

groq_client = Groq(api_key=groq_api_key)

# Dictionary to store user-selected models
user_selected_models = {}

# List of available models
AVAILABLE_MODELS = ["llama-3.1-8b-instant", "llama-3.1-70b-versatile", "mixtral-8x7b-32768", "gemma2-9b-it"]
DEFAULT_MODEL = "llama-3.1-8b-instant"


# Function to create inline keyboard for model selection
def make_keyboard():
    markup = InlineKeyboardMarkup(
        [
            [InlineKeyboardButton(model, callback_data=f"['model', '{model}']")]
            for model in AVAILABLE_MODELS
        ]
    )
    return markup


# Function to call the Groq API with the selected model
async def get_groq_response(prompt: str, model: str) -> str:
    print("get_groq_response")
    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=model,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Error fetching response from Groq: {e}"


# Function to handle messages where the bot is mentioned
async def handle_mention(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    print("handle_mention")
    message = update.message
    user_id = message.from_user.id

    if message and message.text:
        bot_username = context.bot.username

        # Extract the text after the bot mention
        prompt = message.text.replace(f"@{bot_username}", "").strip()

        # Get the selected model for the user, default to a model if not set
        model = user_selected_models.get(user_id, DEFAULT_MODEL)

        # Call Groq API with the extracted prompt and selected model
        response = await get_groq_response(prompt, model)

        # Send response back to the group
        await message.reply_text(response)


# Function to handle messages where the bot is mentioned or /ai is used
async def handle_mention_or_ai(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    print("handle_mention_or_ai")
    message = update.message
    user_id = message.from_user.id

    if message and message.text:
        bot_username = context.bot.username
        text = message.text.strip()

        # Check if the message starts with the bot mention or /ai
        if text.startswith(f"@{bot_username}") or text.lower().startswith("/ai"):
            # Extract the prompt
            if text.startswith(f"@{bot_username}"):
                prompt = text[len(f"@{bot_username}"):].strip()
            else:  # starts with /ai
                prompt = text[3:].strip()

            # If there's no prompt, ask for one
            if not prompt:
                await message.reply_text("Please provide a prompt after mentioning me or using the /ai command.")
                return

            # Get the selected model for the user, default to a model if not set
            model = user_selected_models.get(user_id, DEFAULT_MODEL)

            # Call Groq API with the extracted prompt and selected model
            response = await get_groq_response(prompt, model)

            # Send response back to the group
            await message.reply_text(response)


# Function to handle the /start command
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    print("Bot is started")
    await update.message.reply_text("Hello! Mention me or use /ai in a group to ask something!")


# Function to handle the /model command
async def select_model(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await send_model_selection_menu(update)


# Function to send model selection menu
async def send_model_selection_menu(update: Update) -> None:
    print("send_model_selection_menu")
    reply_markup = make_keyboard()

    # Debugging print to check the keyboard structure
    print(f"Sending inline keyboard for model selection.")

    await update.message.reply_text("Please select a model:", reply_markup=reply_markup)


# Function to handle model selection
async def handle_query(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    print("handle_query")
    query = update.callback_query
    await query.answer()

    data = ast.literal_eval(query.data)

    if data[0] == 'model':
        selected_model = data[1]
        user_id = query.from_user.id
        user_selected_models[user_id] = selected_model
        await query.edit_message_text(f"Model '{selected_model}' has been selected. You can now ask your questions.")
    else:
        await query.answer("Invalid selection.")


def main() -> None:
    print("main()")
    # Get the bot token from environment variable
    telegram_bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not telegram_bot_token:
        raise ValueError("TELEGRAM_BOT_TOKEN is not set in the environment variables.")

    # Set up the application with your bot token
    application = ApplicationBuilder().token(telegram_bot_token).build()

    # Register command handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("model", select_model))

    # Register callback query handler for button presses
    application.add_handler(CallbackQueryHandler(handle_query))

    # Register message handler for mentions or /ai command
    application.add_handler(MessageHandler(
        filters.TEXT & (filters.Entity("mention") | filters.Regex(r'^/ai')),
        handle_mention_or_ai
    ))

    # Start the bot
    application.run_polling()


if __name__ == '__main__':
    main()
