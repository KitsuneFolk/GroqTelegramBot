import ast
import os
from collections import defaultdict
from functools import wraps

import google.generativeai as genai
from dotenv import load_dotenv
from telegram import Update, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.ext import ApplicationBuilder, CommandHandler, CallbackQueryHandler, MessageHandler, filters, ContextTypes

# Load environment variables from the .env file
load_dotenv()

# Initialize Google's generative AI
google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    raise ValueError("GOOGLE_API_KEY is not set in the environment variables.")

genai.configure(api_key=google_api_key)

# Dictionary to store user-selected models
user_selected_models = {}

SYSTEM_PROMPT = "You are a helpful AI assistant."

# Dictionary to store conversation history
conversation_history = defaultdict(list)

# List of available models
AVAILABLE_MODELS = ["gemini-1.5-flash-exp-0827", "gemini-1.5-pro-exp-0827"]
DEFAULT_MODEL = "gemini-1.5-flash-exp-0827"

# Gemini safety settings
safety_settings = [{"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                   {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                   {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                   {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}]


# Function to create inline keyboard for model selection
def make_keyboard():
    markup = InlineKeyboardMarkup(
        [
            [InlineKeyboardButton(model, callback_data=f"['model', '{model}']")]
            for model in AVAILABLE_MODELS
        ]
    )
    return markup


# Function to handle the /start command
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    print("Bot is started")
    await update.message.reply_text("Hello! Mention me or use /ai in a group to ask something!")


# Function to call the Gemini API
async def get_ai_response(messages: list, model: str) -> str:
    print(f"get_ai_response for model: {model}")
    try:
        gemini_model = genai.GenerativeModel(model, safety_settings=safety_settings,
                                             system_instruction=SYSTEM_PROMPT)

        # Start chat with history, but don't send a message yet
        chat = gemini_model.start_chat(history=messages[:-1])  # Exclude the last message

        # Get the last user message
        last_user_message = messages[-1]['parts']
        print("history = ", messages)

        # Only send the last user message to get a response
        response = chat.send_message(last_user_message, generation_config=genai.types.GenerationConfig(
            candidate_count=1,
            max_output_tokens=1024,
            temperature=1.0,
        ))
        return response.text
    except Exception as e:
        return f"Error fetching response: {e}"


# Add this decorator function
def ensure_single_execution(func):
    processed_messages = set()

    @wraps(func)
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE):
        message_id = update.message.message_id
        if message_id in processed_messages:
            return
        processed_messages.add(message_id)
        return await func(update, context)

    return wrapper


# Function to handle messages where the bot is mentioned or /ai is used
@ensure_single_execution
async def handle_mention_or_ai(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    print("handle_mention_or_ai")
    message = update.message
    user_id = message.from_user.id
    chat_id = message.chat_id

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

            # Add the new message to the conversation history
            conversation_history[(chat_id, user_id)].append({"role": "user", "parts": prompt})

            # Call AI API with the conversation history and selected model
            response = await get_ai_response(conversation_history[(chat_id, user_id)], model)

            # Add bot's response to the conversation history
            conversation_history[(chat_id, user_id)].append({"role": "model", "parts": response})

            # Send response back to the group
            await message.reply_text(response)


# Function to handle replies to the bot's messages
async def handle_reply(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    print("handle_reply")
    message = update.message
    user_id = message.from_user.id
    chat_id = message.chat_id

    if message and message.text and message.reply_to_message and message.reply_to_message.from_user.id == context.bot.id:
        # Get the selected model for the user, default to a model if not set
        model = user_selected_models.get(user_id, DEFAULT_MODEL)

        # Add user's reply to the conversation history
        conversation_history[(chat_id, user_id)].append({"role": "user", "parts": message.text})

        # Call AI API with the conversation history and selected model
        response = await get_ai_response(conversation_history[(chat_id, user_id)], model)

        # Add bot's response to the conversation history
        conversation_history[(chat_id, user_id)].append({"role": "model", "parts": response})

        # Send response back to the group
        await message.reply_text(response)


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

    # Register message handler for replies to the bot's messages
    application.add_handler(MessageHandler(
        filters.TEXT & filters.REPLY,
        handle_reply
    ))

    # Start the bot
    application.run_polling()


if __name__ == '__main__':
    main()
