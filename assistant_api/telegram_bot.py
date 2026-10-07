"""
Telegram-бот для RAG-ассистента.
Запускается отдельно от консольного app.py и отвечает текстом ответа.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

from rag_pipeline import RAGPipeline

# assistant_api/.env содержит TELEGRAM_BOT_TOKEN. Родительский .env — запасной источник ключей.
load_dotenv(Path(__file__).parent / ".env")
load_dotenv(Path(__file__).parent.parent / ".env")

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
)
logger = logging.getLogger(__name__)

# Telegram отклоняет сообщения длиннее этого лимита.
TELEGRAM_MESSAGE_LIMIT = 4096

WELCOME_MESSAGE = """👋 Привет! Я AI ассистент с доступом к базе знаний LoveSelf.

Я могу ответить на ваши вопросы, используя информацию о продуктах компании.

📝 Просто отправьте мне свой вопрос, и я постараюсь помочь!

Команды:
/help - справка по использованию бота
/stats - статистика базы знаний

Пример вопроса: "Если я хочу более высокую дозу коллагена, чем в капсулах, какой морской продукт для этого подходит и как он обрабатывается для лучшего усвоения?"
"""

HELP_MESSAGE = """📖 Справка по использованию бота:

🔹 Как задать вопрос:
Просто напишите свой вопрос обычным текстом. Например:
- "Мне нужен протеиновый порошок, который можно добавлять в кашу и фруктовые смузи, без сильного вкуса. Какой это порошок и как правильно питаться, если я ем меньше на Wegovy?"
- "Когда человек, принимающий препарат GLP-1, стал меньше есть и потребляет меньше клетчатки, какие продукты в линейке покрывают потребность в белке и клетчатке?"
- "Я переживаю менопаузу, теряю мышечную массу, и также хочу улучшить концентрацию. Есть ли продукт для этого, который выходит за рамки обычной спортивной добавки?"

🔹 Доступные команды:
/start - приветствие
/help - эта справка
/stats - статистика базы знаний

🔹 Как работает бот:
1. Я ищу интересующую вас информацию в базе знаний
2. Использую найденную информацию для формирования ответа
3. Отвечаю только на основе реальных данных о продуктах компании
"""

ERROR_MESSAGE = "❌ Извините, произошла ошибка при обработке вашего вопроса. Попробуйте еще раз."


def _message_chunks(text: str) -> list[str]:
    """Делит длинный ответ на части, которые Telegram согласен отправить."""
    if len(text) <= TELEGRAM_MESSAGE_LIMIT:
        return [text]
    return [
        text[start:start + TELEGRAM_MESSAGE_LIMIT]
        for start in range(0, len(text), TELEGRAM_MESSAGE_LIMIT)
    ]


class TelegramBot:
    """Команды бота и ответы через RAGPipeline."""

    def __init__(self, pipeline: RAGPipeline):
        self.pipeline = pipeline
        logger.info("TelegramBot инициализирован")

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /start."""
        user = update.effective_user
        await update.message.reply_text(WELCOME_MESSAGE)
        logger.info("Пользователь %s (%s) начал работу с ботом", user.id, user.first_name)

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /help."""
        await update.message.reply_text(HELP_MESSAGE)

    async def stats_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /stats."""
        try:
            stats = await asyncio.to_thread(self.pipeline.get_stats)
            vector_stats = stats["vector_store"]
            cache_stats = stats["cache"]
            stats_message = f"""📊 Статистика:

📚 База знаний:
  • Коллекция: {vector_stats['name']}
  • Документов: {vector_stats['count']}

💾 Кеш:
  • Записей: {cache_stats['total_entries']}

🤖 Модель: {stats['model']}

✅ Бот готов отвечать на ваши вопросы!
"""
            await update.message.reply_text(stats_message)
        except Exception as e:
            logger.error("Ошибка получения статистики: %s", e)
            await update.message.reply_text(f"❌ Ошибка получения статистики: {e}")

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик текстового вопроса. В чат уходит только текст ответа."""
        user = update.effective_user
        user_query = update.message.text
        logger.info("Получен вопрос от %s (ID: %s): %s", user.first_name, user.id, user_query)

        await update.message.chat.send_action(action=ChatAction.TYPING)

        try:
            result = await asyncio.to_thread(self.pipeline.query, user_query)
            for chunk in _message_chunks(result["answer"]):
                await update.message.reply_text(chunk)
            logger.info("Ответ отправлен пользователю %s", user.id)
        except Exception as e:
            logger.error("Ошибка обработки сообщения: %s", e, exc_info=True)
            await update.message.reply_text(ERROR_MESSAGE)


def main():
    """Запуск бота. Процесс остаётся работать и отвечает на сообщения."""
    if not os.getenv("OPENAI_API_KEY"):
        print("Ошибка: OPENAI_API_KEY не установлен")
        sys.exit(1)
    if not os.getenv("TELEGRAM_BOT_TOKEN"):
        print("Ошибка: TELEGRAM_BOT_TOKEN не установлен")
        sys.exit(1)

    print("Инициализация RAG pipeline...")
    pipeline = RAGPipeline(
        collection_name="loveself_collection",
        cache_db_path="api_rag_cache.db",
        data_file="data/loveself-products.txt",
        model="gpt-4o-mini",
    )

    bot = TelegramBot(pipeline)
    application = Application.builder().token(os.environ["TELEGRAM_BOT_TOKEN"]).build()
    application.add_handler(CommandHandler("start", bot.start_command))
    application.add_handler(CommandHandler("help", bot.help_command))
    application.add_handler(CommandHandler("stats", bot.stats_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, bot.handle_message))

    print("Бот запущен. Остановка: Ctrl+C")
    application.run_polling()


if __name__ == "__main__":
    main()
