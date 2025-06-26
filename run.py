import asyncio
import logging

from aiogram import Bot, Dispatcher

from config import TOKEN
from bot.handlers import router

from database_logic.db_logic import init_db

bot = Bot(token=TOKEN)
dp = Dispatcher()

init_db()

async def main():
    dp.include_router(router)
    await dp.start_polling(bot)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print('Exit')
