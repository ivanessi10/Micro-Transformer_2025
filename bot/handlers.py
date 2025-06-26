from aiogram import Router, F
from aiogram.filters import CommandStart, Command, StateFilter
from aiogram.types import Message
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.context import FSMContext

from models.data_answer import GemmaModel

import database_logic.db_logic as db

model = GemmaModel()

router = Router()

class Query(StatesGroup):
    query_waiting_for_text = State()
    new_dialogue_waiting_for_name = State()
    in_dialogue = State()
    waiting_for_open_dialog_name = State()
    deleting_dialogue = State()

@router.message(Command("cancel"))
@router.message(F.text.casefold() == "cancel")
async def cancel_handler(message: Message, state: FSMContext) -> None:
    current_state = await state.get_state()
    if current_state is None:
        return

    await state.clear()
    await message.answer(
        "cancelled"
    )

@router.message(CommandStart())
async def start(message: Message):
    await message.reply("""Привет! 
Я -- твой помощник, то, что я умею ты можешь подробно узнать из команды /help.""")

@router.message(StateFilter(None), Command('query'))
async def query(message: Message, state: FSMContext):
    await message.answer("Введите текст запроса:")

    await state.set_state(Query.query_waiting_for_text)

@router.message(Query.query_waiting_for_text)
async def text(message: Message, state: FSMContext):
    query = message.text
    await message.answer(model.generate_response(query))


    await message.answer("Чем я еще могу помочь?")

@router.message(StateFilter(None), Command('new_dialogue'))
async def new_dialogue(message: Message, state: FSMContext):
    await message.answer("Введите имя своего диалога:")
    
    await state.set_state(Query.new_dialogue_waiting_for_name)

@router.message(Query.new_dialogue_waiting_for_name)
async def new_dialogue_add(message: Message, state: FSMContext):
    name = message.text 
    user_id = message.from_user.id

    db.add_dialogue(user_id ,name)

    await state.clear()
    await message.answer("Диалог успешно создан!")

@router.message(StateFilter(None), Command('get_dialogues'))
async def get_dialogues(message: Message, state: FSMContext):
    user_id = message.from_user.id

    names = db.get_user_dialogues(user_id)    

    await message.answer(' '.join(names))

    await message.answer('Выше список ваших диалогов')

@router.message(StateFilter(None), Command('open_dialogue'))
async def open_dialogue(message: Message, state: FSMContext):
    await message.answer('Введите название диалога, которых хотите открыть:')

    await state.set_state(Query.waiting_for_open_dialog_name)

@router.message(Query.waiting_for_open_dialog_name)
async def check_dialogue(message: Message, state: FSMContext):
    name = message.text
    user_id = message.from_user.id

    
    if name in db.get_user_dialogues(user_id):
        await message.answer(f'Диалог {name} открыт. Введите ваш запрос:')
        await state.update_data(name=message.text)

        await state.set_state(Query.in_dialogue)
    else:
        await message.answer('Такого диалоге у вас нет')

        await state.clear()

@router.message(Command('remind_dialogue'), Query.in_dialogue)
async def remind_dialog(message: Message, state: FSMContext):
    user_id = message.from_user.id
    data = await state.get_data()
    name = data.get("name")

    dialogue_context = db.get_full_dialogue(user_id=user_id, name=name)

    if not dialogue_context:
        await message.answer("Диалог пуст.")
        return

    last_turns = dialogue_context[-4:] 
    await message.answer("\n".join(last_turns))


@router.message(Query.in_dialogue)
async def open_dialogue_context(message: Message, state: FSMContext):
    prompt = message.text
    user_id = message.from_user.id
    data = await state.get_data()
    name = data.get("name")

    query = ' '.join(db.get_full_dialogue(user_id, name))
    query += f" User: {prompt}"

    answer_text = model.generate_response(query)

    db.add_phrase(user_id, name, prompt, answer_text)

    await message.answer(answer_text)


@router.message(StateFilter(None), Command('delete_dialogue'))
async def delete_dialogue_name(message: Message, state: FSMContext):
    await message.answer('Введите название диалога, который хотите удалить:')

    await state.set_state(Query.deleting_dialogue)

@router.message(Query.deleting_dialogue)
async def delete_dialog(message: Message, state: FSMContext):
    user_id = message.from_user.id
    name = message.text

    deleted = db.delete_dialogue(user_id, name)

    if deleted:
        await message.answer(f'Диалог {name} успешно удален')
    else:
        await message.answer(f'Диалог {name} не найден')

        await state.clear()

@router.message(StateFilter(None), Command('help'))
async def help(message: Message, state: State):
    await message.answer(
            """
            - по команде /query войти в режим одноразового диалога
            - по команде /new_dialogue -- создать новый диалог
            - по команде /get_dialogues -- вывести список ваших диалогов
            - по команде /open_dialogue -- открыть существующий диалог, если его нет -- вылезет предупреждение
            - по команде /delete_dialog -- можете удалить свой диалог
            - по команде /cancel или просто написал cancel вы можете выйти из любого режима(режима чата, например)
            - по команде /remind_dialogue -- напомнит вам последние 4 фразы из диалога
            """
            )
    
