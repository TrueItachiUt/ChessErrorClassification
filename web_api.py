from fastapi import FastAPI, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from typing import List
import numpy as np
import re

# Импорт ваших существующих функций
from backend_tools import process_clean_game, to_uci_moves, format_eval, check_data

app = FastAPI(title="Chess Blunder Detector")

# Раздаем статические файлы (HTML, JS, CSS)
app.mount("/static", StaticFiles(directory="static"), name="static")

def clean_pgn_moves(moves_text: str) -> list:
    """
    Очищает PGN от номеров ходов и комментариев.
    Пример: "1.e4 e5 2.Nf3 Nc6" -> ["e4", "e5", "Nf3", "Nc6"]
    """
    # Удаляем номера ходов (цифры с точкой, например "1.", "12...")
    cleaned = re.sub(r'\d+\.+', '', moves_text)
    # Удаляем комментарии в фигурных скобках {comment}
    cleaned = re.sub(r'\{[^}]*\}', '', cleaned)
    # Удаляем комментарии в точках с запятой ;comment
    cleaned = re.sub(r';[^\n]*', '', cleaned)
    # Удаляем результаты игры (1-0, 0-1, 1/2-1/2, *)
    cleaned = re.sub(r'(1-0|0-1|1/2-1/2|\*)', '', cleaned)
    # Разбиваем на отдельные ходы
    moves = cleaned.split()
    # Фильтруем пустые строки
    moves = [m.strip() for m in moves if m.strip()]
    return moves

def convert_numpy_types(obj):
    """
    Рекурсивно преобразует numpy типы в стандартные Python типы для JSON сериализации.
    """
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

@app.get("/")
async def read_root():
    return FileResponse("static/index.html")

@app.post("/api/analyze")
async def analyze_game(fen: str = Form(...), moves_text: str = Form(...)):
    try:
        # 1. Очищаем PGN от номеров ходов и разбиваем на список
        raw_moves = clean_pgn_moves(moves_text)
        if not raw_moves:
            return {"success": False, "error": "Список ходов пуст"}
        
        # 2. Валидация и конвертация в UCI (поддерживает и UCI, и SAN/PGN)
        uci_moves_list = to_uci_moves([fen], [raw_moves])[0]
        check_data(fen, uci_moves_list)
        
        # 3. Запуск вашего пайплайна анализа
        game_data = process_clean_game(uci_moves_list, fen)
        
        # 4. Форматируем вывод для удобного JSON на фронтенде
        formatted_data = []
        for i, data in enumerate(game_data):
            formatted_data.append({
                "move_number": i + 1,
                "side": data["side"],
                "uci": data["uci"],
                "eval_before": format_eval(data["eval_before"]),
                "eval_after": format_eval(data["eval_after"]),
                "is_error": data["is_error"],
                "tactic_class": data["tactic_class"],
                "correct_move": data["correct_move"]
            })
        
        # 5. Преобразуем все numpy типы в стандартные Python типы
        result = {"success": True, "data": formatted_data}
        return convert_numpy_types(result)
        
    except ValueError as e:
        return {"success": False, "error": str(e)}
    except Exception as e:
        return {"success": False, "error": f"Внутренняя ошибка: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    # Запуск сервера. Модели загрузятся при первом запросе благодаря lazy load в backend_tools
    uvicorn.run(app, host="0.0.0.0", port=8000)