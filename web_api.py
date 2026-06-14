from fastapi import FastAPI, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from typing import List
import numpy as np

# Импорт ваших существующих функций
from backend_tools import process_clean_game, to_uci_moves, format_eval, check_data

app = FastAPI(title="Chess Blunder Detector")

# Раздаем статические файлы (HTML, JS, CSS)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def read_root():
    return FileResponse("static/index.html")

@app.post("/api/analyze")
async def analyze_game(fen: str = Form(...), moves_text: str = Form(...)):
    try:
        # 1. Разбиваем строку ходов на список
        raw_moves = moves_text.strip().split()
        if not raw_moves:
            return {"error": "Список ходов пуст"}
        
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
            
        return {"success": True, "data": formatted_data}
        
    except ValueError as e:
        return {"success": False, "error": str(e)}
    except Exception as e:
        return {"success": False, "error": f"Внутренняя ошибка: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    # Запуск сервера. Модели загрузятся при первом запросе благодаря lazy load в backend_tools
    uvicorn.run(app, host="0.0.0.0", port=8000)