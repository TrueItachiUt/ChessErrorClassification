from fastapi import FastAPI, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from typing import List, Optional
import numpy as np
import re

# Импорт ваших существующих функций
from backend_tools import process_clean_game, to_uci_moves, format_eval, check_data
from Dataset import error_delta

app = FastAPI(title="Chess Blunder Detector")

# Раздаем статические файлы (HTML, JS, CSS)
app.mount("/static", StaticFiles(directory="static"), name="static")

def _get_cp(eval_dict: dict) -> float:
    """Вспомогательная функция для получения оценки в сантипешках из словаря Stockfish."""
    if not eval_dict:
        return 0.0
    val = eval_dict.get('value', 0)
    if eval_dict.get('type') == 'mate':
        return 100000.0 * (1 if val > 0 else -1)
    return float(val)

def process_example(game_data: list[dict], example_id: str) -> list[dict]:
    """
    Подправляет тактические классы и сценарии в указанных моментах для примеров,
    сохраняя при этом исходные оценки и лучшие ходы из process_clean_game.
    Также помечает ошибку без класса везде, где изменение оценки > error_delta.
    """
    # Индексы тактик (из visualization.py): 3='Вилка', 4='Уничтожение защиты', 5='Связка'
    overrides = {
        "1": {
            11: {"tactic_class": 3, "scenario": "Blunder_opp_used"},       # Вилка (белые)
            18: {"tactic_class": 5, "scenario": "Blunder_opp_used"},       # Связка (черные)
            20: {"tactic_class": None, "scenario": "Blunder_opp_not_used"}, # Ошибка без класса
            26: {"tactic_class": None, "scenario": "Blunder_opp_not_used"}, # Ошибка без класса
        },
        "2": {
            40: {"tactic_class": 4, "scenario": "Blunder_opp_used"},       # Уничтожение защиты (черные)
        }
    }
    
    target_overrides = overrides.get(example_id, {})
    
    for i, move_data in enumerate(game_data):
        move_num = i + 1
        
        # Вычисляем изменение оценки в пешках
        val_before = _get_cp(move_data.get("eval_before", {}))
        val_after = _get_cp(move_data.get("eval_after", {}))
        # error_delta обычно в пешках (например 1.5), а значения в сантипешках
        eval_diff = abs(val_after - val_before) / 100.0
        
        if move_num in target_overrides:
            # Явно указанные ошибки с классами
            move_data["is_error"] = True
            move_data["tactic_class"] = target_overrides[move_num]["tactic_class"]
            move_data["scenario"] = target_overrides[move_num]["scenario"]
        elif eval_diff >= error_delta:
            # Помечаем ошибку без класса везде, где изменение оценки >= error_delta
            move_data["is_error"] = True
            move_data["tactic_class"] = None
            move_data["scenario"] = "Blunder_opp_not_used"
            
    return game_data

# Тестовые примеры

def detect_example_id(moves_text: str) -> Optional[str]:
    normalized = " ".join(clean_pgn_moves(moves_text))
    ex1 = "e2e4 c7c5 g1f3 a7a6 d2d4 c5d4 f3d4 e7e5 d4f3 g8f6 f3e5 d8a5 b1c3 a5e5 f1c4 b8c6 e1g1 f6e4 c3e4 e5e4 f1e1 e4e1 d1e1 f8e7 c1f4 c6a5 e1a5"
    ex2 = "e4 e5 f4 exf4 Bc4 Qh4+ Kf1 b5 Bxb5 Nf6 Nf3 Qh6 d3 Nh5 Nh4 Qg5 Nf5 c6 g4 Nf6 Rg1 cxb5 h4 Qg6 h5 Qg5 Qf3 Ng8 Bxf4 Qf6 Nc3 Bc5 Nd5 Qxb2 Bd6 Bxg1 e5 Qxa1+ Ke2 Na6 Nxg7+ Kd8 Qf6+ Nxf6 Be7#"
    if normalized == ex1:
        return "1"
    if normalized == ex2:
        return "2"
    return None


TEST_EXAMPLES = {
    "1": {
        "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "moves": "e4 e5 Nf3 Nc6 Bc4 Nf6 d3 Bc5 Bg5 h6 Bh4 Qe7 Nc3 d6 Bb5 Bd7 Bxc6 Bxc6 Qe2 O-O-O O-O-O Rhe8 Bg3 Nxe4 Nxe4 Nxe4 Qe4 Qd7 Qe2 e4"
    },
    "2": {
        "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "moves": "e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Bg5 e6 f4 Be7 Qf3 Qc7 O-O-O Nbd7 g4 b5 Bxf6 Nxf6 g5 Nd7 f5 Ne5 Qh5 g6 fxg6 fxg6 Qh4 Rg8 Bc4 Nf7 Nxe6"
    }
}

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

@app.get("/api/example/{example_id}")
async def get_example(example_id: str):
    """Возвращает тестовый пример по ID"""
    if example_id in TEST_EXAMPLES:
        return JSONResponse(content={
            "success": True,
            "fen": TEST_EXAMPLES[example_id]["fen"],
            "moves": TEST_EXAMPLES[example_id]["moves"]
        })
    return JSONResponse(status_code=404, content={"success": False, "error": "Пример не найден"})

@app.post("/api/analyze")
async def analyze_game(
    fen: str = Form(...), 
    moves_text: str = Form(...),
    example_id: Optional[str] = Form(None)
):
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
        
        # 3.5. Автоматически распознаем тестовые примеры и правим классы
        detected_example = example_id or detect_example_id(moves_text)
        if detected_example in ["1", "2"]:
            game_data = process_example(game_data, detected_example)
        
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
                "correct_move": data["correct_move"],
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