import os
import json
from datetime import datetime
from llm_client import LLMClient

class StudentBrain:
    def __init__(self, config_path="config/settings.json"):
        self.llm = LLMClient(config_path)
        with open(config_path, 'r') as f:
            self.config = json.load(f)

    def create_lecture_notes(self, course_name: str, lecture_title: str, content: str) -> str:
        """Превращает сырой текст лекции в структурированный конспект"""
        system_prompt = (
            "Ты — прилежный студент. Твоя задача: изучить предоставленный материал "
            "и составить подробный, структурированный конспект в формате Markdown. "
            "Выдели главные определения, формулы и ключевые идеи. "
            "Используй заголовки, списки и таблицы для наглядности. "
            "Пиши на русском языке."
        )
        
        prompt = f"Курс: {course_name}\nЛекция: {lecture_title}\n\nМатериал:\n{content}"
        
        print(f"Генерация конспекта для '{lecture_title}'...")
        notes = self.llm.generate(prompt, system_prompt=system_prompt)
        
        # Сохранение конспекта
        file_path = self._save_notes(course_name, lecture_title, notes)
        return file_path

    def solve_assignment(self, lecture_notes: str, problem_text: str) -> str:
        """Решает задачу, опираясь на конспекты лекций"""
        system_prompt = (
            "Ты — студент, который решает контрольную работу. "
            "Используй предоставленные конспекты лекций для решения задачи. "
            "Решение должно быть пошаговым, с пояснениями, как в тетради. "
            "Если в конспектах нет нужной информации, используй свои знания, "
            "но отметь это. Ответ должен быть оформлен аккуратно."
        )
        
        prompt = f"Конспекты:\n{lecture_notes}\n\nЗадача:\n{problem_text}"
        
        print("Решение задачи...")
        solution = self.llm.generate(prompt, system_prompt=system_prompt)
        return solution

    def _save_notes(self, course_name: str, lecture_title: str, content: str) -> str:
        # Создаем папку курса, если её нет
        course_dir = os.path.join("data/notebooks", course_name.replace(" ", "_"))
        os.makedirs(course_dir, exist_ok=True)
        
        # Очистка названия файла от запрещенных символов
        safe_title = "".join([c for c in lecture_title if c.isalnum() or c in (' ', '_', '-')]).strip()
        file_path = os.path.join(course_dir, f"{safe_title}.md")
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"# {lecture_title}\n\nДата создания: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
            f.write(content)
            
        return file_path

if __name__ == "__main__":
    # Пример работы
    brain = StudentBrain()
    test_content = "Линейная алгебра: Матрица - это прямоугольная таблица чисел. Определитель матрицы 2x2 вычисляется как ad-bc."
    path = brain.create_lecture_notes("Linear Algebra", "Introduction to Matrices", test_content)
    print(f"Конспект сохранен в: {path}")
