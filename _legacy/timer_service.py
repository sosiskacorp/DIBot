import time
import random
import json

class TimerService:
    def __init__(self, config_path="config/settings.json"):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        self.sim = self.config.get("time_simulation", {})

    def calculate_human_time(self, complexity="medium", content_length=0):
        """
        Рассчитывает время, которое потратил бы человек.
        complexity: 'easy', 'medium', 'hard'
        content_length: количество слов или символов в задаче
        """
        base_times = {
            "easy": (30, 60),    # 30-60 секунд
            "medium": (120, 300), # 2-5 минут
            "hard": (600, 1800)   # 10-30 минут
        }
        
        low, high = base_times.get(complexity, (120, 300))
        
        # Корректировка по объему контента (если он очень большой)
        if content_length > 1000:
            low += 60
            high += 120
            
        return random.randint(low, high)

    def wait_like_human(self, seconds):
        print(f"Имитация работы человека: ожидание {seconds} сек...")
        # Чтобы не ждать реально 30 минут во время тестов в режиме отладки, 
        # можно добавить коэффициент ускорения в конфиг.
        time.sleep(seconds)
