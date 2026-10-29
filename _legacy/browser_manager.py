import json
import time
import random
from playwright.sync_api import sync_playwright

class BrowserManager:
    def __init__(self, config_path="config/settings.json"):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(
            headless=self.config.get("browser_headless", True),
            args=["--disable-blink-features=AutomationControlled"]
        )
        self.context = self.browser.new_context(
            user_agent=self.config.get("user_agent")
        )
        self.page = self.context.new_page()

    def human_delay(self, base=None):
        """Имитация человеческой паузы между действиями"""
        if base is None:
            base = random.randint(1, 3)
        time.sleep(base + random.uniform(0.5, 1.5))

    def login(self, url, username, password):
        print(f"Попытка входа на {url}...")
        self.page.goto(url)
        self.human_delay()
        
        try:
            # Селекторы могут измениться, их нужно будет уточнить после анализа страницы
            # Здесь используем общие имена полей, которые часто встречаются в таких системах
            self.page.fill('input[name="username"], input[type="text"], #login', username)
            self.human_delay()
            self.page.fill('input[name="password"], input[type="password"], #password', password)
            self.human_delay()
            self.page.click('button[type="submit"], .btn-primary, #login-button')
            self.page.wait_for_load_state("networkidle")
            
            if "login" not in self.page.url:
                print("Авторизация прошла успешно!")
                return True
            else:
                print("Ошибка: страница логина всё ещё открыта.")
                return False
        except Exception as e:
            print(f"Ошибка при попытке входа: {e}")
            return False

    def navigate_to_course(self, course_url):
        print(f"Переход к курсу: {course_url}")
        self.page.goto(course_url)
        self.page.wait_for_load_state("networkidle")
        self.human_delay()

    def close(self):
        self.browser.close()
        self.playwright.stop()

if __name__ == "__main__":
    # Тестовый запуск (замените данные на свои для проверки)
    manager = BrowserManager()
    # success = manager.login("https://dispace.nstu.ru/", "your_login", "your_password")
    # print(f"Login result: {success}")
    manager.close()
