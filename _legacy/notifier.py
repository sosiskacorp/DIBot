import requests
import json

class Notifier:
    def __init__(self, config_path="config/settings.json"):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        self.tg_token = self.config.get("telegram_token")
        self.tg_chat_id = self.config.get("telegram_chat_id")

    def send_message(self, text):
        """Отправка уведомления в Telegram"""
        if not self.tg_token or not self.tg_chat_id:
            print(f"[Notification]: {text}") # Если ТГ не настроен, просто пишем в консоль
            return

        url = f"https://api.telegram.org/bot{self.tg_token}/sendMessage"
        payload = {
            "chat_id": self.tg_chat_id,
            "text": text,
            "parse_mode": "Markdown"
        }
        try:
            requests.post(url, json=payload)
        except Exception as e:
            print(f"Ошибка отправки в Telegram: {e}")

    def send_report(self, course_name, materials_count, notes_created):
        text = (
            f"✅ *Задание выполнено!*\n\n"
            f"📚 *Курс:* {course_name}\n"
            f"📄 *Обработано материалов:* {materials_count}\n"
            f"✍️ *Создано конспектов:* {notes_created}\n"
            f"\nБот закончил работу и ушел отдыхать. 💤"
        )
        self.send_message(text)
