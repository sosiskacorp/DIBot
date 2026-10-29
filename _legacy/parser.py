import os
from bs4 import BeautifulSoup
from playwright.sync_api import Page

class ContentParser:
    def __init__(self):
        # Список исключений для очистки текста (навигация, футеры и т.д.)
        self.blacklist_tags = ['nav', 'footer', 'header', 'script', 'style', 'aside']

    def extract_text_from_page(self, page: Page) -> str:
        """Извлекает основной текстовый контент со страницы"""
        html = page.content()
        soup = BeautifulSoup(html, 'html.parser')
        
        # Удаляем ненужные элементы
        for tag in soup(self.blacklist_tags):
            tag.decompose()
            
        # Пытаемся найти основной блок контента (обычно <main> или по определенным классам)
        main_content = soup.find('main') or soup.find('article') or soup.find('div', class_='content') or soup.body
        
        if main_content:
            # Очищаем текст от лишних пробелов и пустых строк
            text = main_content.get_text(separator='\n', strip=True)
            return text
        return ""

    def handle_pdf(self, page: Page, pdf_url: str) -> str:
        """
        Логика обработки PDF. 
        Playwright не читает PDF напрямую как текст, 
        поэтому здесь мы либо скачиваем его, либо используем внешний инструмент.
        """
        print(f"Обнаружен PDF: {pdf_url}")
        # В упрощенной версии мы можем просто вернуть ссылку, 
        # а в расширенной добавить библиотеку PyMuPDF (fitz)
        return f"[PDF Document]: {pdf_url} (Требуется установка PyMuPDF для парсинга)"

    def parse_course_structure(self, page: Page) -> list:
        """Анализирует страницу курса и возвращает список ссылок на материалы"""
        html = page.content()
        soup = BeautifulSoup(html, 'html.parser')
        materials = []
        
        # Ищем все ссылки, которые похожи на материалы курса
        links = soup.find_all('a', href=True)
        for link in links:
            text = link.get_text(strip=True)
            url = link['href']
            if any(keyword in text.lower() for keyword in ['лекция', 'тест', 'контрольная', 'задание', 'материал']):
                materials.append({'title': text, 'url': url})
                
        return materials
