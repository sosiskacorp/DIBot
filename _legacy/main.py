import os
from browser_manager import BrowserManager
from student_brain import StudentBrain
from parser import ContentParser
from notifier import Notifier

def run_student_bot(course_url, username, password):
    browser = BrowserManager()
    brain = StudentBrain()
    parser = ContentParser()
    notifier = Notifier()
    
    notes_created = 0
    materials_found = 0
    course_name = "NSTU Course"

    try:
        if not browser.login("https://dispace.nstu.ru/", username, password):
            notifier.send_message("❌ Ошибка авторизации в DIBot")
            return

        browser.navigate_to_course(course_url)
        materials = parser.parse_course_structure(browser.page)
        materials_found = len(materials)
        
        for item in materials:
            title = item['title']
            url = item['url']
            
            if "тест" in title.lower() or "контрольная" in title.lower():
                continue
            
            browser.page.goto(url)
            browser.page.wait_for_load_state("networkidle")
            
            content = parser.extract_text_from_page(browser.page)
            if content:
                brain.create_lecture_notes(course_name, title, content)
                notes_created += 1
                
    except Exception as e:
        notifier.send_message(f"⚠️ Критическая ошибка бота: {e}")
    finally:
        browser.close()
        notifier.send_report(course_name, materials_found, notes_created)

if __name__ == "__main__":
    USER = "your_username"
    PASS = "your_password"
    COURSE_URL = "https://dispace.nstu.ru/course/view.php?id=XXX"
    run_student_bot(COURSE_URL, USER, PASS)
