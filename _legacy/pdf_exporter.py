import os
from fpdf import FPDF

class PDFExporter:
    def export_markdown_to_pdf(self, md_path, output_pdf_path):
        """Простая конвертация Markdown конспекта в PDF"""
        if not os.path.exists(md_path):
            return False
            
        with open(md_path, 'r', encoding='utf-8') as f:
            text = f.read()
            
        pdf = FPDF()
        pdf.add_page()
        
        # Добавляем шрифт с поддержкой кириллицы (нужен файл шрифта в папке проекта)
        # Если шрифта нет, используем стандартный (но кириллица может не отображаться)
        try:
            # Предполагаем наличие шрифта DejaVuSans в папке config/fonts/
            # pdf.add_font('DejaVu', '', 'config/fonts/DejaVuSans.ttf', unicode=True)
            # pdf.set_font('DejaVu', size=12)
            pdf.set_font("Arial", size=12) 
        except:
            pdf.set_font("Arial", size=12)

        # Заменяем Markdown-символы на простые переносы для базового PDF
        clean_text = text.replace('# ', '').replace('**', '').replace('* ', '- ')
        
        pdf.multi_cell(0, 10, txt=clean_text)
        pdf.output(output_pdf_path)
        return True
