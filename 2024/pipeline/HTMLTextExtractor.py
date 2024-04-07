import re

from bs4 import BeautifulSoup
import requests

class HTMLTextExtractor:
    def __init__(self, url):
        self.url = url

    def fetch_html(self):
        HEADERS = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.71 Safari/537.36'}
        response = requests.get(self.url, headers=HEADERS)
        if response.status_code == 200:
            return response.text
        else:
            print("Failed to fetch HTML:", response.status_code)
            return None

    def extract_text(self):
        html_content = self.fetch_html()
        if html_content:
            soup = BeautifulSoup(html_content, 'html.parser')
            text = soup.get_text()
            text = re.sub(r'\n\s*\n', '\n', text)
            lines = text.split('\n')
            index_start=-1
            index_end = -1
            lines = [line.strip() for line in lines if line.strip()]
            for i, line in enumerate(lines):
                if 'My Lords,' in line:
                    index_start = i
                    break
            for i, line in enumerate(lines):
                if 'Crown Copyright' in line:
                    index_end = i
                    break
            if index_start != -1 & index_end != -1:
                text =  '\n'.join(lines[max(0, index_start - 2):index_end])

            match = re.search(r'UKHL/\d+/\d+', self.url)
            if match:
                file_path = "data/UKHL_txt/" + re.sub(r'[^\w\s-]', '', match.group()) + ".txt"

            else:
                print("No valid pattern found in the URL.")

            with open(file_path, "w", encoding="utf-8") as file:
                file.write(text)
            print("Text saved to", file_path)
            return file_path
        else:
            return None

'''Example usage:
url = "https://www.bailii.org/uk/cases/UKHL/2001/2.html"

HTMLTextExtractor(url).extract_text()'''