import os
import csv
from bs4 import BeautifulSoup
import chardet
from html import unescape

# 데이터 파일이 들어있는 디렉토리 경로 지정
data_dir = 'dataset/phish_sample_30k'  # 전체 파일이 들어있는 디렉토리 경로

# 결과를 저장할 CSV 파일 이름 지정
output_csv = 'dataset/output.csv'

# CSV 파일 열기 (쓰기 모드)
with open(output_csv, mode='w', newline='', encoding='utf-8') as csv_file:
    # CSV 작성자 생성
    csv_writer = csv.writer(csv_file)
    # CSV 파일에 컬럼 헤더 작성
    csv_writer.writerow(['label', 'Data'])

    # 데이터 디렉토리의 모든 파일을 순회
    for root, dirs, files in os.walk(data_dir):
        for filename in files:
            if filename.endswith('html.txt'):
                file_path = os.path.join(root, filename)
                
                # 파일의 인코딩을 감지하여 읽기
                try:
                    with open(file_path, 'rb') as file:
                        raw_data = file.read()
                        result = chardet.detect(raw_data)
                        encoding = result['encoding']
                    
                    with open(file_path, 'r', encoding=encoding, errors='ignore') as file:
                        html_content = file.read()
                except Exception as e:
                    print(f"파일 '{file_path}'을(를) 열 수 없습니다: {e}")
                    continue
                
                # BeautifulSoup으로 HTML 파싱
                soup = BeautifulSoup(html_content, 'html.parser')
                extracted_content = soup.get_text(separator='\n', strip=True)
                extracted_content = unescape(extracted_content)
                
                # CSV 파일에 파일 이름과 추출된 데이터 작성
                csv_writer.writerow(['1', extracted_content])

print(f"CSV 파일 '{output_csv}'에 데이터가 저장되었습니다.")
