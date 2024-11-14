import os
import csv
from bs4 import BeautifulSoup

# 데이터 파일이 들어있는 디렉토리 경로 지정
data_dir = 'dataset/phish_sample_30k'  # 전체 파일이 들어있는 디렉토리 경로

# 결과를 저장할 CSV 파일 이름 지정
output_csv = 'dataset/output.csv'  # CSV 파일 확장자 추가

# CSV 파일 열기 (쓰기 모드)
with open(output_csv, mode='w', newline='', encoding='utf-8') as csv_file:
    # CSV 작성자 생성
    csv_writer = csv.writer(csv_file)
    # CSV 파일에 컬럼 헤더 작성
    csv_writer.writerow(['Filename', 'Extracted_Content'])  # 적절한 컬럼 이름으로 수정 가능

    # 데이터 디렉토리의 모든 파일을 순회
    for root, dirs, files in os.walk(data_dir):
        for filename in files:
            # 파일 이름이 'html.txt'로 끝나는 파일만 처리
            if filename.endswith('html.txt'):
                file_path = os.path.join(root, filename)
                
                # 파일 열기 (인코딩 오류 무시)
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                        # 파일 내용 읽기
                        html_content = file.read()
                        # BeautifulSoup으로 HTML 파싱
                        soup = BeautifulSoup(html_content, 'html.parser')
                        
                        # 원하는 데이터를 추출 (예: 모든 텍스트 추출)
                        extracted_content = soup.get_text(strip=True)
                        
                        # CSV 파일에 파일 이름과 추출된 데이터 작성
                        csv_writer.writerow([filename, extracted_content])
                except Exception as e:
                    print(f"파일 '{file_path}'을(를) 처리하는 중 오류 발생: {e}")

print(f"CSV 파일 '{output_csv}'에 데이터가 저장되었습니다.")