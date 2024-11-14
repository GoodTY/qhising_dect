# train_html.csv에서 Data -> urls, contents 추출

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from langdetect import detect, DetectorFactory
import html2text

# CSV 파일 경로
input_file = 'dataset/train_html.csv'
output_file = 'dataset/ready_html.csv'

# CSV 파일 불러오기 (문제 있는 행을 건너뜁니다)
df = pd.read_csv(input_file, engine='python', on_bad_lines='skip')

# 'Category' 열에서 'spam'은 1로, 'ham'은 0으로 변경
df['Category'] = np.where(df['Category'] == 'spam', 1, 0)

# 변경된 데이터를 새로운 CSV 파일로 저장
df.to_csv(output_file, index=False)

# 데이터셋 로드
df = pd.read_csv('dataset/ready_html.csv')

# 'Category' 열을 'label'로 변경
df.rename(columns={'Category': 'label'}, inplace=True)

# URL 추출 함수 정의 (타입 체크 추가)
def extract_urls(html_content):
    try:
        if pd.isna(html_content):
            return []
        html_content = str(html_content)
        soup = BeautifulSoup(html_content, 'html.parser')
        return [tag['href'] for tag in soup.find_all(['link', 'a'], href=True)]
    except Exception as e:
        print(f"Error processing content: {str(e)}")
        return []

# Data 컬럼의 NaN 값을 빈 문자열로 대체
df['Data'] = df['Data'].fillna('')

# "Data" 열에서 URL을 추출하여 새로운 열에 저장
df['extracted_urls'] = df['Data'].apply(extract_urls)

# HTML Contents 추출 함수 정의 URL,IMAGE 등 제외
def extract_text(html_content):
    try:
        if pd.isna(html_content):
            return ""

        # HTML2Text 초기화
        h = html2text.HTML2Text()
        h.ignore_links = True  # 링크 무시
        h.ignore_images = True  # 이미지 무시
        h.ignore_tables = True  # 테이블 무시
        h.ignore_emphasis = True  # 강조 무시
        h.ignore_anchors = True  # 앵커 무시

        # HTML을 텍스트로 변환
        text = h.handle(html_content)

        # 불필요한 공백 제거
        text = ' '.join(text.split())

        return text
    except Exception as e:
        print(f"Error extracting text: {str(e)}")
        return ""

# 텍스트가 영어인지 확인하는 함수 정의
def is_english(text):
    try:
        if not text:
            return False
        return detect(text) == 'en'
    except:
        return False
    
# "Data" 열에서 URL과 텍스트를 추출하여 새로운 열에 저장
df['extracted_text'] = df['Data'].apply(extract_text)

# CSV 형식으로 저장
df.to_csv('dataset/ready_html_processed.csv', index=False)