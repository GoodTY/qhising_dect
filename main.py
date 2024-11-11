import numpy as np
import pandas as pd
from datetime import datetime
from bs4 import BeautifulSoup
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import torch.nn as nn
from transformers import BertForSequenceClassification
from torch.optim import Adam
from tqdm import tqdm
import gc
import os
import html2text
from langdetect import detect, DetectorFactory
from transformers import BertTokenizer

torch.cuda.is_available()

# GPU 장치 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

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

# 데이터셋 로드
df = pd.read_csv('dataset/ready_html_processed.csv')

# 문자 단위로 토큰화하여 BERT 입력 형식으로 변환하는 함수 정의
def char_tokenize_for_bert(urls, max_length=512):
    if not urls:
        urls = ['']
    url_text = "[SEP]".join(str(url) for url in urls)[:max_length-2]
    char_tokens = ["[CLS]"] + list(url_text) + ["[SEP]"]

    char_ids = []
    for char in char_tokens:
        if char == "[CLS]":
            char_ids.append(101)
        elif char == "[SEP]":
            char_ids.append(102)
        else:
            char_ids.append(ord(char) if ord(char) < 128 else 0)

    padding_length = max_length - len(char_ids)
    attention_mask = [1] * len(char_ids) + [0] * padding_length
    char_ids.extend([0] * padding_length)
    token_type_ids = [0] * max_length

    return {
        'input_ids': char_ids[:max_length],
        'attention_mask': attention_mask[:max_length],
        'token_type_ids': token_type_ids
    }

torch.cuda.empty_cache()
gc.collect()

# 각 URL 리스트를 문자 단위로 토큰화하여 새로운 열에 저장
df['char_tokenized_urls'] = df['extracted_urls'].apply(lambda x: char_tokenize_for_bert(x))

# 데이터 준비
input_ids = df['char_tokenized_urls'].apply(lambda x: x['input_ids']).tolist()
attention_mask = df['char_tokenized_urls'].apply(lambda x: x['attention_mask']).tolist()
token_type_ids = df['char_tokenized_urls'].apply(lambda x: x['token_type_ids']).tolist()
labels = df['label'].tolist()

del df
gc.collect()

# 학습, 검증, 테스트 데이터 분할
train_ids, test_ids, train_mask, test_mask, train_token_type, test_token_type, train_labels, test_labels = train_test_split(
    input_ids, attention_mask, token_type_ids, labels, test_size=0.2, random_state=42)

train_ids, val_ids, train_mask, val_mask, train_token_type, val_token_type, train_labels, val_labels = train_test_split(
    train_ids, train_mask, train_token_type, train_labels, test_size=0.2, random_state=42)

del input_ids, attention_mask, token_type_ids, labels
gc.collect()

# Dataset 클래스 정의 및 GPU에 직접 전달할 준비
class PhishingDataset(Dataset):
    def __init__(self, input_ids, attention_mask, token_type_ids, labels):
        # 모든 텐서를 GPU로 직접 전달
        self.input_ids = torch.tensor(input_ids, dtype=torch.long).to(device)
        self.attention_mask = torch.tensor(attention_mask, dtype=torch.long).to(device)
        self.token_type_ids = torch.tensor(token_type_ids, dtype=torch.long).to(device)
        self.labels = torch.tensor(labels, dtype=torch.long).to(device)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'token_type_ids': self.token_type_ids[idx],
            'labels': self.labels[idx]
        }

# Dataset 및 DataLoader 정의
train_dataset = PhishingDataset(train_ids, train_mask, train_token_type, train_labels)
val_dataset = PhishingDataset(val_ids, val_mask, val_token_type, val_labels)
test_dataset = PhishingDataset(test_ids, test_mask, test_token_type, test_labels)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)
test_loader = DataLoader(test_dataset, batch_size=8)

# 모델 초기화 및 GPU로 이동
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
model.to(device)

# 손실 함수와 옵티마이저 정의
criterion = torch.nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=1e-5)

# 모델 저장을 위한 디렉토리 설정
save_directory = 'phishing_model'
os.makedirs(save_directory, exist_ok=True)

# 학습 루프 100->10
epochs = 1
steps_per_epoch = len(train_loader)

for epoch in range(epochs):
    model.train()
    total_loss = 0
    print(f"\nEpoch {epoch + 1}/{epochs}")

    epoch_preds = []
    epoch_labels = []

    # 학습 단계
    for batch_idx, batch in enumerate(tqdm(train_loader, desc="Training", leave=False)):
        input_ids_batch = batch['input_ids']
        attention_mask_batch = batch['attention_mask']
        token_type_ids_batch = batch['token_type_ids']
        labels_batch = batch['labels']

        optimizer.zero_grad()

        outputs = model(input_ids_batch, attention_mask=attention_mask_batch, token_type_ids=token_type_ids_batch, labels=labels_batch)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        epoch_preds.extend(predictions.cpu().numpy())
        epoch_labels.extend(labels_batch.cpu().numpy())

        del outputs, loss, logits, predictions
        torch.cuda.empty_cache()

    # 에폭 종료 후 TRAIN 단계 수치 출력
    train_accuracy = accuracy_score(epoch_labels, epoch_preds)
    train_precision = precision_score(epoch_labels, epoch_preds, average='binary')
    train_recall = recall_score(epoch_labels, epoch_preds, average='binary')
    train_f1 = f1_score(epoch_labels, epoch_preds, average='binary')

    print(f"\n[Epoch {epoch + 1} Completed] Training Metrics:")
    print(f"Training Loss: {total_loss / len(train_loader):.4f}")
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Training Precision: {train_precision:.4f}")
    print(f"Training Recall: {train_recall:.4f}")
    print(f"Training F1-Score: {train_f1:.4f}")

    # 검증 단계
    model.eval()
    val_preds = []
    val_labels_list = []
    total_val_loss = 0

    with torch.no_grad():
        for val_batch in tqdm(val_loader, desc="Validation", leave=False):
            val_input_ids = val_batch['input_ids'].to(device)
            val_attention_mask = val_batch['attention_mask'].to(device)
            val_token_type_ids = val_batch['token_type_ids'].to(device)
            val_labels = val_batch['labels'].to(device)

            val_outputs = model(val_input_ids, attention_mask=val_attention_mask, token_type_ids=val_token_type_ids, labels=val_labels)
            val_loss = val_outputs.loss
            total_val_loss += val_loss.item()

            val_logits = val_outputs.logits
            val_predictions = torch.argmax(val_logits, dim=-1)
            val_preds.extend(val_predictions.cpu().numpy())
            val_labels_list.extend(val_labels.cpu().numpy())

    avg_val_loss = total_val_loss / len(val_loader)
    val_accuracy = accuracy_score(val_labels_list, val_preds)
    val_precision = precision_score(val_labels_list, val_preds, average='binary')
    val_recall = recall_score(val_labels_list, val_preds, average='binary')
    val_f1 = f1_score(val_labels_list, val_preds, average='binary')

    print(f"\n[Epoch {epoch + 1} Completed] Validation Metrics:")
    print(f"Validation Loss: {avg_val_loss:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    print(f"Validation Precision: {val_precision:.4f}")
    print(f"Validation Recall: {val_recall:.4f}")
    print(f"Validation F1-Score: {val_f1:.4f}")

    # 테스트 단계
    test_preds = []
    test_labels_list = []
    total_test_loss = 0

    with torch.no_grad():
        for test_batch in tqdm(test_loader, desc="Testing", leave=False):
            test_input_ids = test_batch['input_ids'].to(device)
            test_attention_mask = test_batch['attention_mask'].to(device)
            test_token_type_ids = test_batch['token_type_ids'].to(device)
            test_labels = test_batch['labels'].to(device)

            test_outputs = model(test_input_ids, attention_mask=test_attention_mask, token_type_ids=test_token_type_ids, labels=test_labels)
            test_loss = test_outputs.loss
            total_test_loss += test_loss.item()

            test_logits = test_outputs.logits
            test_predictions = torch.argmax(test_logits, dim=-1)
            test_preds.extend(test_predictions.cpu().numpy())
            test_labels_list.extend(test_labels.cpu().numpy())

    avg_test_loss = total_test_loss / len(test_loader)
    test_accuracy = accuracy_score(test_labels_list, test_preds)
    test_precision = precision_score(test_labels_list, test_preds, average='binary')
    test_recall = recall_score(test_labels_list, test_preds, average='binary')
    test_f1 = f1_score(test_labels_list, test_preds, average='binary')

    print(f"\n[Epoch {epoch + 1} Completed] Test Metrics:")
    print(f"Test Loss: {avg_test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall: {test_recall:.4f}")
    print(f"Test F1-Score: {test_f1:.4f}")

    # 모델 저장
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_save_path = os.path.join(save_directory, f'bert_phishing_model_url_epoch_{epoch + 1}_{timestamp}')
    model.save_pretrained(model_save_path)
    print(f"모델이 {model_save_path}에 저장되었습니다.")

del model
torch.cuda.empty_cache()

##################################################################################################
#HTML CONTENTS 모델 학습 시작

# 위에서 메모리 초기화 했으므로 다시 불러오기
df = pd.read_csv('dataset/ready_html_processed.csv')
# 서브워드 토크나이저 초기화
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# HTML 콘텐츠를 BERT 서브워드 토크나이저로 변환하는 함수
def subword_tokenize_for_bert(html_content, max_length=512):

    # 입력이 NaN이거나 문자열이 아닌 경우 빈 문자열로 처리
    if not isinstance(html_content, str):
        html_content = ""

    encoding = tokenizer.encode_plus(
        html_content,
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    return {
        'input_ids': encoding['input_ids'].squeeze().tolist(),
        'attention_mask': encoding['attention_mask'].squeeze().tolist(),
        'token_type_ids': encoding.get('token_type_ids', torch.zeros_like(encoding['input_ids']).squeeze()).tolist()
    }

# HTML 콘텐츠를 서브워드 방식으로 토큰화하여 새로운 열에 저장
df['subword_tokenized_html'] = df['extracted_text'].apply(lambda x: subword_tokenize_for_bert(x))

# 데이터 준비
html_input_ids = df['subword_tokenized_html'].apply(lambda x: x['input_ids']).tolist()
html_attention_mask = df['subword_tokenized_html'].apply(lambda x: x['attention_mask']).tolist()
html_token_type_ids = df['subword_tokenized_html'].apply(lambda x: x['token_type_ids']).tolist()
labels = df['label'].tolist()

# Train/Validation/Test 8대 2 비율로
train_html_ids, test_html_ids, train_html_mask, test_html_mask, train_labels, test_labels = train_test_split(
    html_input_ids, html_attention_mask, labels, test_size=0.2, random_state=42)

train_html_ids, val_html_ids, train_html_mask, val_html_mask, train_labels, val_labels = train_test_split(
    train_html_ids, train_html_mask, train_labels, test_size=0.2, random_state=42)

# PyTorch Dataset 클래스 정의
class HTMLDataset(Dataset):
    def __init__(self, input_ids, attention_mask, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.input_ids[idx], dtype=torch.long),
            'attention_mask': torch.tensor(self.attention_mask[idx], dtype=torch.long),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# DataLoader 생성 (배치 사이즈 8)
train_dataset = HTMLDataset(train_html_ids, train_html_mask, train_labels)
val_dataset = HTMLDataset(val_html_ids, val_html_mask, val_labels)
test_dataset = HTMLDataset(test_html_ids, test_html_mask, test_labels)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)
test_loader = DataLoader(test_dataset, batch_size=8)

# Device 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# BERT 모델 초기화
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
model.to(device)

# 손실 함수와 옵티마이저 정의
criterion = torch.nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=1e-5)

# 모델 저장을 위한 디렉토리 설정
save_directory = '/content/drive/MyDrive/html_phishing_model'
os.makedirs(save_directory, exist_ok=True)

# 학습 루프 100->10
epochs = 10
for epoch in range(epochs):
    model.train()
    total_loss = 0
    print(f"\nEpoch {epoch + 1}/{epochs}")

    # 학습 단계
    for batch in tqdm(train_loader, desc="Training", leave=False):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        # 순전파
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        # 역전파 및 최적화
        loss.backward()
        optimizer.step()

    # 학습 손실 출력
    avg_train_loss = total_loss / len(train_loader)
    print(f"Training Loss: {avg_train_loss:.4f}")

    # 검증 단계
    model.eval()
    val_preds = []
    val_labels_list = []
    total_val_loss = 0

    with torch.no_grad():
        for val_batch in tqdm(val_loader, desc="Validation", leave=False):
            val_input_ids = val_batch['input_ids'].to(device)
            val_attention_mask = val_batch['attention_mask'].to(device)
            val_labels = val_batch['labels'].to(device)

            val_outputs = model(val_input_ids, attention_mask=val_attention_mask, labels=val_labels)
            val_loss = val_outputs.loss
            total_val_loss += val_loss.item()

            val_logits = val_outputs.logits
            val_predictions = torch.argmax(val_logits, dim=-1)
            val_preds.extend(val_predictions.cpu().numpy())
            val_labels_list.extend(val_labels.cpu().numpy())

    avg_val_loss = total_val_loss / len(val_loader)
    val_accuracy = accuracy_score(val_labels_list, val_preds)
    val_precision = precision_score(val_labels_list, val_preds, average='binary')
    val_recall = recall_score(val_labels_list, val_preds, average='binary')
    val_f1 = f1_score(val_labels_list, val_preds, average='binary')

    print(f"\n[Epoch {epoch + 1} Completed] Validation Metrics:")
    print(f"Validation Loss: {avg_val_loss:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    print(f"Validation Precision: {val_precision:.4f}")
    print(f"Validation Recall: {val_recall:.4f}")
    print(f"Validation F1-Score: {val_f1:.4f}")

    # 모델 저장
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_save_path = os.path.join(save_directory, f'bert_html_model_html_content_epoch_{epoch + 1}_{timestamp}')
    model.save_pretrained(model_save_path)
    print(f"모델이 {model_save_path}에 저장되었습니다.")

# 테스트 단계
model.eval()
test_preds = []
test_labels_list = []
with torch.no_grad():
    for batch in tqdm(test_loader, desc="Testing", leave=False):
        test_input_ids = batch['input_ids'].to(device)
        test_attention_mask = batch['attention_mask'].to(device)
        test_labels = batch['labels'].to(device)

        outputs = model(test_input_ids, attention_mask=test_attention_mask)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        test_preds.extend(predictions.cpu().numpy())
        test_labels_list.extend(test_labels.cpu().numpy())

test_accuracy = accuracy_score(test_labels_list, test_preds)
test_precision = precision_score(test_labels_list, test_preds, average='binary')
test_recall = recall_score(test_labels_list, test_preds, average='binary')
test_f1 = f1_score(test_labels_list, test_preds, average='binary')
print(f"\nTest Metrics:")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Precision: {test_precision:.4f}, Test Recall: {test_recall:.4f}, Test F1-Score: {test_f1:.4f}")
