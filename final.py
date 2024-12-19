import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from transformers import BertForSequenceClassification, BertTokenizer
from tqdm import tqdm

# Device 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Tokenizer 초기화
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 데이터셋 로드
df = pd.read_csv('dataset/ready_html_processed.csv')

# NaN 값을 빈 문자열로 대체
df['extracted_text'] = df['extracted_text'].fillna('')

# 데이터 준비: extracted_text를 토큰화
input_ids = df['extracted_text'].apply(
    lambda x: tokenizer.encode(
        x,
        add_special_tokens=True,
        max_length=512,
        padding='max_length',
        truncation=True
    )
).tolist()

# Attention mask 준비
attention_mask = df['extracted_text'].apply(
    lambda x: tokenizer.encode_plus(
        x,
        add_special_tokens=True,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_attention_mask=True
    )['attention_mask']
).tolist()

# 라벨
labels = df['label'].tolist()

# PyTorch Dataset 정의
class PhishingDataset(Dataset):
    def __init__(self, input_ids, attention_mask, labels):
        self.input_ids = torch.tensor(input_ids, dtype=torch.long)
        self.attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': self.labels[idx]
        }

# 테스트 데이터 준비
test_dataset = PhishingDataset(input_ids, attention_mask, labels)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# URL 모델과 HTML 모델 로드
url_model = BertForSequenceClassification.from_pretrained('phishing_model/bert_phishing_model_url')
url_model.to(device)
url_model.eval()

html_model = BertForSequenceClassification.from_pretrained('html_phishing_model/bert_html_model_html_content')
html_model.to(device)
html_model.eval()

# 앙상블 결과 저장
final_predictions = []
true_labels = []

# 테스트 데이터에서 예측 수행
with torch.no_grad():
    for batch in tqdm(test_loader, desc="Testing"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # URL 모델 예측
        url_outputs = url_model(input_ids=input_ids, attention_mask=attention_mask)
        url_logits = url_outputs.logits
        url_preds = torch.softmax(url_logits, dim=1)

        # HTML 모델 예측
        html_outputs = html_model(input_ids=input_ids, attention_mask=attention_mask)
        html_logits = html_outputs.logits
        html_preds = torch.softmax(html_logits, dim=1)

        # 앙상블: 두 모델의 평균 확률
        ensemble_preds = (url_preds + html_preds) / 2
        final_batch_preds = torch.argmax(ensemble_preds, dim=1)

        # 저장
        final_predictions.extend(final_batch_preds.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

# 성능 평가
accuracy = accuracy_score(true_labels, final_predictions)
precision = precision_score(true_labels, final_predictions, average='binary')
recall = recall_score(true_labels, final_predictions, average='binary')
f1 = f1_score(true_labels, final_predictions, average='binary')

print("\nEnsemble Model Performance:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
