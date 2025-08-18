
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from sklearn.preprocessing import LabelEncoder
import numpy as np
from underthesea import word_tokenize
import warnings
warnings.filterwarnings('ignore')
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import uvicorn
import re
import threading
import time

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class Config:
    """Configuration class for hyperparameters"""
    model_name = "demdecuong/vihealthbert-base-word"
    max_length = 256
    batch_size = 16
    learning_rate = 2e-5
    weight_decay = 0.01
    num_epochs = 20
    warmup_steps = 100
    dropout_rate = 0.3
    hidden_size = 128
    patience = 5
    gradient_clip = 1.0

class EnhancedDiseaseClassifier(nn.Module):
    """Enhanced classifier for disease prediction"""
    def __init__(self, pretrained_model, num_labels, config):
        super(EnhancedDiseaseClassifier, self).__init__()
        self.bert = pretrained_model
        self.config = config

        # Freeze some layers for better transfer learning
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False

        # Enhanced classifier head
        self.dropout1 = nn.Dropout(config.dropout_rate)
        self.fc1 = nn.Linear(self.bert.config.hidden_size, config.hidden_size)
        self.dropout2 = nn.Dropout(config.dropout_rate)
        self.fc2 = nn.Linear(config.hidden_size, config.hidden_size // 2)
        self.dropout3 = nn.Dropout(config.dropout_rate)
        self.classifier = nn.Linear(config.hidden_size // 2, num_labels)

        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(config.hidden_size)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size // 2)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights properly"""
        for module in [self.fc1, self.fc2, self.classifier]:
            nn.init.xavier_normal_(module.weight)
            nn.init.constant_(module.bias, 0)

    def forward(self, input_ids, attention_mask=None):
        # Get BERT outputs
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # Use mean pooling instead of just CLS token
        last_hidden_state = outputs.last_hidden_state

        # Mean pooling with attention mask
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            sum_embeddings = torch.sum(last_hidden_state * mask_expanded, 1)
            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
            pooled_output = sum_embeddings / sum_mask
        else:
            pooled_output = torch.mean(last_hidden_state, 1)

        # Forward through enhanced classifier
        x = self.dropout1(pooled_output)
        x = F.relu(self.layer_norm1(self.fc1(x)))
        x = self.dropout2(x)
        x = F.relu(self.layer_norm2(self.fc2(x)))
        x = self.dropout3(x)
        logits = self.classifier(x)

        return logits

def preprocess_text(text):
    """Enhanced text preprocessing for Vietnamese medical text"""
    if isinstance(text, str):
        try:
            # Word tokenization for Vietnamese
            segmented = word_tokenize(text, format="text")
            return segmented.strip()
        except Exception as e:
            print(f"Error in text preprocessing: {e}")
            return text.strip()
    return ""

def predict_disease(model, tokenizer, label_encoder, text, device, max_length=256, top_k=3):
    model.eval()
    processed_text = preprocess_text(text)
    encoding = tokenizer(
        processed_text,
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]

    top_indices = np.argsort(probs)[::-1][:top_k]
    top_labels = label_encoder.inverse_transform(top_indices)
    top_probs = probs[top_indices]
    return top_labels, top_probs, probs

def predict_disease_with_rules(text, disease_keywords, label_encoder):
    input_words = set(preprocess_text(text).lower().split())
    scores = []
    for label in label_encoder.classes_:
        keywords = set(disease_keywords.get(label, []))
        score = len(input_words & keywords) / (len(keywords) or 1)
        scores.append(score)
    return np.array(scores)

def hybrid_predict(model, tokenizer, label_encoder, text, device, disease_keywords, max_length=256, top_k=3, alpha=0.7):
    # Model prediction
    top_labels, top_probs, probs = predict_disease(model, tokenizer, label_encoder, text, device, max_length, top_k=len(label_encoder.classes_))
    # Rule-based prediction
    rule_scores = predict_disease_with_rules(text, disease_keywords, label_encoder)
    # Weighted sum
    combined = alpha * probs + (1 - alpha) * rule_scores
    top_indices = np.argsort(combined)[::-1][:top_k]
    top_labels = label_encoder.inverse_transform(top_indices)
    top_scores = combined[top_indices]
    return top_labels, top_scores, combined

# Hàm chuyển tên bệnh sang dạng tự nhiên
def beautify_label(label: str) -> str:
    return re.sub(r'_+', ' ', label).strip()

# Định nghĩa request/response schema
class PredictRequest(BaseModel):
    text: str

class DiseasePrediction(BaseModel):
    disease: str
    confidence: float

class PredictResponse(BaseModel):
    predictions: List[DiseasePrediction]

# Load model
print("Loading model...")

# Disease keywords
# keywords_extractor.py
from collections import Counter, defaultdict
import pandas as pd
import re

try:
    from underthesea import word_tokenize
    def vn_tokenize(text: str):
        return word_tokenize(text.lower())
except Exception:
    def vn_tokenize(text: str):
        return text.lower().split()

VI_STOPWORDS = {
    'và', 'của', 'có', 'là', 'được', 'trong', 'với', 'cho', 'từ', 'trên',
    'theo', 'về', 'như', 'khi', 'nếu', 'để', 'này', 'đó', 'những', 'các',
    'một', 'hai', 'ba', 'bốn', 'năm', 'sáu', 'bảy', 'tám', 'chín', 'mười',
    'người', 'bệnh', 'nhân', 'bị', 'cảm', 'thấy', 'triệu', 'chứng', 'dấu',
    'hiệu', 'tình', 'trạng', 'xuất', 'hiện', 'gặp', 'phải', 'thường',
    'rất', 'khá', 'hơi', 'một', 'chút', 'ít', 'nhiều', 'lúc', 'khi'
}

# Một số âm tiết thường gặp để tách nhanh các từ ghép bị dính, ví dụ: "nướctiểu" -> "nước tiểu"
SYLLABLE_PREFIXES = [
    'đau', 'khó', 'buồn', 'mệt', 'khát', 'sốt', 'nghẹt', 'chóng',
    'hoa', 'đói', 'gầy', 'tê', 'cứng', 'sưng', 'khàn', 'nhức',
    'căng', 'đầy', 'táo', 'lỏng', 'nóng', 'lạnh', 'ớn', 'chua',
    'nước', 'tiểu', 'phát', 'ban', 'hậu', 'môn', 'xanh', 'xao'
]

SYMPTOM_HEADS = {
    "đau", "nhức", "buốt", "rát", "sưng", "viêm", "ngứa",
    "khô", "nứt", "tê", "mỏi", "cứng", "phù", "chảy",
    "khó", "bí", "tiêu", "tiết", "ho", "nôn", "buồn", "chóng", "hoa"
}
BODY_PARTS = {
    "bụng", "đầu", "họng", "ngực", "lưng", "cổ", "vai", "gối", "khớp",
    "da", "mũi", "mắt", "tai", "miệng", "răng", "lưỡi",
    "dạ dày", "ruột", "phổi", "gan", "thận", "tim"
}
FILLER_WORDS = {"nước"}  # để bắt "chảy nước mũi"
KNOWN_MULTIWORD = {
    "buồn nôn", "khó thở", "tiêu chảy", "đau rát họng", "đau bụng",
    "đau đầu", "đầy hơi", "chuột rút", "chảy nước mũi", "khô da",
    "đau ngực", "đau lưng", "sưng khớp"
}


from collections import Counter, defaultdict
import re



def _norm_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _tokens(text: str):
    # tokenization đã có vn_tokenize ở file của bạn
    try:
        return [t for t in vn_tokenize(_norm_text(text)) if t.strip()]
    except Exception:
        return _norm_text(text).split()

def _ngram_candidates(tokens):
    """Sinh candidate bigram/trigram theo luật đơn giản cho triệu chứng."""
    n = len(tokens)
    cands = []

    # Bigram: head + body
    for i in range(n - 1):
        a, b = tokens[i], tokens[i + 1]
        if a in SYMPTOM_HEADS and (b in BODY_PARTS or f"{a} {b}" in KNOWN_MULTIWORD):
            cands.append(f"{a} {b}")

    # Trigram: head + filler + body  (vd: chảy nước mũi)
    for i in range(n - 2):
        a, b, c = tokens[i], tokens[i + 1], tokens[i + 2]
        if a in SYMPTOM_HEADS and b in FILLER_WORDS and c in BODY_PARTS:
            cands.append(f"{a} {b} {c}")

    return cands

def _match_known_phrases(raw_text: str):
    """Bắt các cụm đã biết trực tiếp từ text (regex chặt để tránh ăn nhầm)."""
    text = _norm_text(raw_text)
    hits = []
    for phrase in KNOWN_MULTIWORD:
        # word-boundary đơn giản cho tiếng Việt (dựa trên khoảng trắng/dấu đầu-cuối)
        if re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", text):
            hits.append(phrase)
    return hits

def _select_top_phrases(counts: Counter, max_keywords: int):
    """Ưu tiên cụm dài hơn, loại các mục là 'substring' của mục đã chọn."""
    # sắp theo (độ dài từ, tần suất) giảm dần
    items = sorted(counts.items(), key=lambda kv: (len(kv[0].split()), kv[1]), reverse=True)
    selected = []
    for phrase, _ in items:
        if all(phrase not in big and not big.startswith(phrase + " ") for big in selected):
            selected.append(phrase)
        if len(selected) >= max_keywords:
            break
    return selected

def extract_keywords_from_symptoms(disease_data, max_keywords: int = 8):
    """
    Trích xuất keyword cho mỗi bệnh, ưu tiên cụm triệu chứng đa từ (bigram/trigram).
    - Bắt cụm kiểu: 'đau bụng', 'khô da', 'sưng khớp', 'chảy nước mũi', ...
    - Giữ cụm dài, tránh trùng lặp với token con.
    """
    print("Đang trích xuất từ khóa...")
    disease_symptoms = defaultdict(list)
    for symptom, disease in disease_data:
        disease_symptoms[disease].append(str(symptom))

    disease_keywords = {}
    for disease, symptoms in disease_symptoms.items():
        phrase_counter = Counter()

        for s in symptoms:
            # 1) match các cụm đã biết
            for p in _match_known_phrases(s):
                phrase_counter[p] += 1

            # 2) sinh candidate theo luật head/body
            toks = _tokens(s)
            for p in _ngram_candidates(toks):
                phrase_counter[p] += 1

        # 3) fallback: nếu thiếu cụm, bổ sung một số đơn từ (ít) có nghĩa
        if len(phrase_counter) < max_keywords:
            # thêm đơn từ meaningful (head/body) xuất hiện trong tokens
            unigram_counts = Counter()
            for s in symptoms:
                toks = _tokens(s)
                for t in toks:
                    if t in SYMPTOM_HEADS or t in BODY_PARTS:
                        unigram_counts[t] += 1
            # gộp thêm một số đơn từ (không lấn át cụm)
            for w, c in unigram_counts.most_common(max(0, max_keywords - len(phrase_counter))):
                if w not in phrase_counter:
                    phrase_counter[w] = c

        # 4) chọn top, ưu tiên cụm dài
        top_keywords = _select_top_phrases(phrase_counter, max_keywords)
        disease_keywords[disease] = top_keywords
        print(f"{disease}: {len(symptoms)} triệu chứng -> {len(top_keywords)} từ khóa")

    return disease_keywords


def load_disease_data_from_csv(csv_path: str):
    """
    Đọc CSV và lấy cột 0 (bệnh), cột 2 (triệu chứng).
    Trả về list các tuple (symptom_text, disease_label).
    """
    df = pd.read_csv(csv_path)
    df = df.iloc[:, [0, 2]]
    df.columns = ['benh', 'trieu_chung']
    df = df.dropna()
    df['benh'] = df['benh'].astype(str).str.replace(' ', '_')
    disease_data = [(str(row['trieu_chung']), str(row['benh'])) for _, row in df.iterrows()]
    return disease_data


excel_path = "./data_processed/augmented_medical_data.csv"
disease_data = load_disease_data_from_csv(excel_path)
disease_keywords = extract_keywords_from_symptoms(disease_data, max_keywords=8)


# Initialize label encoder
label_encoder = LabelEncoder()
label_encoder.fit(list(disease_keywords.keys()))

# Load model checkpoint
from huggingface_hub import hf_hub_download

path_checkpoint = hf_hub_download(
    repo_id="LuongDat/heath_api",
    filename="vihealthbert_disease_model.pth"
)

# Load config and tokenizer
config = Config()
tokenizer = AutoTokenizer.from_pretrained(config.model_name)
vihealthbert = AutoModel.from_pretrained(config.model_name)

# Create model and load trained weights
model = EnhancedDiseaseClassifier(vihealthbert, len(label_encoder.classes_), config)

checkpoint = torch.load(path_checkpoint, map_location=torch.device('cpu'), weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

device = torch.device('cpu')

print("Model loaded successfully!")

# Khởi tạo FastAPI
app = FastAPI(
    title="ViHealthBERT Disease Prediction API",
    description="API for predicting diseases from Vietnamese medical text",
    version="1.0.0"
)

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "ViHealthBERT Disease Prediction API",
        "status": "running",
        "endpoints": {
            "predict": "/predict",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "tokenizer_loaded": tokenizer is not None
    }

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """Predict disease from text symptoms"""
    try:
        text = request.text.strip()
        if not text:
            raise HTTPException(status_code=400, detail="Text cannot be empty")

        # Get top 2 predictions
        top_labels, top_scores, _ = hybrid_predict(
            model, tokenizer, label_encoder, text, device, disease_keywords,
            max_length=256, top_k=2, alpha=0.7
        )

        predictions = [
            DiseasePrediction(disease=beautify_label(label), confidence=float(score))
            for label, score in zip(top_labels, top_scores)
        ]

        return PredictResponse(predictions=predictions)

    except Exception as e:
        print(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# Chạy server trong thread riêng
def run_server():
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

# Khởi tạo và start thread
server_thread = threading.Thread(target=run_server, daemon=True)
server_thread.start()

# Đợi server khởi động
time.sleep(3)

print("🚀 API is running at: http://localhost:8000")
print("📚 Documentation at: http://localhost:8000/docs")
print("🏥 Health check at: http://localhost:8000/health")
print("✅ Server is running in background. You can now run other cells!")
