# Pretrain_VihealthBert_DiseaseClassifier

---

Dự án huấn luyện mô hình **ViHealthBERT-based Disease Classifier** nhằm cung cấp embeddigns/ngữ cảnh chất lượng cao cho các ứng dụng y tế tiếng Việt (ví dụ: phân loại bệnh, dự đoán bệnh). Mô hình này hiện đang được sử dụng trong dự án Website_Healthcare_Booking_VoVBacSi_v2.

## Giới thiệu
Dự án này huấn luyện một mô hình **Disease Classifier dựa trên ViHealthBERT**—một biến thể của BERT đã được tối ưu hóa cho lĩnh vực y tế tiếng Việt—nhằm phục vụ các tác vụ như phân loại bệnh dựa trên triệu chứng hoặc văn bản mô tả y khoa.

---

## Kiến trúc & Công nghệ
- Nền tảng **Python**, sử dụng thư viện như `transformers`, `PyTorch`.
- Mô hình khởi tạo từ **ViHealthBERT** (pre-trained model cho domain y tế) :contentReference[oaicite:1]{index=1}.
- Sử dụng các kỹ thuật như **fine-tuning**, **masking**, **classfication head**,...

---

## Nguồn dữ liệu & Pre-training
- Dữ liệu huấn luyện gồm các văn bản y tế tiếng Việt (có thể là tri thức y khoa, triệu chứng, v.v.).
- Quá trình pre-train sử dụng **task classification** hoặc supervised fine-tuning trên tập dữ liệu nhãn bệnh lý.

---

## Fine-tuning & Ứng dụng
Mô hình được fine-tune trên các tập dữ liệu có nhãn bệnh cụ thể (có thể là mô tả triệu chứng → bệnh). Kết quả được triển khai dưới dạng service (API hoặc module gọi nội bộ) trong hệ thống Website_Healthcare_Booking_VoVBacSi_v2 để đưa ra gợi ý chẩn đoán.

---

## Hạn chế
- Dữ liệu huấn luyện có thể không bao quát hết các bệnh và triệu chứng phổ biến.
- Mô hình có thể không xử lý tốt các trường hợp tri thức y khoa phức tạp.
- Chưa có khả năng explainable AI (giải thích ra lý do dự đoán).
---
## Hướng phát triển tương lai
- Thu thập thêm dữ liệu y tế đa dạng, có expert labeling.
- Cải tiến mô hình (như model size nhỏ hơn, tốc độ inference nhanh, hỗ trợ giải thích).
- Triển khai dưới dạng API REST hoặc microservice độc lập.
- UI/UX đơn giản để người dùng cuối (người bệnh hoặc bác sĩ) dễ sử dụng.

---



## Cấu trúc thư mục
    ```bash
    config/ # cấu hình huấn luyện (học rate, epochs,...)
    data/ # dữ liệu huấn luyện & validation
    models/ # mô hình đã huấn luyện lưu trữ
    scripts/ # script tiền xử lý, huấn luyện, đánh giá
    train.py # script chính để train mô hình
    infer.py # script để inference (dự đoán bệnh)
    requirements.txt # các thư viện cần thiết
    README.md # file hướng dẫn này

---

## Yêu cầu hệ thống
- Python 3.8+  
- PyTorch (>=1.x) hoặc TensorFlow  
- Thư viện: `transformers`, `scikit-learn`, `pandas`, `numpy`, v.v.

---

## Cài đặt & Chạy thử
    ```bash
      git clone https://github.com/LuongDat9999/Pretrain_VihealthBert_DiseaseClassifier.git
      cd Pretrain_VihealthBert_DiseaseClassifier
      pip install -r requirements.txt
      
      # Tạo mô hình, fine-tune
      python train.py --config config/train_config.yaml
      
      # Kiểm thử mô hình
      python infer.py --model models/latest_model.pt --input "sốt, đau họng, mệt mỏi"

### Trích dẫn ViHealthBERT:
 https://github.com/demdecuong/vihealthbert/blob/main/README.md
 Minh Phuc Nguyen, Vu Hoang Tran, Vu Hoang, Ta Duc Huy, Trung H. Bui, Steven Q. H. Truong. (2022). ViHealthBERT: Pre-trained Language Models for Vietnamese in Health Text Mining. LREC 2022. :contentReference[oaicite:14]{index=14}


