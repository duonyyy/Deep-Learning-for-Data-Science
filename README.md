<div align="center">
  <img src="https://www.uit.edu.vn/media/Logo_UIT_Web_6d18902c3d.png" width="300" alt="UIT Logo"/>
</div>

---

# 🧠 Deep Learning trong Khoa Học Dữ Liệu (DS201) – Báo Cáo Thực Hành

**Tác giả:** Dương Tùng Dương  
**Môn học:** Deep Learning trong Khoa Học Dữ Liệu (DS201)  
**Trường:** Đại học Công nghệ Thông tin, ĐHQG-HCM (UIT)  

---

## 📋 Tổng Quan Các Bài Thực Hành

| Lab | Tên nội dung | Mô hình & Kỹ thuật | Bộ dữ liệu | Thư mục |
|:---:|:---|:---|:---|:---:|
| **1** | Mạng Neuron Đơn Giản & So Sánh CNN/MLP | 1-layer & 3-layer MLP, CNN cơ bản | MNIST | [`LAB1/`](LAB1) |
| **2** | Mạng Neural Tích Chập (CNN) | LeNet, GoogLeNet (Inception), ResNet-18, Pretrained ResNet-50 | MNIST, VinaFood21 | [`LAB2/`](LAB2) |
| **3** | RNN: Phân Loại Văn Bản & Gán Nhãn Chuỗi | 5-layer LSTM, 5-layer GRU, BiLSTM-CRF | UIT-VSFC, PhoNER COVID-19 | [`LAB3/`](LAB3) |
| **4** | Mạng Neural Hồi Quy – Dịch Máy (NMT) | Seq2Seq LSTM (Basic, Bahdanau Attention, Luong Attention) | PhoMT (Anh → Việt) | [`LAB4/`](LAB4) |
| **5** | Transformer Encoder | Transformer Encoder 3 lớp (Self-Attention) | UIT-ViOCD, PhoNER COVID-19 | [`LAB5/`](LAB5) |
| **6** | NLP Nâng Cao, Transformers & LLMs | Pretrained BERT/RoBERTa/PhoBERT, Seq2Seq T5/BART/ViT5, QLoRA Fine-tuning LLM | IMDB, BBC, SAMSum, VietNews | [`LAB6/`](LAB6) |

---

## 🗂️ Cấu Trúc Chi Tiết Từng Lab

### 🔹 Lab 1 – Xây Dựng Mạng Neuron Đơn Giản
* **Bộ dữ liệu:** [MNIST](http://yann.lecun.com/exdb/mnist/) (lưu trữ sẵn tại thư mục [`LAB1/mnist-dataset/`](LAB1/mnist-dataset))
* **Độ đo đánh giá:** Accuracy, Macro Precision, Macro Recall, Macro F1
* **Các Notebook:**
  * [`LAB1/LAB1_MNIST.ipynb`](LAB1/LAB1_MNIST.ipynb): Quy trình chuẩn PyTorch huấn luyện 1-layer và 3-layer MLP, ma trận nhầm lẫn (Confusion Matrix) và phân tích các trường hợp đoán sai.
  * [`LAB1/LAB1_MNIST_BAI_TAP_MO_RONG.ipynb`](LAB1/LAB1_MNIST_BAI_TAP_MO_RONG.ipynb): Thực nghiệm mở rộng (khảo sát Learning Rate `1e-2`, `1e-3`, `1e-4`, kiểm chứng Overfitting khi bỏ Dropout, so sánh optimizer Adam vs SGD, 3 cặp chữ số nhầm nhiều nhất).
  * [`LAB1/cnn_vs_mlp.ipynb`](LAB1/cnn_vs_mlp.ipynb): So sánh trực tiếp kiến trúc CNN và MLP về số lượng tham số, thời gian hội tụ và độ chính xác.

---

### 🔹 Lab 2 – Mạng Neural Tích Chập (CNN)
* **Bộ dữ liệu:** MNIST, VinaFood21
* **Độ đo đánh giá:** Accuracy, Precision, Recall, Macro F1
* **Các Notebook:**
  * [`LAB2/Bai_1_LeNet/LAB2_Bai1_LeNet.ipynb`](LAB2/Bai_1_LeNet/LAB2_Bai1_LeNet.ipynb): Cài đặt và huấn luyện kiến trúc LeNet kinh điển trên MNIST.
  * [`LAB2/Bai_2_GoogLeNet/LAB2_Bai2_GoogLeNet.ipynb`](LAB2/Bai_2_GoogLeNet/LAB2_Bai2_GoogLeNet.ipynb) & [`BAI_2.ipynb`](LAB2/Bai_2_GoogLeNet/BAI_2.ipynb): Mô hình GoogLeNet với khối Inception đa nhánh.
  * [`LAB2/Bai_3_ResNet/LAB2_Bai3_ResNet.ipynb`](LAB2/Bai_3_ResNet/LAB2_Bai3_ResNet.ipynb): Xây dựng mạng ResNet-18 từ đầu với kết nối tắt (Skip Connections).
  * [`LAB2/Bai_4_Pretrained/LAB2_Bai4_Pretrained.ipynb`](LAB2/Bai_4_Pretrained/LAB2_Bai4_Pretrained.ipynb): Fine-tuning mô hình Pretrained ResNet-50 từ Hugging Face trên tập VinaFood21.
  * [`LAB2/assignments.ipynb`](LAB2/assignments.ipynb): Báo cáo và bài tập thực hành tổng hợp Lab 2.

---

### 🔹 Lab 3 – RNN cho Phân Loại Văn Bản & Gán Nhãn Chuỗi
* **Bộ dữ liệu:**
  * [UIT-VSFC](https://drive.google.com/drive/folders/1rdcXNGt_3-QUvV8EtSvVsLMVeHmk6Yqk?usp=drive_link) (Vietnamese Students' Feedback Corpus) – lưu trữ sẵn tại [`LAB3/data/UIT-VSFC/`](LAB3/data/UIT-VSFC/)
  * [PhoNER COVID-19](https://github.com/VinAIResearch/PhoNER_COVID19)
* **Độ đo đánh giá:** F1-score (Macro F1 / Weighted F1)
* **Các Notebook:**
  * [`LAB3/LAB3_Bai1_LSTM_VSFC.ipynb`](LAB3/LAB3_Bai1_LSTM_VSFC.ipynb): Mô hình 5-layer LSTM (`hidden_size=256`) phân loại cảm xúc câu phản hồi sinh viên.
  * [`LAB3/LAB3_Bai2_GRU_VSFC.ipynb`](LAB3/LAB3_Bai2_GRU_VSFC.ipynb): Mô hình 5-layer GRU (`hidden_size=256`) phân loại cảm xúc câu phản hồi sinh viên.
  * [`LAB3/LAB3_Bai3_BiLSTM_CRF_PhoNER.ipynb`](LAB3/LAB3_Bai3_BiLSTM_CRF_PhoNER.ipynb): Mô hình BiLSTM-CRF gán nhãn thực thể tên (Named Entity Recognition).
  * [`LAB3/assignments.ipynb`](LAB3/assignments.ipynb): Bài tập tổng hợp Lab 3.

---

### 🔹 Lab 4 – Mạng Neural Hồi Quy Cho Dịch Máy (NMT)
* **Bộ dữ liệu:** [PhoMT](https://drive.google.com/drive/folders/186OAOuSEYEDVcry7WP5UBdqECXo26QAb?usp=drive_link) (Dịch máy Anh – Việt)
* **Độ đo đánh giá:** BLEU Score, ROUGE-L
* **Các Notebook:**
  * [`LAB4/LAB4_Bai1_Basic_LSTM.ipynb`](LAB4/LAB4_Bai1_Basic_LSTM.ipynb): Seq2Seq cơ bản với 3-layer LSTM Encoder & Decoder.
  * [`LAB4/LAB4_Bai2_Bahdanau_Attention.ipynb`](LAB4/LAB4_Bai2_Bahdanau_Attention.ipynb): Seq2Seq LSTM tích hợp cơ chế chú ý cộng tính (Additive Attention - Bahdanau et al., 2014).
  * [`LAB4/LAB4_Bai3_Luong_Attention.ipynb`](LAB4/LAB4_Bai3_Luong_Attention.ipynb): Seq2Seq LSTM tích hợp cơ chế chú ý nhân (Multiplicative/Dot Attention - Luong et al., 2015).

---

### 🔹 Lab 5 – Transformer Encoder
* **Bộ dữ liệu:** [UIT-ViOCD](https://drive.google.com/drive/folders/1Lu9axyLkw7dMx80uLRgvCnZsmNzhJWAa?usp=sharing), [PhoNER COVID-19](https://github.com/VinAIResearch/PhoNER_COVID19)
* **Độ đo đánh giá:** F1-score
* **Các Notebook:**
  * [`LAB5/LAB5_Bai1_Transformer_Classification.ipynb`](LAB5/LAB5_Bai1_Transformer_Classification.ipynb): Kiến trúc 3-layer Transformer Encoder (theo bài báo *Attention Is All You Need*) ứng dụng phân loại miền dữ liệu (Domain Classification).
  * [`LAB5/LAB5_Bai2_Transformer_NER.ipynb`](LAB5/LAB5_Bai2_Transformer_NER.ipynb): Kiến trúc Transformer Encoder cho bài toán gán nhãn chuỗi / nhận dạng thực thể tên (NER).

---

### 🔹 Lab 6 – NLP Nâng Cao, Mô Hình Ngôn Ngữ & Fine-Tuning LLM
* **Bộ dữ liệu:** IMDB Reviews, BBC News, PhoBERT UIT-VSFC, SAMSum, VietNews
* **Kỹ thuật nổi bật:** Pretrained Transformers, Text Summarization, Parameter-Efficient Fine-Tuning (LoRA & QLoRA 4-bit)
* **Các Notebook:**
  * [`LAB6/LAB07_Text_Classification_GPU_COLAB_UNIFIED.ipynb`](LAB6/LAB07_Text_Classification_GPU_COLAB_UNIFIED.ipynb): Phân loại văn bản nhị phân & đa lớp (IMDB + BBC News) tối ưu hóa GPU.
  * [`LAB6/LAB07_Text_Classification_EXTENDED.ipynb`](LAB6/LAB07_Text_Classification_EXTENDED.ipynb): Thực nghiệm nâng cao phân loại văn bản với Doc2Vec và TF-IDF baseline.
  * [`LAB6/LAB08_Transformers_BERT_HuggingFace_GPU_Colab.ipynb`](LAB6/LAB08_Transformers_BERT_HuggingFace_GPU_Colab.ipynb): Fine-tuning BERT, DistilBERT, RoBERTa và PhoBERT-base-v2 trên Hugging Face.
  * [`LAB6/LAB09_T5_BART_ViT5_Text_Summarization_GPU_Colab.ipynb`](LAB6/LAB09_T5_BART_ViT5_Text_Summarization_GPU_Colab.ipynb): Tóm tắt văn bản trừu tượng (Abstractive Summarization) với T5-small, BART-base và ViT5-base.
  * [`LAB6/LAB11_Fine_Tuning_LLM_LoRA_QLoRA_GPU_Colab.ipynb`](LAB6/LAB11_Fine_Tuning_LLM_LoRA_QLoRA_GPU_Colab.ipynb): Huấn luyện chỉ thị (Instruction Tuning) mô hình ngôn ngữ lớn (LLM) tiếng Việt bằng LoRA 16-bit và QLoRA 4-bit qua `peft` & `bitsandbytes`.
  * [`LAB6/LAB12_Bai_Kiem_Tra_Tong_Hop_NLP_T4_Colab.ipynb`](LAB6/LAB12_Bai_Kiem_Tra_Tong_Hop_NLP_T4_Colab.ipynb): Bài kiểm tra tổng hợp kiến thức NLP toàn diện từ Deep Learning truyền thống đến LLM.

---

## 📂 Sơ Đồ Cấu Trúc Dự Án

```text
Deep-Learning-for-Data-Science/
│
├── README.md                                          # Tài liệu tổng quan dự án
│
├── LAB1/                                              # Lab 1: Mạng Neuron cơ bản & CNN/MLP
│   ├── mnist-dataset/                                 # Dữ liệu MNIST ubyte
│   ├── cnn_vs_mlp.ipynb
│   ├── LAB1_MNIST_BAI_TAP_MO_RONG.ipynb
│   └── LAB1_MNIST.ipynb
│
├── LAB2/                                              # Lab 2: CNN, LeNet, GoogLeNet, ResNet
│   ├── assignments.ipynb
│   ├── Bai_1_LeNet/
│   │   └── LAB2_Bai1_LeNet.ipynb
│   ├── Bai_2_GoogLeNet/
│   │   ├── BAI_2.ipynb
│   │   └── LAB2_Bai2_GoogLeNet.ipynb
│   ├── Bai_3_ResNet/
│   │   └── LAB2_Bai3_ResNet.ipynb
│   └── Bai_4_Pretrained/
│       └── LAB2_Bai4_Pretrained.ipynb
│
├── LAB3/                                              # Lab 3: RNN, LSTM, GRU, BiLSTM-CRF
│   ├── assignments.ipynb
│   ├── data/
│   │   └── UIT-VSFC/                                  # Dữ liệu phân loại cảm xúc JSON
│   ├── LAB3_Bai1_LSTM_VSFC.ipynb
│   ├── LAB3_Bai2_GRU_VSFC.ipynb
│   └── LAB3_Bai3_BiLSTM_CRF_PhoNER.ipynb
│
├── LAB4/                                              # Lab 4: NMT, Seq2Seq & Attention
│   ├── LAB4_Bai1_Basic_LSTM.ipynb
│   ├── LAB4_Bai2_Bahdanau_Attention.ipynb
│   └── LAB4_Bai3_Luong_Attention.ipynb
│
├── LAB5/                                              # Lab 5: Transformer Encoder
│   ├── LAB5_Bai1_Transformer_Classification.ipynb
│   └── LAB5_Bai2_Transformer_NER.ipynb
│
└── LAB6/                                              # Lab 6: Advanced NLP, Transformers & LLM
    ├── LAB07_Text_Classification_EXTENDED.ipynb
    ├── LAB07_Text_Classification_GPU_COLAB_UNIFIED.ipynb
    ├── LAB08_Transformers_BERT_HuggingFace_GPU_Colab.ipynb
    ├── LAB09_T5_BART_ViT5_Text_Summarization_GPU_Colab.ipynb
    ├── LAB11_Fine_Tuning_LLM_LoRA_QLoRA_GPU_Colab.ipynb
    └── LAB12_Bai_Kiem_Tra_Tong_Hop_NLP_T4_Colab.ipynb
```

---

## 🛠️ Yêu Cầu Môi Trường & Cài Đặt

Khuyến khích chạy các notebook trên **Google Colab (GPU)** hoặc môi trường cục bộ với GPU NVIDIA (CUDA):

```bash
# Cài đặt các thư viện Deep Learning cốt lõi
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Cài đặt thư viện NLP, Transformers & LLM
pip install transformers datasets accelerate peft bitsandbytes evaluate rouge-score

# Cài đặt công cụ phân tích và trực quan hóa dữ liệu
pip install numpy pandas matplotlib seaborn scikit-learn idx2numpy tqdm
```

---

*© Dương Tùng Dương*  
*Khoa Khoa học và Kỹ thuật Thông tin – Trường Đại học Công nghệ Thông tin, ĐHQG-HCM*
