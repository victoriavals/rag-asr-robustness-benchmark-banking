# RAG ASR Robustness Benchmark - Banking

A comprehensive **RAG (Retrieval-Augmented Generation) system** with ASR (Automatic Speech Recognition) robustness benchmarking for banking domain. Features dual-mode operation: a Live Demo chatbot and a Research Lab for multi-accent evaluation.

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.40+-red.svg)](https://streamlit.io/)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o-green.svg)](https://openai.com/)
[![Qdrant](https://img.shields.io/badge/Qdrant-Vector%20DB-purple.svg)](https://qdrant.tech/)

---

## 🎯 Project Overview

This application provides:

1. **Live Banking Assistant** - Interactive RAG-powered chatbot with text/voice input
2. **Research Lab** - ASR benchmarking across English accents (US, GB, AU) using MINDS-14 dataset
3. **Evaluation Metrics** - WER, Hit Rate, and Embedding Similarity analysis
4. **Visualization** - Interactive Plotly charts for comparative analysis

---

## 🌟 Key Features

### Mode 1: Live Demo
- 💬 **Chat Interface** with text and voice input
- 🎤 **Whisper ASR** for voice transcription
- 🔍 **RAG Pipeline** with context retrieval from Qdrant
- 🤖 **GPT-4o** for intelligent responses
- 📝 **Chat History** with session persistence

### Mode 2: Research Lab

#### Single Sample Analysis
- Select from 14 languages in MINDS-14 dataset
- Audio preview with ground truth transcription
- Side-by-side comparison: Text baseline vs Voice experiment
- Metrics: WER, Embedding Similarity, Retrieved Documents

#### Batch Evaluation
- Multi-accent analysis: 🇺🇸 en-US, 🇬🇧 en-GB, 🇦🇺 en-AU
- Configurable sample size (5-100) or full dataset (1,809 samples)
- Interactive Plotly visualizations:
  - 📉 WER comparison bar chart
  - 🎯 Text vs Voice hit rate (grouped bars)
  - 📊 ASR degradation metrics
  - 📋 Intent-level breakdown
- CSV export for further analysis

---

## 📊 Datasets

### MINDS-14
**Source**: [PolyAI/minds14](https://huggingface.co/datasets/PolyAI/minds14)

| Property | Value |
|----------|-------|
| Languages | 14 (cs-CZ, de-DE, en-AU, en-GB, en-US, es-ES, fr-FR, it-IT, ko-KR, nl-NL, pl-PL, pt-PT, ru-RU, zh-CN) |
| English Samples | en-US: 563, en-GB: 592, en-AU: 654 |
| Audio Format | 8kHz WAV |
| Intents | 14 (balance, pay_bill, card_issues, etc.) |

### Banking Knowledge Base
- **File**: `data/bank_policy.txt`
- **Language**: Indonesian (Bahasa Indonesia)
- **Content**: Neo Bank policies and FAQs

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **Frontend** | Streamlit 1.40+ |
| **LLM** | OpenAI GPT-4o |
| **ASR** | OpenAI Whisper |
| **Embeddings** | OpenAI text-embedding-3-small |
| **Vector DB** | Qdrant |
| **Visualization** | Plotly Express |
| **Metrics** | jiwer (WER), scikit-learn (similarity) |
| **Dataset** | HuggingFace Datasets |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.12+
- OpenAI API Key
- Qdrant instance (local Docker or Qdrant Cloud)

### 1. Clone Repository
```bash
git clone https://github.com/victoriavals/rag-asr-robustness-benchmark-banking.git
cd rag-asr-robustness-benchmark-banking
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Setup Environment Variables

Create `.env` file:
```env
OPENAI_API_KEY=your_openai_api_key_here
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=test-ai-nlp
```

### 4. Start Qdrant (Local Option)

First, pull the Qdrant Docker image:
```bash
docker pull qdrant/qdrant
```

Then, run Qdrant with persistent storage:
```bash
docker run -p 6333:6333 -p 6334:6334 -v "${pwd}/qdrant_storage:/qdrant/storage:z" qdrant/qdrant
```

**Qdrant is now accessible at:**
| Service | URL |
|---------|-----|
| REST API | http://localhost:6333 |
| Web UI Dashboard | http://localhost:6333/dashboard |
| gRPC API | http://localhost:6334 |

> **Note**: Data will be persisted in the `./qdrant_storage` directory.

### 5. Ingest Knowledge Base
```bash
python ingest_data.py
```

### 6. Run Application
```bash
streamlit run main.py
```

Open browser at `http://localhost:8501`

---

## 📁 Project Structure

```
rag-asr-robustness-benchmark-banking/
├── .env                          # Environment variables (not in git)
├── requirements.txt              # Python dependencies
├── main.py                       # Streamlit app entry point
├── ingest_data.py                # Knowledge base ingestion script
├── data/
│   ├── bank_policy.txt           # Banking knowledge base
│   └── minds14/                  # Cached MINDS-14 audio files
└── src/
    ├── config.py                 # Pydantic environment config
    ├── openai_services.py        # OpenAI API wrappers
    ├── vector_db.py              # Qdrant operations with retry logic
    ├── minds14_loader.py         # MINDS-14 dataset loader
    ├── evaluation_engine.py      # WER & Hit Rate metrics
    └── tabs/
        ├── live_demo.py          # Live chat interface
        └── research_lab.py       # Batch evaluation & charts
```

---

## 📈 Evaluation Metrics

### Word Error Rate (WER)
```
WER = (Substitutions + Insertions + Deletions) / Total Words
```
- **0%** = Perfect transcription
- **Lower is better**

### Retrieval Hit Rate
```
Hit = 1 if any(intent_keyword in retrieved_docs) else 0
Hit Rate = sum(hits) / total_samples × 100%
```
- Measures if retrieval is relevant to intent
- **Higher is better**

### ASR Degradation
```
Degradation = Text Hit Rate - Voice Hit Rate
```
- Shows how ASR errors impact retrieval
- **Lower is better**

---

## 🎨 Screenshots

### Live Demo
Chat interface with text and voice input, powered by RAG.

### Research Lab - Single Sample
Side-by-side comparison showing WER and embedding similarity.

### Research Lab - Batch Evaluation
Interactive charts comparing ASR performance across accents.

---

## 🔧 Configuration

### OpenAI Models
- **LLM**: `gpt-4o`
- **Embeddings**: `text-embedding-3-small`
- **ASR**: `whisper-1`

### Qdrant Settings
- **Vector Size**: 1536 (OpenAI embedding dimension)
- **Distance Metric**: Cosine
- **Collection**: Configurable via `.env`

---

## 📝 Usage Examples

### Live Demo Mode
1. Select **Mode 1: Live Banking Assistant**
2. Type question or record voice
3. Receive AI-powered response with context

### Research Lab - Single Sample
1. Select **Mode 2: Research Lab** → **Single Sample Analysis**
2. Choose language (e.g., en-US)
3. Select sample index
4. Click **Run Experiment**
5. View WER, similarity, and retrieved docs

### Research Lab - Batch Evaluation
1. Select **Mode 2: Research Lab** → **Batch Evaluation**
2. Choose locales (en-US, en-GB, en-AU)
3. Set sample size or use full dataset
4. Click **Start Multi-Region Analysis**
5. View interactive charts and download CSV

---

## 🐛 Troubleshooting

### Qdrant Connection Error
```
WinError 10061: No connection could be made
```

**Solutions**:
1. **Start Qdrant locally**:
   ```bash
   docker run -p 6333:6333 qdrant/qdrant
   ```

2. **Use Qdrant Cloud** (free tier):
   - Sign up at [cloud.qdrant.io](https://cloud.qdrant.io/)
   - Update `QDRANT_URL` in `.env`

3. **Verify connection**:
   - Visit `http://localhost:6333/dashboard`

### FFmpeg/torchcodec Error
Audio decoding is configured to use `soundfile` instead of `torchcodec`, avoiding FFmpeg dependency issues on Windows.

---

## 📦 Dependencies

Core libraries:
- `streamlit>=1.40.0` - Web UI
- `openai>=1.0.0` - GPT-4o, Whisper, Embeddings
- `qdrant-client>=1.16.0` - Vector database
- `datasets>=2.14.0` - HuggingFace datasets
- `jiwer>=3.0.0` - WER calculation
- `plotly>=5.18.0` - Interactive charts
- `pandas>=2.1.0` - Data analysis
- `pydantic-settings>=2.0.0` - Environment config

See [requirements.txt](requirements.txt) for full list.

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **OpenAI** for GPT-4o, Whisper, and Embeddings APIs
- **Qdrant** for vector database
- **PolyAI** for MINDS-14 dataset
- **Streamlit** for the amazing web framework
- **HuggingFace** for datasets library

---

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

## 🔗 Links

- [MINDS-14 Dataset](https://huggingface.co/datasets/PolyAI/minds14)
- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)
- [Streamlit Documentation](https://docs.streamlit.io/)
