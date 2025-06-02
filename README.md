# SOP Chatbot with LayoutLMv3/LayoutXLM

This is an LLM-enabled chatbot that processes Standard Operating Procedures (SOPs) and provides intelligent responses based on the document content. It uses LayoutLMv3/LayoutXLM for enhanced document understanding and OCR capabilities.

## Setup

1. Create a `.env` file in the project root with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

2. Install system dependencies:
   ```bash
   # Install Tesseract OCR
   brew install tesseract

   # Install poppler (required for PDF processing)
   brew install poppler

   # Install pandoc (for better docx handling)
   brew install pandoc
   ```

3. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Place your SOP documents in the `localdata/SOPs/` directory.

5. Build the vector store with LayoutLMv3/LayoutXLM:
   ```bash
   python rebuild_vector_store.py
   ```

6. Run the API server:
   ```bash
   python api.py
   ```

## Usage

The chatbot exposes two API endpoints:

1. POST `/ask`
   - Send questions to the chatbot
   - Request body: `{"text": "your question here"}`

2. GET `/chat-history`
   - Retrieve the conversation history

The chatbot will:
- Load and process all documents from the SOPs directory
- Extract text from documents using advanced OCR with LayoutLMv3/LayoutXLM
- Create embeddings using HuggingFace models
- Store vectors locally for faster subsequent access
- Provide relevant answers with source references

## Features

- Advanced document processing with LayoutLMv3/LayoutXLM
- Enhanced OCR capabilities for images in PDF and DOCX files
- Document layout understanding for better text extraction
- Vector storage with FAISS for efficient retrieval
- Conversation memory to maintain context
- Source tracking for answers
- Local vector store caching for improved performance

## Python Dependencies

The project uses the following key Python packages:
- transformers - For LayoutLMv3/LayoutXLM models
- datasets - For handling document datasets
- timm - For vision models
- torch - For tensor operations
- langchain - For document processing and LLM integration
- faiss-cpu - For vector storage
- pytesseract - For OCR fallback
- PyMuPDF - For PDF processing

Note: For detectron2 (optional for advanced object detection), you may need to install it separately after installing the other dependencies:
```bash
# First make sure torch is installed
pip install torch

# Then install detectron2
pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

Alternatively, you can skip detectron2 installation as it's not strictly required for basic functionality. The system will fall back to using pytesseract for OCR if detectron2 is not available.

## Testing and Utilities

### Handling Warnings

#### Tokenizers Parallelism Warnings

When running the scripts, you might see warnings about tokenizers parallelism:

```
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
```

To avoid these warnings, you can use the provided `set_env.sh` script which sets the `TOKENIZERS_PARALLELISM` environment variable:

```bash
# Make the script executable (if not already)
chmod +x set_env.sh

# Run commands with the environment variable set
./set_env.sh python rebuild_vector_store.py
./set_env.sh python test_layout_model.py path/to/your/document.pdf
```

Alternatively, you can set the environment variable manually:

```bash
export TOKENIZERS_PARALLELISM=false
python rebuild_vector_store.py
```

#### Device Argument Deprecation Warnings

You may also see warnings about the `device` argument being deprecated:

```
FutureWarning: The `device` argument is deprecated and will be removed in v5 of Transformers.
```

These warnings are harmless and can be safely ignored. They occur because the Transformers library is evolving, and the way devices are specified is changing. The code has been updated to use the newer `device_map` parameter, but some internal components of the library may still use the deprecated approach.

The updated `set_env.sh` script now automatically suppresses these FutureWarnings, so you shouldn't see them when running commands through the script:

```bash
./set_env.sh python rebuild_vector_store.py  # No FutureWarnings will be shown
```

If you want to suppress these warnings in your own Python scripts without using `set_env.sh`, you can add the following code at the beginning of your script:

```python
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.modeling_utils")
```

### Test LayoutLMv3/LayoutXLM Document Processing

You can test the LayoutLMv3/LayoutXLM document processing capabilities on a specific file:

```bash
python test_layout_model.py path/to/your/document.pdf
```

Or with a specific model:

```bash
python test_layout_model.py path/to/your/document.docx --model microsoft/layoutxlm-base
```

### Rebuild Vector Store

To rebuild the vector store with the latest documents and improved processing:

```bash
python rebuild_vector_store.py
```

This will create a backup of the existing vector store before rebuilding.
