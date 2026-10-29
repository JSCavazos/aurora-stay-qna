
# Match the local notebook's Python minor version.
FROM python:3.12-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir --upgrade pip
# Install CPU Torch before resolving the remaining inference dependencies.
RUN pip install --no-cache-dir torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt
ENV HF_HOME=/app/huggingface
# Pin the same model revision used to embed the serialized notebook chunks.
RUN python -c "from langchain_huggingface import HuggingFaceEmbeddings; HuggingFaceEmbeddings(model_name='HarishMaths/Hotel-Policy-Embedding', model_kwargs={'device':'cpu','revision':'3c9f67da8057fb1e9f742c9e6680d5a4b33b4dcd'}, encode_kwargs={'normalize_embeddings':True})"
ENV TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1
EXPOSE 7860
CMD ["gunicorn", "-w", "1", "--timeout", "120", "-b", "0.0.0.0:7860", "app:rag_api"]
