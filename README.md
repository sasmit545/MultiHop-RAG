# 💡 MultiHop-RAG
A Dataset for **Evaluating Retrieval-Augmented Generation Across Documents**  

# 🚀 Overview
**MultiHop-RAG**: a QA dataset to evaluate retrieval and reasoning across documents with metadata in the RAG pipelines. It contains 2556 queries, with evidence for each query distributed across 2 to 4 documents. The queries also involve document metadata, reflecting complex scenarios commonly found in real-world RAG applications.  
📄 Paper Link: [MultiHop-RAG: Benchmarking Retrieval-Augmented Generation for Multi-Hop Queries](https://arxiv.org/pdf/2401.15391.pdf)  
🤗 [Hugging Face dataloader](https://huggingface.co/datasets/yixuantt/MultiHopRAG)

# Simple Use Case
Please try 'simple_retrieval.py,' a sample use case demonstrating retrieval using this dataset. 
```
pip install llama-index
```
```shell
# test simple retrieval and save results
python simple_retrieval.py --retriever BAAI/llm-embedder

# test simple retrieval with rerank and save results
python simple_retrieval.py --retriever BAAI/llm-embedder --rerank
```
![rag.png](resource/rag.png)
# Citation
```
Please cite the Arxiv paper; thanks.
```
# License
MultiHop-RAG is licensed under [ODC-BY](https://opendatacommons.org/licenses/by/1-0/)
