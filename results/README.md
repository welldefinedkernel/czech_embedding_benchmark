# Benchmark Results

## 1. MS MARCO — Czech

Czech queries against Czech passages (machine-translated MS MARCO validation split).

| # | Model | MRR@10 | nDCG@10 | MAP@10 | Recall@1 | Recall@10 | Recall@20 | Recall@100 | Date |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | [`nvidia/llama-embed-nemotron-8b`](mteb/msmarco/nvidia__llama-embed-nemotron-8b/) | 0.4938 | 0.5679 | 0.4868 | 0.3270 | 0.8130 | 0.8831 | 0.9597 | 2026-07-27 |
| 2 | [`intfloat/multilingual-e5-large`](mteb/msmarco/intfloat__multilingual-e5-large/) | 0.4672 | 0.5370 | 0.4599 | 0.3094 | 0.7699 | 0.8417 | 0.9312 | 2026-07-29 |
| 3 | [`Qwen/Qwen3-Embedding-8B`](mteb/msmarco/Qwen__Qwen3-Embedding-8B/) | 0.4624 | 0.5368 | 0.4553 | 0.2986 | 0.7842 | 0.8608 | 0.9517 | 2026-07-24 |
| 4 | [`codefuse-ai/F2LLM-v2-14B`](mteb/msmarco/codefuse-ai__F2LLM-v2-14B/) | 0.4585 | 0.5351 | 0.4515 | 0.2916 | 0.7892 | 0.8661 | 0.9527 | 2026-07-27 |
| 5 | [`codefuse-ai/F2LLM-v2-8B`](mteb/msmarco/codefuse-ai__F2LLM-v2-8B/) | 0.4567 | 0.5327 | 0.4498 | 0.2907 | 0.7849 | 0.8630 | 0.9521 | 2026-07-24 |
| 6 | [`Octen/Octen-Embedding-8B`](mteb/msmarco/Octen__Octen-Embedding-8B/) | 0.4464 | 0.5214 | 0.4391 | 0.2827 | 0.7717 | 0.8526 | 0.9484 | 2026-07-24 |
| 7 | [`Qwen/Qwen3-Embedding-4B`](mteb/msmarco/Qwen__Qwen3-Embedding-4B/) | 0.4453 | 0.5198 | 0.4383 | 0.2831 | 0.7678 | 0.8470 | 0.9441 | 2026-07-28 |
| 8 | [`microsoft/harrier-oss-v1-27b`](mteb/msmarco/microsoft__harrier-oss-v1-27b/) | 0.4429 | 0.5207 | 0.4360 | 0.2777 | 0.7792 | 0.8588 | 0.9518 | 2026-07-28 |
| 9 | [`jinaai/jina-embeddings-v5-text-small`](mteb/msmarco/jinaai__jina-embeddings-v5-text-small/) | 0.4406 | 0.5112 | 0.4334 | 0.2847 | 0.7473 | 0.8253 | 0.9266 | 2026-07-24 |
| 10 | [`BAAI/bge-m3`](mteb/msmarco/BAAI__bge-m3/) | 0.4386 | 0.5097 | 0.4317 | 0.2828 | 0.7464 | 0.8217 | 0.9136 | 2026-07-28 |
| 11 | [`tencent/KaLM-Embedding-Gemma3-12B-2511`](mteb/msmarco/tencent__KaLM-Embedding-Gemma3-12B-2511/) | 0.4260 | 0.5121 | 0.4196 | 0.2498 | 0.7965 | 0.8793 | 0.9661 | 2026-07-27 |
| 12 | [`intfloat/multilingual-e5-small`](mteb/msmarco/intfloat__multilingual-e5-small/) | 0.4054 | 0.4711 | 0.3986 | 0.2612 | 0.6911 | 0.7689 | 0.8773 | 2026-07-28 |
| 13 | [`microsoft/harrier-oss-v1-0.6b`](mteb/msmarco/microsoft__harrier-oss-v1-0.6b/) | 0.3994 | 0.4704 | 0.3923 | 0.2479 | 0.7087 | 0.7927 | 0.9054 | 2026-07-23 |
| 14 | [`google/embeddinggemma-300m`](mteb/msmarco/google__embeddinggemma-300m/) | 0.3680 | 0.4328 | 0.3620 | 0.2304 | 0.6490 | 0.7284 | 0.8492 | 2026-07-29 |
| 15 | [`geevec-ai/geevec-embeddings-1.0-lite`](mteb/msmarco/geevec-ai__geevec-embeddings-1.0-lite/) | 0.3138 | 0.3736 | 0.3081 | 0.1905 | 0.5743 | 0.6558 | 0.7915 | 2026-07-27 |
| 16 | [`Seznam/simcse-retromae-small-cs`](mteb/msmarco/Seznam__simcse-retromae-small-cs/) | 0.1465 | 0.1802 | 0.1423 | 0.0809 | 0.2972 | 0.3683 | 0.5281 | 2026-08-17 |

## 2. MS MARCO — English

Original English queries against English passages (MS MARCO validation split).

| # | Model | MRR@10 | nDCG@10 | MAP@10 | Recall@1 | Recall@10 | Recall@20 | Recall@100 | Date |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | [`nvidia/llama-embed-nemotron-8b`](mteb/msmarco/nvidia__llama-embed-nemotron-8b/) | 0.5699 | 0.6443 | 0.5629 | 0.3940 | 0.8887 | 0.9433 | 0.9900 | 2026-07-27 |
| 2 | [`codefuse-ai/F2LLM-v2-14B`](mteb/msmarco/codefuse-ai__F2LLM-v2-14B/) | 0.5495 | 0.6243 | 0.5421 | 0.3734 | 0.8716 | 0.9291 | 0.9823 | 2026-07-28 |
| 3 | [`codefuse-ai/F2LLM-v2-8B`](mteb/msmarco/codefuse-ai__F2LLM-v2-8B/) | 0.5465 | 0.6216 | 0.5393 | 0.3712 | 0.8693 | 0.9272 | 0.9819 | 2026-07-24 |
| 4 | [`intfloat/multilingual-e5-large`](mteb/msmarco/intfloat__multilingual-e5-large/) | 0.5447 | 0.6176 | 0.5373 | 0.3716 | 0.8588 | 0.9160 | 0.9751 | 2026-07-29 |
| 5 | [`Qwen/Qwen3-Embedding-8B`](mteb/msmarco/Qwen__Qwen3-Embedding-8B/) | 0.5389 | 0.6153 | 0.5315 | 0.3616 | 0.8677 | 0.9290 | 0.9863 | 2026-07-25 |
| 6 | [`microsoft/harrier-oss-v1-27b`](mteb/msmarco/microsoft__harrier-oss-v1-27b/) | 0.5353 | 0.6143 | 0.5283 | 0.3554 | 0.8742 | 0.9345 | 0.9878 | 2026-07-30 |
| 7 | [`Qwen/Qwen3-Embedding-4B`](mteb/msmarco/Qwen__Qwen3-Embedding-4B/) | 0.5315 | 0.6077 | 0.5240 | 0.3558 | 0.8599 | 0.9237 | 0.9834 | 2026-07-29 |
| 8 | [`microsoft/harrier-oss-v1-0.6b`](mteb/msmarco/microsoft__harrier-oss-v1-0.6b/) | 0.5281 | 0.6053 | 0.5210 | 0.3516 | 0.8600 | 0.9220 | 0.9826 | 2026-07-23 |
| 9 | [`jinaai/jina-embeddings-v5-text-small`](mteb/msmarco/jinaai__jina-embeddings-v5-text-small/) | 0.5277 | 0.6029 | 0.5204 | 0.3541 | 0.8518 | 0.9169 | 0.9799 | 2026-07-24 |
| 10 | [`Octen/Octen-Embedding-8B`](mteb/msmarco/Octen__Octen-Embedding-8B/) | 0.5191 | 0.5967 | 0.5118 | 0.3430 | 0.8531 | 0.9198 | 0.9819 | 2026-07-25 |
| 11 | [`intfloat/multilingual-e5-small`](mteb/msmarco/intfloat__multilingual-e5-small/) | 0.5174 | 0.5902 | 0.5097 | 0.3480 | 0.8327 | 0.8957 | 0.9662 | 2026-07-28 |
| 12 | [`geevec-ai/geevec-embeddings-1.0-lite`](mteb/msmarco/geevec-ai__geevec-embeddings-1.0-lite/) | 0.5111 | 0.5874 | 0.5040 | 0.3376 | 0.8392 | 0.9064 | 0.9758 | 2026-07-27 |
| 13 | [`google/embeddinggemma-300m`](mteb/msmarco/google__embeddinggemma-300m/) | 0.4981 | 0.5757 | 0.4911 | 0.3256 | 0.8324 | 0.9026 | 0.9767 | 2026-07-29 |
| 14 | [`BAAI/bge-m3`](mteb/msmarco/BAAI__bge-m3/) | 0.4946 | 0.5697 | 0.4870 | 0.3239 | 0.8193 | 0.8887 | 0.9673 | 2026-07-29 |
| 15 | [`tencent/KaLM-Embedding-Gemma3-12B-2511`](mteb/msmarco/tencent__KaLM-Embedding-Gemma3-12B-2511/) | 0.4685 | 0.5573 | 0.4617 | 0.2820 | 0.8498 | 0.9233 | 0.9860 | 2026-07-28 |
| 16 | [`Seznam/simcse-retromae-small-cs`](mteb/msmarco/Seznam__simcse-retromae-small-cs/) | 0.1110 | 0.1354 | 0.1076 | 0.0628 | 0.2208 | 0.2724 | 0.4012 | 2026-08-17 |

## 3. CTDC synthetic — Czech

Synthetic Czech questions against the Czech Text Document Corpus (14,887 queries over a 14,690-document corpus of native Czech news articles).

| # | Model | MRR@10 | nDCG@10 | MAP@10 | Recall@1 | Recall@10 | Recall@20 | Recall@100 | Date |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | [`BAAI/bge-m3`](mteb/ctdc_synthetic/BAAI__bge-m3/) | 0.7863 | 0.8226 | 0.7863 | 0.6997 | 0.9340 | 0.9579 | 0.9874 | 2026-08-14 |
| 2 | [`microsoft/harrier-oss-v1-27b`](mteb/ctdc_synthetic/microsoft__harrier-oss-v1-27b/) | 0.7829 | 0.8217 | 0.7829 | 0.6893 | 0.9407 | 0.9653 | 0.9923 | 2026-08-14 |
| 3 | [`Qwen/Qwen3-Embedding-8B`](mteb/ctdc_synthetic/Qwen__Qwen3-Embedding-8B/) | 0.7816 | 0.8192 | 0.7815 | 0.6920 | 0.9346 | 0.9579 | 0.9900 | 2026-08-14 |
| 4 | [`tencent/KaLM-Embedding-Gemma3-12B-2511`](mteb/ctdc_synthetic/tencent__KaLM-Embedding-Gemma3-12B-2511/) | 0.7808 | 0.8180 | 0.7805 | 0.6933 | 0.9336 | 0.9595 | 0.9887 | 2026-08-14 |
| 5 | [`nvidia/llama-embed-nemotron-8b`](mteb/ctdc_synthetic/nvidia__llama-embed-nemotron-8b/) | 0.7789 | 0.8168 | 0.7788 | 0.6892 | 0.9334 | 0.9581 | 0.9880 | 2026-08-14 |
| 6 | [`Qwen/Qwen3-Embedding-4B`](mteb/ctdc_synthetic/Qwen__Qwen3-Embedding-4B/) | 0.7735 | 0.8114 | 0.7734 | 0.6842 | 0.9279 | 0.9550 | 0.9869 | 2026-08-14 |
| 7 | [`microsoft/harrier-oss-v1-0.6b`](mteb/ctdc_synthetic/microsoft__harrier-oss-v1-0.6b/) | 0.7716 | 0.8094 | 0.7715 | 0.6822 | 0.9256 | 0.9526 | 0.9866 | 2026-08-14 |
| 8 | [`Octen/Octen-Embedding-8B`](mteb/ctdc_synthetic/Octen__Octen-Embedding-8B/) | 0.7608 | 0.8021 | 0.7608 | 0.6619 | 0.9285 | 0.9540 | 0.9887 | 2026-08-14 |
| 9 | [`intfloat/multilingual-e5-large`](mteb/ctdc_synthetic/intfloat__multilingual-e5-large/) | 0.7573 | 0.7992 | 0.7571 | 0.6574 | 0.9281 | 0.9540 | 0.9874 | 2026-08-14 |
| 10 | [`jinaai/jina-embeddings-v5-text-small`](mteb/ctdc_synthetic/jinaai__jina-embeddings-v5-text-small/) | 0.7431 | 0.7837 | 0.7430 | 0.6504 | 0.9093 | 0.9378 | 0.9794 | 2026-08-14 |
| 11 | [`intfloat/multilingual-e5-small`](mteb/ctdc_synthetic/intfloat__multilingual-e5-small/) | 0.7242 | 0.7671 | 0.7240 | 0.6252 | 0.9000 | 0.9309 | 0.9749 | 2026-08-14 |
| 12 | [`codefuse-ai/F2LLM-v2-8B`](mteb/ctdc_synthetic/codefuse-ai__F2LLM-v2-8B/) | 0.6994 | 0.7479 | 0.6994 | 0.5875 | 0.8974 | 0.9310 | 0.9780 | 2026-08-14 |
| 13 | [`codefuse-ai/F2LLM-v2-14B`](mteb/ctdc_synthetic/codefuse-ai__F2LLM-v2-14B/) | 0.6970 | 0.7464 | 0.6970 | 0.5810 | 0.8981 | 0.9338 | 0.9799 | 2026-08-14 |
| 14 | [`google/embeddinggemma-300m`](mteb/ctdc_synthetic/google__embeddinggemma-300m/) | 0.6892 | 0.7331 | 0.6891 | 0.5909 | 0.8696 | 0.9064 | 0.9619 | 2026-08-14 |
| 15 | [`geevec-ai/geevec-embeddings-1.0-lite`](mteb/ctdc_synthetic/geevec-ai__geevec-embeddings-1.0-lite/) | 0.5900 | 0.6363 | 0.5897 | 0.4896 | 0.7818 | 0.8348 | 0.9220 | 2026-08-14 |
| 16 | [`Seznam/simcse-retromae-small-cs`](mteb/ctdc_synthetic/Seznam__simcse-retromae-small-cs/) | 0.4689 | 0.5236 | 0.4688 | 0.3611 | 0.6969 | 0.7682 | 0.8968 | 2026-08-17 |

## 4. CTDC synthetic — after reranking

Each retriever's top 100 documents rescored by a cross-encoder. ΔMRR@10 is measured against section 3. Recall@100 is unchanged.

### [`jinaai/jina-reranker-v2-base-multilingual`](https://huggingface.co/jinaai/jina-reranker-v2-base-multilingual)

| # | Retriever | MRR@10 | nDCG@10 | MAP@10 | Recall@1 | Recall@10 | Recall@20 | Recall@100 | ΔMRR@10 | Date |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | [`microsoft/harrier-oss-v1-27b`](mteb/ctdc_synthetic_rerank/microsoft__harrier-oss-v1-27b/jinaai__jina-reranker-v2-base-multilingual/) | 0.8672 | 0.8947 | 0.8672 | 0.7955 | 0.9772 | 0.9860 | 0.9923 | +0.0843 | 2026-08-16 |
| 2 | [`Qwen/Qwen3-Embedding-8B`](mteb/ctdc_synthetic_rerank/Qwen__Qwen3-Embedding-8B/jinaai__jina-reranker-v2-base-multilingual/) | 0.8656 | 0.8929 | 0.8656 | 0.7943 | 0.9752 | 0.9839 | 0.9900 | +0.0841 | 2026-08-15 |
| 3 | [`Octen/Octen-Embedding-8B`](mteb/ctdc_synthetic_rerank/Octen__Octen-Embedding-8B/jinaai__jina-reranker-v2-base-multilingual/) | 0.8656 | 0.8927 | 0.8656 | 0.7944 | 0.9744 | 0.9829 | 0.9887 | +0.1048 | 2026-08-16 |
| 4 | [`tencent/KaLM-Embedding-Gemma3-12B-2511`](mteb/ctdc_synthetic_rerank/tencent__KaLM-Embedding-Gemma3-12B-2511/jinaai__jina-reranker-v2-base-multilingual/) | 0.8654 | 0.8925 | 0.8654 | 0.7944 | 0.9741 | 0.9829 | 0.9887 | +0.0846 | 2026-08-15 |
| 5 | [`intfloat/multilingual-e5-large`](mteb/ctdc_synthetic_rerank/intfloat__multilingual-e5-large/jinaai__jina-reranker-v2-base-multilingual/) | 0.8645 | 0.8917 | 0.8645 | 0.7933 | 0.9734 | 0.9817 | 0.9874 | +0.1073 | 2026-08-16 |
| 6 | [`nvidia/llama-embed-nemotron-8b`](mteb/ctdc_synthetic_rerank/nvidia__llama-embed-nemotron-8b/jinaai__jina-reranker-v2-base-multilingual/) | 0.8643 | 0.8915 | 0.8643 | 0.7929 | 0.9734 | 0.9817 | 0.9880 | +0.0853 | 2026-08-16 |
| 7 | [`BAAI/bge-m3`](mteb/ctdc_synthetic_rerank/BAAI__bge-m3/jinaai__jina-reranker-v2-base-multilingual/) | 0.8637 | 0.8909 | 0.8637 | 0.7923 | 0.9728 | 0.9814 | 0.9874 | +0.0774 | 2026-08-16 |
| 8 | [`Qwen/Qwen3-Embedding-4B`](mteb/ctdc_synthetic_rerank/Qwen__Qwen3-Embedding-4B/jinaai__jina-reranker-v2-base-multilingual/) | 0.8636 | 0.8907 | 0.8636 | 0.7928 | 0.9723 | 0.9811 | 0.9869 | +0.0901 | 2026-08-15 |
| 9 | [`microsoft/harrier-oss-v1-0.6b`](mteb/ctdc_synthetic_rerank/microsoft__harrier-oss-v1-0.6b/jinaai__jina-reranker-v2-base-multilingual/) | 0.8634 | 0.8906 | 0.8634 | 0.7922 | 0.9724 | 0.9808 | 0.9866 | +0.0918 | 2026-08-16 |
| 10 | [`codefuse-ai/F2LLM-v2-14B`](mteb/ctdc_synthetic_rerank/codefuse-ai__F2LLM-v2-14B/jinaai__jina-reranker-v2-base-multilingual/) | 0.8605 | 0.8872 | 0.8605 | 0.7902 | 0.9676 | 0.9750 | 0.9799 | +0.1635 | 2026-08-16 |
| 11 | [`codefuse-ai/F2LLM-v2-8B`](mteb/ctdc_synthetic_rerank/codefuse-ai__F2LLM-v2-8B/jinaai__jina-reranker-v2-base-multilingual/) | 0.8597 | 0.8862 | 0.8597 | 0.7897 | 0.9658 | 0.9733 | 0.9780 | +0.1603 | 2026-08-16 |
| 12 | [`jinaai/jina-embeddings-v5-text-small`](mteb/ctdc_synthetic_rerank/jinaai__jina-embeddings-v5-text-small/jinaai__jina-reranker-v2-base-multilingual/) | 0.8589 | 0.8855 | 0.8589 | 0.7889 | 0.9657 | 0.9737 | 0.9794 | +0.1158 | 2026-08-15 |
| 13 | [`intfloat/multilingual-e5-small`](mteb/ctdc_synthetic_rerank/intfloat__multilingual-e5-small/jinaai__jina-reranker-v2-base-multilingual/) | 0.8567 | 0.8831 | 0.8567 | 0.7871 | 0.9627 | 0.9701 | 0.9749 | +0.1325 | 2026-08-16 |
| 14 | [`google/embeddinggemma-300m`](mteb/ctdc_synthetic_rerank/google__embeddinggemma-300m/jinaai__jina-reranker-v2-base-multilingual/) | 0.8462 | 0.8722 | 0.8462 | 0.7774 | 0.9503 | 0.9576 | 0.9619 | +0.1570 | 2026-08-16 |
| 15 | [`geevec-ai/geevec-embeddings-1.0-lite`](mteb/ctdc_synthetic_rerank/geevec-ai__geevec-embeddings-1.0-lite/jinaai__jina-reranker-v2-base-multilingual/) | 0.8147 | 0.8390 | 0.8147 | 0.7494 | 0.9119 | 0.9185 | 0.9220 | +0.2247 | 2026-08-16 |
| 16 | [`Seznam/simcse-retromae-small-cs`](mteb/ctdc_synthetic_rerank/Seznam__simcse-retromae-small-cs/jinaai__jina-reranker-v2-base-multilingual/) | 0.7994 | 0.8217 | 0.7994 | 0.7395 | 0.8886 | 0.8941 | 0.8968 | +0.3305 | 2026-08-17 |

### [`infgrad/Prism-Qwen3.5-Reranker-0.8B`](https://huggingface.co/infgrad/Prism-Qwen3.5-Reranker-0.8B)

| # | Retriever | MRR@10 | nDCG@10 | MAP@10 | Recall@1 | Recall@10 | Recall@20 | Recall@100 | ΔMRR@10 | Date |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | [`Qwen/Qwen3-Embedding-8B`](mteb/ctdc_synthetic_rerank/Qwen__Qwen3-Embedding-8B/infgrad__Prism-Qwen3.5-Reranker-0.8B/) | 0.8474 | 0.8669 | 0.8336 | 0.7492 | 0.9680 | 0.9804 | 0.9900 | +0.0658 | 2026-08-24 |
| 2 | [`microsoft/harrier-oss-v1-27b`](mteb/ctdc_synthetic_rerank/microsoft__harrier-oss-v1-27b/infgrad__Prism-Qwen3.5-Reranker-0.8B/) | 0.8461 | 0.8685 | 0.8349 | 0.7497 | 0.9700 | 0.9823 | 0.9923 | +0.0632 | 2026-08-25 |
| 3 | [`BAAI/bge-m3`](mteb/ctdc_synthetic_rerank/BAAI__bge-m3/infgrad__Prism-Qwen3.5-Reranker-0.8B/) | 0.8455 | 0.8657 | 0.8324 | 0.7464 | 0.9666 | 0.9792 | 0.9874 | +0.0591 | 2026-08-26 |
| 4 | [`tencent/KaLM-Embedding-Gemma3-12B-2511`](mteb/ctdc_synthetic_rerank/tencent__KaLM-Embedding-Gemma3-12B-2511/infgrad__Prism-Qwen3.5-Reranker-0.8B/) | 0.8452 | 0.8664 | 0.8328 | 0.7482 | 0.9681 | 0.9800 | 0.9887 | +0.0644 | 2026-08-24 |
| 5 | [`intfloat/multilingual-e5-large`](mteb/ctdc_synthetic_rerank/intfloat__multilingual-e5-large/infgrad__Prism-Qwen3.5-Reranker-0.8B/) | 0.8449 | 0.8662 | 0.8329 | 0.7481 | 0.9668 | 0.9801 | 0.9874 | +0.0876 | 2026-08-26 |
| 6 | [`nvidia/llama-embed-nemotron-8b`](mteb/ctdc_synthetic_rerank/nvidia__llama-embed-nemotron-8b/infgrad__Prism-Qwen3.5-Reranker-0.8B/) | 0.8448 | 0.8665 | 0.8332 | 0.7482 | 0.9674 | 0.9784 | 0.9880 | +0.0659 | 2026-08-24 |
| 7 | [`Octen/Octen-Embedding-8B`](mteb/ctdc_synthetic_rerank/Octen__Octen-Embedding-8B/infgrad__Prism-Qwen3.5-Reranker-0.8B/) | 0.8445 | 0.8663 | 0.8328 | 0.7478 | 0.9676 | 0.9795 | 0.9887 | +0.0837 | 2026-08-24 |
| 8 | [`Qwen/Qwen3-Embedding-4B`](mteb/ctdc_synthetic_rerank/Qwen__Qwen3-Embedding-4B/infgrad__Prism-Qwen3.5-Reranker-0.8B/) | 0.8435 | 0.8647 | 0.8309 | 0.7456 | 0.9668 | 0.9781 | 0.9869 | +0.0700 | 2026-08-24 |
| 9 | [`microsoft/harrier-oss-v1-0.6b`](mteb/ctdc_synthetic_rerank/microsoft__harrier-oss-v1-0.6b/infgrad__Prism-Qwen3.5-Reranker-0.8B/) | 0.8430 | 0.8640 | 0.8304 | 0.7449 | 0.9658 | 0.9776 | 0.9866 | +0.0714 | 2026-08-25 |
| 10 | [`codefuse-ai/F2LLM-v2-14B`](mteb/ctdc_synthetic_rerank/codefuse-ai__F2LLM-v2-14B/infgrad__Prism-Qwen3.5-Reranker-0.8B/) | 0.8398 | 0.8611 | 0.8282 | 0.7440 | 0.9607 | 0.9727 | 0.9799 | +0.1428 | 2026-08-25 |
| 11 | [`jinaai/jina-embeddings-v5-text-small`](mteb/ctdc_synthetic_rerank/jinaai__jina-embeddings-v5-text-small/infgrad__Prism-Qwen3.5-Reranker-0.8B/) | 0.8383 | 0.8603 | 0.8276 | 0.7431 | 0.9592 | 0.9708 | 0.9794 | +0.0952 | 2026-08-26 |
| 12 | [`intfloat/multilingual-e5-small`](mteb/ctdc_synthetic_rerank/intfloat__multilingual-e5-small/infgrad__Prism-Qwen3.5-Reranker-0.8B/) | 0.8378 | 0.8597 | 0.8270 | 0.7432 | 0.9584 | 0.9691 | 0.9749 | +0.1136 | 2026-08-26 |
| 13 | [`codefuse-ai/F2LLM-v2-8B`](mteb/ctdc_synthetic_rerank/codefuse-ai__F2LLM-v2-8B/infgrad__Prism-Qwen3.5-Reranker-0.8B/) | 0.8373 | 0.8606 | 0.8278 | 0.7432 | 0.9594 | 0.9706 | 0.9780 | +0.1378 | 2026-08-26 |
| 14 | [`google/embeddinggemma-300m`](mteb/ctdc_synthetic_rerank/google__embeddinggemma-300m/infgrad__Prism-Qwen3.5-Reranker-0.8B/) | 0.8269 | 0.8484 | 0.8165 | 0.7345 | 0.9448 | 0.9555 | 0.9619 | +0.1377 | 2026-08-26 |
| 15 | [`geevec-ai/geevec-embeddings-1.0-lite`](mteb/ctdc_synthetic_rerank/geevec-ai__geevec-embeddings-1.0-lite/infgrad__Prism-Qwen3.5-Reranker-0.8B/) | 0.7955 | 0.8176 | 0.7877 | 0.7100 | 0.9080 | 0.9176 | 0.9220 | +0.2055 | 2026-08-26 |
| 16 | [`Seznam/simcse-retromae-small-cs`](mteb/ctdc_synthetic_rerank/Seznam__simcse-retromae-small-cs/infgrad__Prism-Qwen3.5-Reranker-0.8B/) | 0.7809 | 0.8026 | 0.7750 | 0.7028 | 0.8857 | 0.8931 | 0.8968 | +0.3120 | 2026-08-24 |

### [`mixedbread-ai/mxbai-rerank-large-v2`](https://huggingface.co/mixedbread-ai/mxbai-rerank-large-v2)

| # | Retriever | MRR@10 | nDCG@10 | MAP@10 | Recall@1 | Recall@10 | Recall@20 | Recall@100 | ΔMRR@10 | Date |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | [`microsoft/harrier-oss-v1-27b`](mteb/ctdc_synthetic_rerank/microsoft__harrier-oss-v1-27b/mixedbread-ai__mxbai-rerank-large-v2/) | 0.8134 | 0.8464 | 0.8096 | 0.7183 | 0.9582 | 0.9764 | 0.9923 | +0.0305 | 2026-08-22 |
| 2 | [`Qwen/Qwen3-Embedding-8B`](mteb/ctdc_synthetic_rerank/Qwen__Qwen3-Embedding-8B/mixedbread-ai__mxbai-rerank-large-v2/) | 0.8132 | 0.8458 | 0.8095 | 0.7193 | 0.9563 | 0.9751 | 0.9900 | +0.0316 | 2026-08-25 |
| 3 | [`tencent/KaLM-Embedding-Gemma3-12B-2511`](mteb/ctdc_synthetic_rerank/tencent__KaLM-Embedding-Gemma3-12B-2511/mixedbread-ai__mxbai-rerank-large-v2/) | 0.8129 | 0.8455 | 0.8093 | 0.7192 | 0.9556 | 0.9734 | 0.9887 | +0.0321 | 2026-08-25 |
| 4 | [`intfloat/multilingual-e5-large`](mteb/ctdc_synthetic_rerank/intfloat__multilingual-e5-large/mixedbread-ai__mxbai-rerank-large-v2/) | 0.8122 | 0.8454 | 0.8089 | 0.7179 | 0.9566 | 0.9737 | 0.9874 | +0.0550 | 2026-08-22 |
| 5 | [`nvidia/llama-embed-nemotron-8b`](mteb/ctdc_synthetic_rerank/nvidia__llama-embed-nemotron-8b/mixedbread-ai__mxbai-rerank-large-v2/) | 0.8122 | 0.8446 | 0.8084 | 0.7185 | 0.9546 | 0.9721 | 0.9880 | +0.0333 | 2026-08-23 |
| 6 | [`Octen/Octen-Embedding-8B`](mteb/ctdc_synthetic_rerank/Octen__Octen-Embedding-8B/mixedbread-ai__mxbai-rerank-large-v2/) | 0.8120 | 0.8455 | 0.8091 | 0.7187 | 0.9563 | 0.9739 | 0.9887 | +0.0512 | 2026-08-23 |
| 7 | [`Qwen/Qwen3-Embedding-4B`](mteb/ctdc_synthetic_rerank/Qwen__Qwen3-Embedding-4B/mixedbread-ai__mxbai-rerank-large-v2/) | 0.8120 | 0.8447 | 0.8086 | 0.7194 | 0.9545 | 0.9717 | 0.9869 | +0.0385 | 2026-08-25 |
| 8 | [`BAAI/bge-m3`](mteb/ctdc_synthetic_rerank/BAAI__bge-m3/mixedbread-ai__mxbai-rerank-large-v2/) | 0.8114 | 0.8435 | 0.8076 | 0.7184 | 0.9527 | 0.9708 | 0.9874 | +0.0250 | 2026-08-25 |
| 9 | [`microsoft/harrier-oss-v1-0.6b`](mteb/ctdc_synthetic_rerank/microsoft__harrier-oss-v1-0.6b/mixedbread-ai__mxbai-rerank-large-v2/) | 0.8095 | 0.8423 | 0.8059 | 0.7155 | 0.9529 | 0.9707 | 0.9866 | +0.0379 | 2026-08-22 |
| 10 | [`intfloat/multilingual-e5-small`](mteb/ctdc_synthetic_rerank/intfloat__multilingual-e5-small/mixedbread-ai__mxbai-rerank-large-v2/) | 0.8083 | 0.8411 | 0.8060 | 0.7182 | 0.9479 | 0.9633 | 0.9749 | +0.0842 | 2026-08-22 |
| 11 | [`codefuse-ai/F2LLM-v2-14B`](mteb/ctdc_synthetic_rerank/codefuse-ai__F2LLM-v2-14B/mixedbread-ai__mxbai-rerank-large-v2/) | 0.8072 | 0.8403 | 0.8043 | 0.7148 | 0.9495 | 0.9674 | 0.9799 | +0.1101 | 2026-08-22 |
| 12 | [`codefuse-ai/F2LLM-v2-8B`](mteb/ctdc_synthetic_rerank/codefuse-ai__F2LLM-v2-8B/mixedbread-ai__mxbai-rerank-large-v2/) | 0.8071 | 0.8402 | 0.8044 | 0.7155 | 0.9492 | 0.9657 | 0.9780 | +0.1076 | 2026-08-22 |
| 13 | [`jinaai/jina-embeddings-v5-text-small`](mteb/ctdc_synthetic_rerank/jinaai__jina-embeddings-v5-text-small/mixedbread-ai__mxbai-rerank-large-v2/) | 0.8068 | 0.8392 | 0.8037 | 0.7155 | 0.9475 | 0.9659 | 0.9794 | +0.0637 | 2026-08-22 |
| 14 | [`google/embeddinggemma-300m`](mteb/ctdc_synthetic_rerank/google__embeddinggemma-300m/mixedbread-ai__mxbai-rerank-large-v2/) | 0.7985 | 0.8300 | 0.7953 | 0.7077 | 0.9352 | 0.9502 | 0.9619 | +0.1093 | 2026-08-22 |
| 15 | [`geevec-ai/geevec-embeddings-1.0-lite`](mteb/ctdc_synthetic_rerank/geevec-ai__geevec-embeddings-1.0-lite/mixedbread-ai__mxbai-rerank-large-v2/) | 0.7715 | 0.8018 | 0.7691 | 0.6865 | 0.9009 | 0.9134 | 0.9220 | +0.1815 | 2026-08-22 |
| 16 | [`Seznam/simcse-retromae-small-cs`](mteb/ctdc_synthetic_rerank/Seznam__simcse-retromae-small-cs/mixedbread-ai__mxbai-rerank-large-v2/) | 0.7615 | 0.7897 | 0.7599 | 0.6825 | 0.8796 | 0.8912 | 0.8968 | +0.2926 | 2026-08-24 |

## 5. MTEB (Multilingual, v2) — retrieval subset

Official MTEB retrieval tasks, scored with each task's own main metric and averaged over all splits and subsets.

| # | Model | Mean(Task) | Tasks |
| --- | --- | --- | --- |
| 1 | [`microsoft/harrier-oss-v1-0.6b`](mteb/mteb_multilingual_retrieval/microsoft__harrier-oss-v1-0.6b/) | 0.7117 | 18 |
| 2 | [`google/embeddinggemma-300m`](mteb/mteb_multilingual_retrieval/google__embeddinggemma-300m/) | 0.6058 | 18 |
| 3 | [`BAAI/bge-m3`](mteb/mteb_multilingual_retrieval/BAAI__bge-m3/) | 0.5638 | 18 |
| 4 | [`intfloat/multilingual-e5-large`](mteb/mteb_multilingual_retrieval/intfloat__multilingual-e5-large/) | 0.5626 | 18 |
| 5 | [`intfloat/multilingual-e5-small`](mteb/mteb_multilingual_retrieval/intfloat__multilingual-e5-small/) | 0.5088 | 18 |

### Per-task scores

| Task | `microsoft/harrier-oss-v1-0.6b` | `google/embeddinggemma-300m` | `BAAI/bge-m3` | `intfloat/multilingual-e5-large` | `intfloat/multilingual-e5-small` |
| --- | --- | --- | --- | --- | --- |
| AILAStatutes | 0.3933 | 0.3025 | 0.2919 | 0.2084 | 0.1901 |
| ArguAna | 0.6546 | 0.6597 | 0.5402 | 0.5436 | 0.3909 |
| BelebeleRetrieval | 0.7745 | 0.7238 | 0.7815 | 0.7791 | 0.6629 |
| CovidRetrieval | 0.8365 | 0.7846 | 0.7755 | 0.7561 | 0.7282 |
| HagridRetrieval | 0.9884 | 0.9877 | 0.9877 | 0.9891 | 0.9855 |
| LEMBPasskeyRetrieval | 0.7725 | 0.6125 | 0.5900 | 0.3825 | 0.3825 |
| LegalBenchCorporateLobbying | 0.9437 | 0.9480 | 0.9025 | 0.8972 | 0.8947 |
| MIRACLRetrievalHardNegatives | 0.6648 | 0.6458 | 0.6962 | 0.6675 | 0.6009 |
| MLQARetrieval | 0.7307 | 0.7730 | 0.7331 | 0.7421 | 0.6385 |
| SCIDOCS | 0.2369 | 0.1948 | 0.1633 | 0.1745 | 0.1390 |
| SpartQA | 0.8478 | 0.0698 | 0.0749 | 0.0565 | 0.0543 |
| StackOverflowQA | 0.9372 | 0.8623 | 0.8056 | 0.8889 | 0.8194 |
| StatcanDialogueDatasetRetrieval | 0.5025 | 0.4118 | 0.2167 | 0.1063 | 0.1033 |
| TRECCOVID | 0.8674 | 0.7671 | 0.5485 | 0.7121 | 0.7257 |
| TempReasonL1 | 0.3375 | 0.0062 | 0.0099 | 0.0114 | 0.0080 |
| TwitterHjerneRetrieval | 0.6382 | 0.6631 | 0.7149 | 0.7540 | 0.5818 |
| WikipediaRetrievalMultilingual | 0.8498 | 0.8998 | 0.8987 | 0.9082 | 0.8772 |
| WinoGrande | 0.8340 | 0.5917 | 0.4172 | 0.5498 | 0.3746 |
