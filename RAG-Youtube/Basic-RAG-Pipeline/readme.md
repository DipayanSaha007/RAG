In **RAG**,

the RAG pipeline is divided into several parts. The 1st is the **Ingestion Pipeline.**

**\*1>** **Ingestion Pipeline -->**

In **Ingestion Pipeline, the** documents from the companys dataset hv way many tokens which the LLM models can not process at once. So, from those huge num of tokens we chunck out tokens of a passege. Like 10M tokens are there and we create 10,000 tokens chunck and which creates total 10,000 chuncks.

Next, we pass those chuncks to the **embeddings models, which** converts english words and passages into **vector dimentional embeddings.** Generally, embeddings models like "text-embedding-large" openai model uses 3072 dimentions for each vectors. Now the total vector embeddings are stored in a **Vector DB.** Some vector dbs are **ChromaDB, FAISS by META.**

This is the **Ingestion Pipeline. Up Next is the Retrieval Pipeline.**

...

**\*2>** **Retrieval Pipeline -->**

In **Retrieval pipeline**, the user **Query is converted into tokens and then converted into vector embeddings using same embeddings model\*\***.

** We use the **same embeddings models** for both **Ingestion and Retrieval pipeline. If we use different models for Ingestion and Retrival, the embeddings do not mach.** So the embeddings vector won't be able to find matches.**

Then the vector embeddings of the query is used to **Retrieve the vectors from vector db which are similar.** Similar means in the embeddings graph, the vectors with similar english meanings will sit closer to each other.

After finding the similar vectors from the vector db, **We give those chuncks of tokens of those vectors to the LLM modelsand say that here are the references for you to search from to answer the user query.**

**So now, the reference space is way smaller that before which works smoothly with the LLMs.**

...

**\*3> Cosine Similarity -->**

**Cosine Similarity is the reason why we're able to fetch the matching chuvncks for a user query from the vecto database.**

Cosine Similarity measures the angle between vectors, not their magnitude.

The cosine similarity value ranges from 0 to 1, with 0 being the least similar and 1 being the most similar.

**Fomula -> cosine_similarity = (A . B) / (||A|| x ||B||)**

**A . B = dot product of vectors A and B**

**||A|| = magnitude (length) of vector A, same with ||B|| .**

Magnitude of a vector is mainly alwaays 1. So, the divided part is all 1.

WIth newer embeddings models like openai-embeddings-3-small, the magnitude of vectors is **always 1.0.**

...

**\*4>** **Basic RAG vs History-aware Generation -->**

In **B\*\***asic RAG,\*\* each query is treated independently. The retriever takes your exact question and searches for chunks.

In **History-aware RAG,** there's oe critical extra step**: Query Reformulation.**

Without **History-aware RAG,** queries like "What is NVIDIA's newest chip?" and "What is their revenue from it?" are considered different. After executing query 1, in basic RAG it searches the vector db for next query only, which does not give full context to the users quesion.

With **History-aware RAG,** Queries **Reformulates** to: "What's NVIDIA's revenue from Hopper GPU architecture?". It combines the users 1st and 2nd query with the 1st answer from the llm.

...

**\*5>** **RAG-chunking Strategies -->**

**Chunking** is the critical 2nd step in **RAG Ingestion Pipeline -** it determines how your content gets divided for retrieval.

Your RAG system doesn't search entire documents. It searches chunks. So the final answer generation quality depends on those chunks.

**-- The 5 Chunking Strategies -->**

**A} CharacterTextSplitter (Beyond basic chunk_size)**

**B} RecursiveCharacterTextSplitter (tries to split at natural bounderies)**

**C} Document-Specific Splitting**

**D} Semantic Splitting**

**E} Agentic Splitting (AI-Powered Chunking)**
