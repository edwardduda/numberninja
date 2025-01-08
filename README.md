## Number Ninja - Bringing math tutoring to high school students in need. 

## Table of Contents
1. [Introduction](#introduction)
2. [Background](#background)
3. [Cosine Similarity and its Role in Context](#cosine-similarity-and-its-role-in-context)
4. [Heirarchical Navigable Small World](#Heirarchical Navigable Small World)
5. [Installation and Setup](#installation-and-setup)
6. [Usage](#usage)
7. [Approach](#approach)
8. [Results](#results)
9. [Conclusion](#conclusion)
10. [Contributing](#contributing)
11. [References](#references)


## Background

Vector embeddings are a way of representing discrete data in a continuous vector space of n dimensions. This process transforms data into a numerical format that preserves the relationships and structure within the data. By mapping data to a vector embedding space, these embeddings allow AI algorithms, such as neural networks, to interpret and utilize the underlying patterns and meanings within the data.

In transformer neural networks, the primary architecture for modern-day LLMs, the encoder is responsible for taking input data (such as text) and processing it through multiple layers of self-attention and feed-forward neural networks. Through this process, the encoder creates high-dimensional vector representations that capture the semantic meaning and contextual information of the input data. These embeddings can then be used for various downstream tasks, such as language translation, sentiment analysis, and more. However, it is important to note that not all transformer-based neural networks use encoders; vector embeddings can be generated in other ways as well.

## Cosine similarity and its role in context
There are many ways to determine the congruency of a pair of vectors, one of them being cosine similarity. Cosine similarity is used to determine how similar the direction of the vectors is. The closer this value is to 1, the more similar their direction. This measure is crucial for assessing the contextual relevance between different pieces of data within the embedding space. Words with more association have a higher similarity.

![Cosine Similarity applied.](https://github.com/edwardduda/NeuroContext/blob/174930272101e329650ac97317969644adc38f3a/1_jptD3Rur8gUOftw-XHrezQ-2342057285.png)

## Heirarchical Navigable Small World

Hierarchical NSW (Navigable Small World) graph search method greatly improves upon earlier approaches for approximate nearest neighbor searches, particularly in handling high-dimensional data like vector embeddings. It is robust and performs consistently well across different types of datasets and doesn't require tailoring for specific problems. This makes it highly practical for real-world applications like large vector databases. While there are opportunities to further enhance its efficiency and scalability, it already sets a new standard in terms of speed and accuracy.

![HNSW.](readmeimgs/hnswdiagram.png)
