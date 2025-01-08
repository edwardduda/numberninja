## Number Ninja - Bringing math tutoring to high school students in need. 

## Table of Contents
1. [Introduction](#introduction)
2. [Background](#background)
3. [Cosine Similarity and its Role in Context](#cosine-similarity-and-its-role-in-context)
4. [Heirarchical Navigable Small World](#heirarchical-navigable-small-world)
5. [Installation and Setup](#installation-and-setup)
6. [Usage & Approach](#usage-and-approach)
7. [Future Implementations](#future-implementations)
8. [Conclusion](#Conclusion)
9. [References and Acknowledgements](#references_and_acknowledgements)

## Problem

Mathematics is known for being one of the most challenging and least enjoyed subjects by high school students. Many students struggle to engage with math due to a lack of personalized guidance, step-by-step explanations, and relatable examples. Traditional tutoring methods are often inaccessible, expensive, or fail to adapt to the unique learning pace and needs of each student. Advanced AI systems like Large Language Models (LLMs) are a possible solution, however they are prone to errors in solving mathematical problems do to the tokenization process and architecture of the model, leading to further frustration and mistrust.

Number Ninja aims to address these challenges by developing an intelligent, math-tutoring AI model specifically designed for high school students. By leveraging a combination of advanced vector databases, chain-of-thought reasoning, and prompt engineering, the model provides step-by-step problem-solving support. It not only guides students through solutions interactively but also adapts to their individual learning needs, making math more approachable, reliable, and engaging.

## Background

Vector embeddings are a way of representing discrete data in a continuous vector space of n dimensions. This process transforms data into a numerical format that preserves the relationships and structure within the data. By mapping data to a vector embedding space, these embeddings allow AI algorithms, such as neural networks, to interpret and utilize the underlying patterns and meanings within the data.

In transformer neural networks, the primary architecture for modern-day LLMs, the encoder is responsible for taking input data (such as text) and processing it through multiple layers of self-attention and feed-forward neural networks. Through this process, the encoder creates high-dimensional vector representations that capture the semantic meaning and contextual information of the input data. These embeddings can then be used for various downstream tasks, such as language translation, sentiment analysis, and more. However, it is important to note that not all transformer-based neural networks use encoders; vector embeddings can be generated in other ways as well.

## Cosine similarity and its role in context
There are many ways to determine the congruency of a pair of vectors, one of them being cosine similarity. Cosine similarity is used to determine how similar the direction of the vectors is. The closer this value is to 1, the more similar their direction. This measure is crucial for assessing the contextual relevance between different pieces of data within the embedding space. Words with more association have a higher similarity.

![Cosine Similarity applied.](https://github.com/edwardduda/NeuroContext/blob/174930272101e329650ac97317969644adc38f3a/1_jptD3Rur8gUOftw-XHrezQ-2342057285.png)

## Heirarchical Navigable Small World

Hierarchical NSW (Navigable Small World) graph search method greatly improves upon earlier approaches for approximate nearest neighbor searches, particularly in handling high-dimensional data like vector embeddings. It is robust and performs consistently well across different types of datasets and doesn't require tailoring for specific problems. This makes it highly practical for real-world applications like large vector databases. While there are opportunities to further enhance its efficiency and scalability, it already sets a new standard in terms of speed and accuracy.

![HNSW.](readmeimgs/hnswdiagram.png)

## Installation and Setup

1. This project requires an API key from Groq, (https://groq.com/) which supports free fast-inference for open-source models this project using Llama3.1 70B, and a vector database either hosted locally or in the cloud.

pip install Flask
pip install groq import Groq
pip install pandas as pd
pip install sentence_transformers
pip install dotenv
pip install flask_cors import CORS
pip install sqlalchemy

Set up a virtual environment, using either conda or venv and install packages. Python 3.11 was used for this project. 

## Usage and Approach
A JavaScript application is used to communicate with the APIs via Python and mySQL. The student inputs a math problem, which is encoded, and an HNSW search is performed inside the vector database to find math problems most similar to the student's query. Each problem in the database has a solution that uses a chain-of-thought approach, meaning the solution is broken down step-by-step to reach the answer. This approach improves reliability, as LLMs are known to struggle with math. Using prompt engineering, the LLM uses these examples as a guide to solve the student's problem. The model is instructed not to provide the solution immediately but to go step-by-step, waiting for the user's validation at each step. The model's output is generated in TeX format and converted to a more legible answer when appropriate. Things like age and grade level are initially recorded so they can be used in the prompt, as well as previous responses.

    Example:
      response_params = f"user=u,u.first_name={self.user_firstname},u.age={self.user_age},u.grade_level={self.user_gradelvl},math_equation_format=Tex,confirm-each-step,response=math_tutor"

      for i in range(context_df.shape[0]):
            respondent = str(context_df.at[i, 'respondent_id']) + ".out="
            context += respondent + str(context_df.at[i, 'message']) + ","

        # Include the user's input and format the prompt properly
        prompt = f"{response_params}\n\nContext: {context}\n\nUser Input: {user_input}\n\nResponse:"
        return prompt

## Future Implementations
Things like remember the name, age, current courses, and grade_lvl could be officially recorded and stored properly in a secure database. This wasn't implemented in this case because of the outside of the scope of the project. The front-end of the application would be reworked by someone with more experience creating a professional-looking UI and debugging the javascript. Full disclosure, ChatGPT and Clause 3.5 Sonnet were used to generate the javascript code. A front-end developer would be a better choice to improve functionality and security.

There is awareness about the legal hurdles of recording information, especially from minors, however it shouldn't be too complex of issue since simple information like name, age, and grade level are solely for the purpose of aiding the model.

## Conclusion
Overall, the project was successful implemented and offers a solution for students who either don't have the academic or financial support to learn math well. It supports advanced subjects like calculus, combinatorics, and number theroy alongside the traditional high school math subjects.

## References and Acknowledgements

Malkov, Y. A., & Yashunin, D. A. (2018). Efficient and robust approximate nearest neighbor search using hierarchical navigable small world graphs. arXiv preprint arXiv:1603.09320. Retrieved from https://arxiv.org/abs/1603.09320

Groq. (n.d.). Groq AI accelerators for high-performance computing. Retrieved January 8, 2025, from https://www.groq.com





