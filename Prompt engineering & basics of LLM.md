# 🧠 GenAI Interview Prep Notes

## Q1 - What is the difference between Predictive/Discriminative AI and Generative AI?

### 🎯 Simple Definitions:

| Type of AI | What it does | Analogy | Industry Example |
|------------|--------------|---------|------------------|
| **Predictive / Discriminative AI** | Learns to **classify** or **predict** labels from data | Like a teacher grading exams — it says whether an answer is right or wrong | A model that predicts customer churn or classifies fraudulent vs. legitimate transactions |
| **Generative AI** | Learns to **generate** new data (text, images, code, etc.) | Like an artist creating a new painting or a writer drafting an article | Chatbots that write reports, AI that generates marketing copy, or creates code snippets |

---

### 🧠 How They Work:

#### Predictive AI:
- Trained on labeled data (e.g., emails labeled spam or not spam)
- Learns boundaries to distinguish one class from another
- Example models: Logistic Regression, Random Forest, XGBoost

#### Generative AI:
- Trained on **huge amounts of unlabeled data**
- Learns to generate content that resembles the training data
- Powered by **Large Language Models (LLMs)** like GPT, Claude, LLaMA

---

### 🏭 AWS Industry-Level Examples:

#### Predictive AI:
- Forecasting sales using Amazon Forecast
- Predicting server failures using Amazon Lookout for Equipment
- Classifying support tickets with Amazon Comprehend

#### Generative AI:
- Generating financial summaries from raw numbers using Amazon Q
- Creating personalized marketing content using Amazon Bedrock
- Automating code review using Amazon CodeWhisperer

---

### 🧠 High-School Analogy:
- **Predictive AI** is like taking a multiple-choice test — it selects the right answer from existing options.
- **Generative AI** is like writing an essay — it comes up with something new based on what it has learned.

---

## Q2 - What is an LLM and how are LLMs trained?

### 🧠 What is an LLM?

**LLM stands for Large Language Model.**

It’s an advanced type of AI model trained to understand, process, and generate human-like text. These models are "large" because they are trained on **huge volumes of text data** and have **billions of parameters** (internal weights that help make predictions).

---

### 📘 High-School Analogy:

Imagine you’re trying to write like Shakespeare.  
You read thousands of pages of Shakespeare’s plays and poems. Eventually, you get so good that you can write new lines that sound just like him.

That’s what an LLM does — it reads **massive amounts of text** and **learns patterns in language**, so it can generate new text that sounds like it came from a human.

---

### 🏭 Industry-Level Analogy (for Solution Architects):

Think of an LLM like an ultra-smart **BI analyst** that has read **every company report, customer conversation, marketing material, and technical doc** in your enterprise.  
When asked a question like:  
> “Summarize our top three challenges in APAC last quarter,”  
it can generate a coherent answer because it has "seen" so much relevant data during training.

---

### 🛠️ How are LLMs trained?

LLMs are trained in **two main phases**:

---

#### 1. 📖 Pretraining (General Knowledge Phase)

- **What happens:**  
  The model is fed a massive dataset (like books, websites, Wikipedia, forums, code, etc.).
  
- **Objective:**  
  It learns how language works — grammar, facts, reasoning, how questions are asked/answered, etc.

- **Technique:**  
  Self-supervised learning: The model is asked to **predict the next word** in a sentence (e.g., "The sky is ___").

- **Example:**  
  Given “AWS Glue is used for,” it learns that “ETL” is a good next word by seeing it frequently in similar contexts.

---

#### 2. 🧠 Fine-Tuning (Specialization Phase)

- **What happens:**  
  The base model is refined on a **smaller, domain-specific dataset** or trained with **human feedback**.

- **Techniques used:**
  - **Supervised Fine-Tuning:** Feed labeled Q&A pairs.
  - **Reinforcement Learning with Human Feedback (RLHF):** Humans rate model responses to improve quality.

- **Enterprise Example:**  
  Fine-tuning a general model like LLaMA on your company's financial data so it can accurately answer:
  > “What was our YoY revenue growth across product lines?”

---

### 🧰 Tools and Ecosystem for Training LLMs

| Component         | Description                                                                 |
|------------------|-----------------------------------------------------------------------------|
| **Foundation Models** | Pretrained models like GPT, Claude, LLaMA, Mistral                      |
| **Frameworks**         | Hugging Face Transformers, DeepSpeed, PyTorch, TensorFlow               |
| **Infrastructure**     | GPU clusters, AWS Trainium/Inferentia, Amazon SageMaker                 |
| **Data Sources**       | Common Crawl, PubMed, GitHub, enterprise docs, call transcripts         |

---

### 🚀 Bonus Tip for Interviews:

> "LLMs are trained using a combination of statistical prediction and human feedback — like teaching a child how to speak by letting them read everything in a library and then gently correcting their responses when they speak."

---

# Q3 - What is a Token in a Language Model?

In the context of language models, a **token** is a unit of text that the model processes during training or inference. Tokens are the basic building blocks that the model uses to understand and generate text. The way tokens are defined and used depends on the tokenization method employed by the model.

## Tokenization

Tokenization is the process of breaking down text into smaller, manageable pieces (tokens) that the model can process. These tokens can represent:
- A single word (e.g., "chat")
- Part of a word (e.g., "unhappiness" might be split into "un" and "happiness")
- A character or a subword unit, depending on the tokenization approach

Tokenization is a critical step because it converts raw text into a format the model can work with, enabling it to learn patterns and generate coherent responses.

## Types of Tokenization

Different models use different methods of tokenization:
- **Word-level tokenization**: Breaks down the text into words (simple but not ideal for languages with many compound words).
- **Subword tokenization**: Breaks down words into smaller pieces (subwords) to handle out-of-vocabulary words more effectively.
- **Character-level tokenization**: Breaks text down into individual characters (less common in modern models, but can work well for languages with complex morphology).

## Tokens in Language Models

- A token can be a single character, a subword, or even a complete word.
- In modern language models like GPT-4 or LLaMA, **subword tokenization** is often used, where words are split into smaller meaningful units, allowing the model to handle rare or unknown words better.

For example:
- The word "chat" might be a single token.
- A more complex word like "unhappiness" might be split into multiple tokens like "un," "happiness," or even smaller subword units like "un," "hap," "pi," "ness."

## Token Limit

Most language models, including GPT, have a **token limit** for the number of tokens they can process in one go. This limit includes both the input and the output tokens. For instance, GPT-4 has token limits that can go up to 8,000 tokens or more, depending on the model variant. When using a language model, it's important to consider this token limit, as longer inputs or outputs might exceed the capacity of the model.

### Example:
- If you input a text of 3,000 tokens and ask for a 5,000-token response, the total would exceed the token limit, and the model might either truncate the response or fail to process the input completely.

## Conclusion

Understanding tokens and tokenization is fundamental to working with language models. Tokenization breaks down text into manageable units, allowing models to process and generate human-like text. The token limit is a critical consideration when working with large inputs or outputs in models like GPT-4, LLaMA, and other large language models.



# Q4 - 🧮 How to Estimate the Cost of Running a SaaS-Based vs. Open-Source LLM

Let’s imagine you're trying to run a lemonade stand 🍋. You can either:
- **Buy pre-made lemonade** from a vendor (SaaS), or
- **Make your own lemonade** at home (Open Source).

Both have costs—just in different ways. Let’s break it down using the LLM (Large Language Model) context.

---

## 🏢 Option 1: SaaS-Based LLM (e.g., OpenAI GPT, Claude, Amazon Bedrock)

You're **paying a vendor** to use their model via API.

### 🔍 How to Estimate the Cost:
1. **Know the model pricing** (e.g., per 1,000 tokens):
   - Example: OpenAI GPT-4 might cost $0.03 per 1K prompt tokens and $0.06 per 1K completion tokens.

2. **Estimate your usage**:
   - One sentence ≈ 20–30 tokens.
   - A typical user session might use 1,000–2,000 tokens.

3. **Multiply the cost**:
   ```
   Cost = (Prompt tokens + Completion tokens) ÷ 1,000 × Rate
   Example: 2,000 tokens × $0.03 = $0.06 per interaction
   ```

4. **Factor in concurrency & traffic**:
   - For 10,000 users/day, cost could be ~$600/day or ~$18,000/month (based on model, usage patterns).

✅ **Pros**: No infra maintenance, easy scaling, secure  
⚠️ **Cons**: Cost grows linearly with usage

---

## 🏗️ Option 2: Open Source LLM (e.g., LLaMA, Mistral, DeepSeek)

You're **hosting the model yourself**—either on-prem or in the cloud.

### 🔍 How to Estimate the Cost:
1. **Choose the model size**:
   - 7B, 13B, 70B parameters, etc.
   - Larger models = more GPU and memory required.

2. **Estimate Infra Needs**:
   - Example: LLaMA-13B may need 1×A100 GPU (80GB) or 2×T4s
   - On AWS: `ml.g5.2xlarge` (T4) ≈ $0.77/hr

3. **Include Other Costs**:
   - Inference server (e.g., vLLM, Text Generation Inference)
   - Storage (model weights ~10–40GB)
   - Load balancer, autoscaling, etc.

4. **Do the math**:
   ```
   1 GPU (24x7) × $0.77/hr ≈ $550/month
   + Storage + Networking ≈ $100–300
   ```

✅ **Pros**: Control, customization, no per-query fees  
⚠️ **Cons**: Ops overhead, upfront infra costs, need for AI/ML skills

---

## 🧠 Example: Comparing Both for a Use Case

| Use Case                        | SaaS LLM               | Open-Source LLM         |
|-------------------------------|------------------------|-------------------------|
| Daily Token Usage (1M tokens) | ~$30–$100/day          | ~$20–$50/day (infra)    |
| Startup Time                  | Instant                | Few hours/days setup    |
| Data Control                  | Limited                | Full                    |
| Scalability                   | Easy                   | Manual (or autoscale)   |
| Cost Predictability           | Pay-per-use            | Fixed infra + tuning    |

---

## 🧰 Bonus Tip (for AWS Solutions Architects 😉)

- Use **Amazon Bedrock** for SaaS-based models
- Use **Amazon SageMaker**, **EC2 (G5)**, or **EKS + Ray + vLLM** for hosting open models
- Consider **Amazon Inference Pricing Estimator** for precise cost modeling

