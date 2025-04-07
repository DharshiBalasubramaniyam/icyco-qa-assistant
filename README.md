# QA assistant for Icyco

![image](https://github.com/user-attachments/assets/95088b62-225f-4cde-9876-93f978ce714c)


Icyco QA Assistant is an console based AI-powered Question Answering system built for the fictional ice cream shop **Icyco**. It uses Retrieval-Augmented Generation (RAG) to provide accurate answers based on company documents like policies, product information, and FAQs.

---

## 🧠 Features

- 💬 Ask questions about Icyco’s products, pricing, policies, etc.
- 📄 Ingests and processes internal documents (PDFs)
- 🔍 Retrieves relevant information using semantic search
- 🤖 Generates answers using Google Generative AI
- ⚡ Fast and responsive user experience with LangChain and Pinecone

---

## 🛠️ Tech Stack

- [LangChain](https://www.langchain.com/)
- [Pinecone](https://www.pinecone.io/)
- [Google Generative AI](https://ai.google.dev/)
- [Python](https://www.python.org/)

---

## 🚀 Getting Started

### 1. Fork and Clone the repo

```bash
git clone https://github.com/your-username/icyco-qa-assistant.git
cd icyco-qa-assistant
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set environment variables

Create a `.env` file similar to `.env.example` and provide necessary details.

### 4. Run the QA app

````bash
python app.py
````

Checkout my blog **[Getting Started with RAG by Building a QA Assistant](https://medium.com/@dharshib.8/getting-started-with-rag-by-building-a-qa-assistant-a72b9140b554)**, for more detail explanation of the implementation.

---

## Demo

```
You: Hi!
Assistant: Hi there! How can I make your day sweeter with some Icyco info? 🍦


You: What are the flavors icyco offers?
Assistant: We offer a wide range of flavors here at Icyco, including regular, gelato, sorbet, frozen yogurt, non-dairy, and vegan! 🍦 You can also browse our products by visiting https://icyco.vercel.app/products!


You: What about your delivery hours?
Assistant: We deliver ice cream every day from 10 am to 11 pm. Availability may vary by location.


You: Do you share user information for third parties?
Assistant: We do not sell your personal information. However, we may share your information with third-party service providers who assist with payment processing, delivery services, or website hosting. We may also share your information if required by law, such as in response to a court order or government request, and in the event of a merger or acquisition.


You: Who is the Owner of icyco?
Assistant: That’s a great question about Icyco, but I couldn’t find the answer in the info I have. You might want to reach out to Icyco directly for the most up-to-date details!


You: Exit
Assistant: Bye
```
