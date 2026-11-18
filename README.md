# CodeAlpha FAQ Chatbot

An FAQ chatbot built for **CodeAlpha AI Internship – Task 2**. It matches a user's question to the most similar FAQ and replies with that FAQ's answer, through a simple Streamlit chat UI.

## Topic

**E-commerce customer support.** The chatbot answers frequently asked questions of an online store, covering:

- Greetings and small talk (hello, how are you, what is your name)
- Return policy and refunds
- Order tracking
- Shipping times and international shipping
- Account help (password reset)
- Contacting customer support
- Payment methods
- Cancelling an order
- Damaged or wrong items
- Discount and coupon codes

## How it works

1. **FAQ dataset** – customer-support FAQs (returns, shipping, tracking, password reset, contact, payments, cancellation, damaged items, discounts), each with several question variants.
2. **Preprocessing (NLTK)** – lowercasing, punctuation removal, tokenization, stopword removal and lemmatization.
3. **Matching** – TF-IDF vectors (unigrams + bigrams) compared with **cosine similarity**.
4. **Response** – the best-matching answer is shown. If the similarity is below the threshold (`0.25`), a polite fallback message is returned.

## Project structure

```
CodeAlpha_FAQ_Chatbot/
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

NLTK data (`punkt`, `stopwords`, `wordnet`) is downloaded automatically on first run.

## Suggested questions

The sidebar lists 10 suggested questions, all of which the bot can answer. Click one to ask it instantly, or type your own question in the chat box.

## Example questions

- "What is your return policy?"
- "Where is my package?"
- "I forgot my password"
- "Do you ship abroad?"

## Customizing

Edit the `FAQS` list in `app.py` to use your own questions and answers, and tune `SIMILARITY_THRESHOLD` to make matching stricter or looser.