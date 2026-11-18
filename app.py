"""
CodeAlpha AI Internship - Task 2: FAQ Chatbot

Pipeline:
  1. FAQ dataset (several question variants per answer)
  2. NLTK preprocessing (lowercase, clean, tokenize, stopwords, lemmatize)
  3. TF-IDF vectorization + cosine similarity to find the closest FAQ
  4. Streamlit chat UI

Run with:  streamlit run app.py
"""

import re

import nltk
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Minimum cosine similarity needed to accept a match.
SIMILARITY_THRESHOLD = 0.25

FALLBACK_ANSWER = (
    "I'm sorry, I couldn't quite understand that. Could you rephrase your "
    "question? You can also reach our support team at support@example.com."
)

# Questions shown in the sidebar so users know what they can ask.
SUGGESTED_QUESTIONS = [
    "What is your return policy?",
    "How can I track my order?",
    "What are your shipping times?",
    "Do you offer international shipping?",
    "How do I reset my password?",
    "How can I contact customer support?",
    "What payment methods do you accept?",
    "How can I cancel my order?",
    "What if I received a damaged item?",
    "Do you have any discount codes?",
]

# --------------------------------------------------------------------------
# FAQ DATASET
# Each FAQ has several question variants so matching is more robust.
# --------------------------------------------------------------------------
FAQS = [
    {
        "questions": ["hello", "hi", "hey", "good morning", "good evening"],
        "answer": (
            "Hello! Welcome to our support. How can I help you with our "
            "products or policies today?"
        ),
    },
    {
        "questions": ["how are you", "how are you doing", "how is it going"],
        "answer": (
            "I'm just a virtual assistant, but I'm doing great and ready to "
            "help you! What can I assist you with?"
        ),
    },
    {
        "questions": ["what is your name", "who are you", "are you a bot"],
        "answer": (
            "I am your automated customer support assistant, here to answer "
            "your FAQs!"
        ),
    },
    {
        "questions": [
            "What is your return policy?",
            "How can I return an item?",
            "Can I get a refund?",
            "I want to return my product",
        ],
        "answer": (
            "You can return any unused item within 30 days of purchase for a "
            "full refund with the original receipt."
        ),
    },
    {
        "questions": [
            "How can I track my order?",
            "Where is my order?",
            "Track my package",
            "Where is my package?",
            "Where is my parcel?",
            "Order tracking link",
        ],
        "answer": (
            "Once your order ships, we will send you a confirmation email "
            "containing your tracking link and carrier details."
        ),
    },
    {
        "questions": [
            "What are your shipping times?",
            "How long does delivery take?",
            "When will my order arrive?",
            "How fast is shipping?",
        ],
        "answer": (
            "Standard shipping typically takes 3-5 business days, while "
            "expedited shipping takes 1-2 business days."
        ),
    },
    {
        "questions": [
            "Do you offer international shipping?",
            "Do you ship to other countries?",
            "Can you deliver abroad?",
        ],
        "answer": (
            "Yes, we ship to over 50 countries worldwide. Shipping fees and "
            "delivery estimates are calculated at checkout."
        ),
    },
    {
        "questions": [
            "How do I reset my password?",
            "I forgot my password",
            "Can't log in to my account",
            "Change my password",
        ],
        "answer": (
            "Click on 'Forgot Password' on the login page, enter your "
            "registered email, and follow the instructions sent to your inbox."
        ),
    },
    {
        "questions": [
            "How can I contact customer support?",
            "I need to talk to a human",
            "What is your support email?",
            "Customer service contact",
        ],
        "answer": (
            "You can reach our support team 24/7 via live chat on our website "
            "or by emailing support@example.com."
        ),
    },
    {
        "questions": [
            "What payment methods do you accept?",
            "How can I pay for my order?",
            "Do you accept PayPal?",
            "Can I pay with a credit card?",
        ],
        "answer": (
            "We accept Visa, MasterCard, American Express, PayPal and Apple "
            "Pay. All payments are processed securely at checkout."
        ),
    },
    {
        "questions": [
            "How can I cancel my order?",
            "I want to cancel my order",
            "Can I cancel after placing an order?",
            "Cancel my purchase",
        ],
        "answer": (
            "You can cancel your order within 12 hours of placing it from the "
            "'My Orders' page. Once it has shipped, you can return it instead "
            "under our return policy."
        ),
    },
    {
        "questions": [
            "What if I received a damaged item?",
            "My item arrived broken",
            "I received the wrong product",
            "The product is defective",
        ],
        "answer": (
            "We're sorry about that! Contact our support team within 7 days "
            "of delivery with your order number and a photo, and we will send "
            "a replacement or issue a full refund."
        ),
    },
    {
        "questions": [
            "Do you have any discount codes?",
            "How do I use a coupon code?",
            "Are there any sales or promotions?",
            "Do you offer discounts?",
        ],
        "answer": (
            "Enter your coupon code in the 'Promo Code' box at checkout. "
            "Subscribe to our newsletter to get 10% off your first order and "
            "early access to sales."
        ),
    },
]


# --------------------------------------------------------------------------
# NLTK SETUP (cached, runs once, never crashes the app if offline)
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def setup_nltk():
    """Download required NLTK data only if missing. Returns a status dict."""
    resources = {
        "punkt": "tokenizers/punkt",
        "punkt_tab": "tokenizers/punkt_tab",
        "stopwords": "corpora/stopwords",
        "wordnet": "corpora/wordnet",
    }
    status = {}
    for name, path in resources.items():
        try:
            nltk.data.find(path)
            status[name] = True
        except LookupError:
            try:
                status[name] = bool(nltk.download(name, quiet=True))
            except Exception:
                status[name] = False
    return status


@st.cache_resource(show_spinner=False)
def get_nlp_tools():
    """Build stopword set and lemmatizer once."""
    setup_nltk()
    try:
        stop_words = set(stopwords.words("english"))
    except LookupError:
        stop_words = set()
    return stop_words, WordNetLemmatizer()


def tokenize(text):
    """Tokenize with NLTK; fall back to whitespace split if data is missing."""
    try:
        return nltk.word_tokenize(text)
    except LookupError:
        return text.split()


def lemmatize(lemmatizer, token):
    try:
        return lemmatizer.lemmatize(token)
    except LookupError:
        return token


# --------------------------------------------------------------------------
# TEXT PREPROCESSING
# --------------------------------------------------------------------------
def preprocess_text(text):
    """
    Lowercase -> remove punctuation/digits -> tokenize -> remove stopwords
    -> lemmatize.

    If removing stopwords would leave nothing (e.g. "how are you"), the
    stopwords are kept so the text still has a usable representation.
    """
    stop_words, lemmatizer = get_nlp_tools()

    text = re.sub(r"[^a-z\s]", " ", text.lower())
    tokens = tokenize(text)

    filtered = [t for t in tokens if t not in stop_words]
    if not filtered:  # all words were stopwords -> keep them
        filtered = tokens

    return " ".join(lemmatize(lemmatizer, t) for t in filtered)


# --------------------------------------------------------------------------
# VECTORIZATION & SIMILARITY ENGINE
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def build_engine():
    """Flatten FAQ variants, preprocess them and fit the TF-IDF vectorizer."""
    variant_texts = []
    variant_to_faq = []  # maps each variant to its FAQ index
    for faq_idx, faq in enumerate(FAQS):
        for question in faq["questions"]:
            variant_texts.append(preprocess_text(question))
            variant_to_faq.append(faq_idx)

    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(variant_texts)
    return vectorizer, matrix, variant_to_faq


def get_best_answer(user_query, threshold=SIMILARITY_THRESHOLD):
    """Return the answer of the most similar FAQ, or a fallback message."""
    if not user_query.strip():
        return FALLBACK_ANSWER

    vectorizer, matrix, variant_to_faq = build_engine()
    processed_query = preprocess_text(user_query)
    if not processed_query:
        return FALLBACK_ANSWER

    query_vector = vectorizer.transform([processed_query])
    similarities = cosine_similarity(query_vector, matrix).flatten()

    best_idx = int(similarities.argmax())
    if similarities[best_idx] < threshold:
        return FALLBACK_ANSWER

    return FAQS[variant_to_faq[best_idx]]["answer"]


# --------------------------------------------------------------------------
# STREAMLIT CHAT UI
# --------------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="FAQ Assistant", page_icon="💬", layout="centered"
    )

    st.markdown(
        """
        <style>
        .stChatMessage[data-testid="stChatMessage"]:nth-child(odd) {
            flex-direction: row-reverse;
            text-align: right;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("💬 Customer Support FAQ Bot")
    st.write(
        "Ask me anything about our return policy, shipping times, account "
        "help, and more!"
    )

    with st.sidebar:
        st.header("About")
        st.write(
            "FAQ chatbot built with NLTK preprocessing, TF-IDF and cosine "
            "similarity."
        )

        st.divider()
        st.subheader("💡 Suggested questions")
        st.caption("Click a question to ask it, or type your own.")
        for i, suggestion in enumerate(SUGGESTED_QUESTIONS):
            if st.button(suggestion, key=f"suggestion_{i}", use_container_width=True):
                st.session_state.pending_question = suggestion

        st.divider()
        if st.button("Clear chat"):
            st.session_state.pop("messages", None)
            st.rerun()

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Hello! How can I help you today? Try asking about our "
                    "return policy or shipping times."
                ),
            }
        ]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    typed_input = st.chat_input("Type your question here...")
    clicked_input = st.session_state.pop("pending_question", None)
    user_input = typed_input or clicked_input
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        bot_response = get_best_answer(user_input)

        st.session_state.messages.append(
            {"role": "assistant", "content": bot_response}
        )
        with st.chat_message("assistant"):
            st.markdown(bot_response)


if __name__ == "__main__":
    main()