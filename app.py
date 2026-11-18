import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# --- NLTK SETUP & DOWNLOADS ---
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
try:
  nltk.download('punkt_tab')
except:
  pass

# --- FAQ DATASET (Including Greetings) ---
faqs = [
    {
        "question": "hello hi hey",
        "answer": (
            "Hello! Welcome to our support. How can I help you with our"
            " products or policies today?"
        ),
    },
    {
        "question": "how are you",
        "answer": (
            "I'm just a virtual assistant, but I'm doing great and ready to"
            " help you! What can I assist you with?"
        ),
    },
    {
        "question": "what is your name",
        "answer": (
            "I am your automated customer support assistant, here to answer your"
            " FAQs!"
        ),
    },
    {
        "question": "What is your return policy?",
        "answer": (
            "You can return any unused item within 30 days of purchase for a"
            " full refund with the original receipt."
        ),
    },
    {
        "question": "How can I track my order?",
        "answer": (
            "Once your order ships, we will send you a confirmation email"
            " containing your tracking link and carrier details."
        ),
    },
    {
        "question": "What are your shipping times?",
        "answer": (
            "Standard shipping typically takes 3-5 business days, while"
            " expedited shipping takes 1-2 business days."
        ),
    },
    {
        "question": "Do you offer international shipping?",
        "answer": (
            "Yes, we ship to over 50 countries worldwide. Shipping fees and"
            " delivery estimates are calculated at checkout."
        ),
    },
    {
        "question": "How do I reset my password?",
        "answer": (
            "Click on 'Forgot Password' on the login page, enter your"
            " registered email, and follow the instructions sent to your inbox."
        ),
    },
    {
        "question": "How can I contact customer support?",
        "answer": (
            "You can reach our support team 24/7 via live chat on our website"
            " or by emailing support@example.com."
        ),
    },
]


# --- TEXT PREPROCESSING FUNCTION ---
def preprocess_text(text):
  text = text.lower()
  text = re.sub(r'[^a-zA-Z\s]', '', text)
  tokens = nltk.word_tokenize(text)
  lemmatizer = WordNetLemmatizer()
  stop_words = set(stopwords.words('english'))
  cleaned_tokens = [
      lemmatizer.lemmatize(token)
      for token in tokens
      if token not in stop_words
  ]
  return ' '.join(cleaned_tokens)


# --- VECTORIZATION & SIMILARITY ENGINE ---
processed_faq_questions = [preprocess_text(faq['question']) for faq in faqs]

vectorizer = TfidfVectorizer()
faq_vectors = vectorizer.fit_transform(processed_faq_questions)


def get_best_answer(user_query, threshold=0.05):  # Threshold lowered to 0.05
  processed_query = preprocess_text(user_query)
  query_vector = vectorizer.transform([processed_query])
  similarities = cosine_similarity(query_vector, faq_vectors).flatten()

  best_match_idx = similarities.argmax()
  best_score = similarities[best_match_idx]

  if best_score < threshold:
    return (
        "I'm sorry, I couldn't quite understand that. You can reach our"
        ' support team at support@example.com for further assistance!'
    )

  return faqs[best_match_idx]['answer']


# --- STREAMLIT CHAT UI ---
st.set_page_config(page_title="FAQ Assistant", page_icon="💬", layout="centered")

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
    "Ask me anything about our return policy, shipping times, account help, and"
    " more!"
)

if 'messages' not in st.session_state:
  st.session_state.messages = [{
      'role': 'assistant',
      'content': (
          'Hello! How can I help you today? Try asking about our return policy'
          ' or shipping times.'
      ),
  }]

for message in st.session_state.messages:
  with st.chat_message(message['role']):
    st.markdown(message['content'])

if user_input := st.chat_input('Type your question here...'):
  st.session_state.messages.append({'role': 'user', 'content': user_input})
  with st.chat_message('user'):
    st.markdown(user_input)

  bot_response = get_best_answer(user_input)

  st.session_state.messages.append(
      {'role': 'assistant', 'content': bot_response}
  )
  with st.chat_message('assistant'):
    st.markdown(bot_response)