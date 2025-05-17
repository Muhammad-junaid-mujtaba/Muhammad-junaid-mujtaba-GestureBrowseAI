import streamlit as st
from langchain_groq import ChatGroq
from duckduckgo_search import DDGS
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
import os
import cv2
import mediapipe as mp
import pyautogui
import threading
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

prev_y, prev_x = None, None

# Set up LLM
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
llm = ChatGroq(model_name="llama3-8b-8192", groq_api_key=GROQ_API_KEY)

def search_duckduckgo(query, num_results=1):
    """Fetches search results from DuckDuckGo."""
    with DDGS() as ddgs:
        return list(ddgs.text(query, max_results=num_results))

def get_webpage_content(url):
    options = Options()
    options.add_experimental_option("detach", True)  # Keeps browser open
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.get(url)
    time.sleep(5)  # Shortened time for better performance
    return url

def summarize_results(query):
    """Fetches search results, extracts webpage content, and summarizes it."""
    results = search_duckduckgo(query)
    if not results:
        return "No search results found.", []
    
    first_result = results[0]
    url = first_result.get('href', '')
    page_content = get_webpage_content(url) if url else "No content available."
    
    summary_prompt = f"Summarize this webpage content in 2 lines:\n{page_content}"
    summary = llm.invoke(summary_prompt)
    return summary, results

def process_frame():
    global prev_y, prev_x
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(frame_rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                index_x = hand_landmarks.landmark[8].x
                index_y = hand_landmarks.landmark[8].y

                if prev_y is not None and prev_x is not None:
                    diff_y = prev_y - index_y
                    diff_x = prev_x - index_x
                    
                    scroll_speed = 20
                    if abs(diff_y) > 0.01:
                        pyautogui.scroll(scroll_speed if diff_y > 0 else -scroll_speed)
                    if abs(diff_x) > 0.01:
                        pyautogui.hscroll(-scroll_speed if diff_x > 0 else scroll_speed)

                prev_y, prev_x = index_y, index_x

        cv2.imshow("Hand Gesture Navigation", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# Start gesture processing in a separate thread
gesture_thread = threading.Thread(target=process_frame, daemon=True)
gesture_thread.start()

# Streamlit UI
st.set_page_config(page_title="DuckDuckGo AI Search", layout="wide")
st.title("🔍 DuckDuckGo AI-Powered Search")
query = st.text_input("Enter your search query:", "")

if st.button("Search") and query:
    with st.spinner("Fetching results..."):
        summary, results = summarize_results(query)

    st.subheader("🔎 Search Results:")
    for res in results:
        st.write(f"- [{res['title']}]({res['href']})")
    
    st.subheader("💡 AI Summary:")
    st.write(summary)

st.markdown("### 📱 Access this page from your mobile device using the IP and port shown in your terminal.")
