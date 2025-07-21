import streamlit as st
import json
import torch
import numpy as np
from sentence\_transformers import SentenceTransformer
from utils.embedding import load\_dataset, compute\_question\_embeddings
from utils.course\_query import extract\_course\_query, get\_courses\_for\_query, load\_course\_data
from utils.greetings import (
is\_greeting, greeting\_responses,
extract\_course\_code, get\_course\_by\_code,
is\_small\_talk, small\_talk\_response
)

@st.cache\_resource
def load\_all():
model = SentenceTransformer("all-MiniLM-L6-v2")
df = load\_dataset("data/crescent\_qa.json")
embeddings = compute\_question\_embeddings(df\["question"].tolist(), model)
course\_data = load\_course\_data("data/course\_data.json")
return model, df, embeddings, course\_data

model, df, embeddings, course\_data = load\_all()

def find\_best\_match(user\_question, model, embeddings, df, threshold=0.6):
from sentence\_transformers.util import cos\_sim
user\_embedding = model.encode(user\_question.strip().lower(), convert\_to\_tensor=True)
cosine\_scores = cos\_sim(user\_embedding, embeddings)\[0]
best\_score = torch.max(cosine\_scores).item()
best\_idx = torch.argmax(cosine\_scores).item()
if best\_score >= threshold:
return df.iloc\[best\_idx]\["answer"]
return None

st.set\_page\_config(page\_title="Crescent University Chatbot", layout="centered")
st.title("🎓 Crescent University Chatbot")
st.markdown("Ask me anything about departments, courses, or general university info!")

if "chat" not in st.session\_state:
st.session\_state.chat = \[]
if "bot\_greeted" not in st.session\_state:
st.session\_state.bot\_greeted = False

USER\_AVATAR = "🧑‍💻"
BOT\_AVATAR = "🎓"

user\_input = st.chat\_input("Type your question here...")
if user\_input:
st.session\_state.chat.append({"role": "user", "text": user\_input})
normalized\_input = user\_input.lower()

```
# Greeting check
if is_greeting(user_input) and not st.session_state.bot_greeted:
    response = greeting_responses(user_input)
    st.session_state.bot_greeted = True

# Small talk detection
elif is_small_talk(user_input):
    response = small_talk_response(user_input)

# Course code lookup
else:
    course_code = extract_course_code(user_input)
    if course_code:
        course_response = get_course_by_code(course_code, course_data)
        if course_response:
            response = f"📘 *Here’s the info for* `{course_code}`:\n\n{course_response}"
        else:
            response = f"🤔 I couldn't find any details for `{course_code}`. Please check the code and try again."
    else:
        general_keywords = [
            "admission", "requirement", "fee", "tuition", "duration", "length",
            "cut off", "hostel", "accommodation", "location", "accreditation"
        ]
        if any(word in normalized_input for word in general_keywords):
            result = find_best_match(user_input, model, embeddings, df)
        else:
            query_info = extract_course_query(user_input)
            if query_info and any([query_info.get("department"), query_info.get("level"), query_info.get("semester")]):
                result = get_courses_for_query(query_info, course_data)
            else:
                result = find_best_match(user_input, model, embeddings, df)

        if result:
            response = f"✨ Here’s what I found:\n\n{result}"
        else:
            response = "😕 I couldn’t find an answer to that. Try rephrasing it?"

st.session_state.chat.append({"role": "bot", "text": response})
```

for message in st.session\_state.chat:
avatar = USER\_AVATAR if message\["role"] == "user" else BOT\_AVATAR
with st.chat\_message(message\["role"], avatar=avatar):
st.markdown(message\["text"])
