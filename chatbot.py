import sys
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample dataset of health and nutrition FAQs
faq_data = [
    {
        "question": "What are the best foods for weight loss?",
        "answer": "Foods rich in fiber, protein, and healthy fats like vegetables, lean meats, and nuts are good for weight loss."
    },
    {
        "question": "How much water should I drink daily?",
        "answer": "It is generally recommended to drink about 8 glasses (2 liters) of water daily, but needs vary by individual."
    },
    {
        "question": "What are healthy hydration tips?",
        "answer": "Drink water regularly throughout the day, avoid sugary drinks, and consume fruits and vegetables with high water content."
    },
    {
        "question": "How can I gain weight healthily?",
        "answer": "Increase calorie intake with nutrient-dense foods like nuts, dairy, whole grains, and strength training exercises."
    },
    {
        "question": "What exercises help with weight control?",
        "answer": "A combination of cardio, strength training, and flexibility exercises helps maintain a healthy weight."
    },
    {
        "question": "What is a balanced diet?",
        "answer": "A balanced diet includes a variety of foods from all food groups: fruits, vegetables, grains, protein, and dairy."
    },
    {
        "question": "How important is nutrition for overall health?",
        "answer": "Good nutrition is essential for energy, immune function, and preventing chronic diseases."
    },
    {
        "question": "Can I get nutrition advice from this chatbot?",
        "answer": "Yes, this chatbot provides scientifically backed nutrition advice based on reliable sources."
    },
    {
        "question": "What should I eat before exercising?",
        "answer": "Eat a light meal with carbohydrates and protein about 1-3 hours before exercise."
    },
    {
        "question": "How often should I exercise for good health?",
        "answer": "Aim for at least 150 minutes of moderate aerobic activity or 75 minutes of vigorous activity per week."
    }
]

def preprocess_text(text):
    # Lowercase and remove punctuation
    text = text.lower()
    return text.translate(str.maketrans('', '', string.punctuation))

def main():
    print("Welcome to the Health and Nutrition FAQ Chatbot!")
    print("Ask your questions about diet, hydration, exercise, and nutrition.")
    print("Type 'exit' to quit.\n")

    questions = [preprocess_text(item["question"]) for item in faq_data]
    answers = [item["answer"] for item in faq_data]

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(questions)

    while True:
        user_input = input("You: ")
        if user_input.strip().lower() == "exit":
            print("Chatbot: Goodbye! Stay healthy.")
            break

        user_query = preprocess_text(user_input)
        user_vec = vectorizer.transform([user_query])

        similarities = cosine_similarity(user_vec, tfidf_matrix)
        max_sim_index = similarities.argmax()
        max_sim_score = similarities[0, max_sim_index]

        if max_sim_score < 0.1:
            print("Chatbot: Sorry, I don't have an answer for that. Please try asking something else.")
        else:
            print(f"Chatbot: {answers[max_sim_index]}")

if __name__ == "__main__":
    main()
