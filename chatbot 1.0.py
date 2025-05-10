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
    },
    {
        "question": "Are carbohydrates bad for you?",
        "answer": "No, carbohydrates are a primary energy source. Choose complex carbs like whole grains over simple sugars."
    },
    {
        "question": "What are healthy snack options?",
        "answer": "Healthy snacks include fruits, nuts, yogurt, and whole-grain crackers."
    },
    {
        "question": "How does fiber benefit my health?",
        "answer": "Fiber aids digestion, helps maintain blood sugar levels, and can lower cholesterol."
    },
    {
        "question": "What is the role of protein in the body?",
        "answer": "Protein is essential for building and repairing tissues, and it's a building block of bones, muscles, and skin."
    },
    {
        "question": "How can I reduce my sugar intake?",
        "answer": "Limit sugary drinks, read food labels, and choose natural sweeteners like fruits."
    },
    {
        "question": "What are the signs of dehydration?",
        "answer": "Common signs include dry mouth, fatigue, dizziness, and dark-colored urine."
    },
    {
        "question": "Is it necessary to take vitamin supplements?",
        "answer": "If you have a balanced diet, supplements may not be necessary, but consult a healthcare provider for personalized advice."
    },
    {
        "question": "What foods are high in iron?",
        "answer": "Red meat, beans, lentils, spinach, and fortified cereals are good iron sources."
    },
    {
        "question": "How can I improve my metabolism?",
        "answer": "Regular exercise, adequate sleep, and eating protein-rich foods can boost metabolism."
    },
    {
        "question": "What is the importance of breakfast?",
        "answer": "Breakfast kickstarts your metabolism and provides energy for the day."
    },
    {
        "question": "Can I lose weight without exercising?",
        "answer": "Weight loss is primarily about calorie intake, but exercise helps maintain muscle mass and overall health."
    },
    {
        "question": "What are trans fats?",
        "answer": "Trans fats are unhealthy fats found in some processed foods; they increase the risk of heart disease."
    },
    {
        "question": "How much salt should I consume daily?",
        "answer": "The recommended limit is less than 2,300 mg of sodium per day."
    },
    {
        "question": "What are antioxidants?",
        "answer": "Antioxidants are compounds that protect your cells from damage caused by free radicals."
    },
    {
        "question": "Are organic foods healthier?",
        "answer": "Organic foods have fewer pesticides, but nutritional differences are minimal."
    },
    {
        "question": "What is BMI?",
        "answer": "BMI stands for Body Mass Index, a measure of body fat based on height and weight."
    },
    {
        "question": "How can I manage portion sizes?",
        "answer": "Use smaller plates, read serving sizes on labels, and avoid eating straight from the package."
    },
    {
        "question": "What are the benefits of omega-3 fatty acids?",
        "answer": "Omega-3s support heart health, reduce inflammation, and may improve brain function."
    },
    {
        "question": "Is skipping meals bad for you?",
        "answer": "Skipping meals can lead to overeating later and may affect blood sugar levels."
    },
    {
        "question": "What are probiotics?",
        "answer": "Probiotics are beneficial bacteria that support gut health."
    },
    {
        "question": "How does alcohol affect nutrition?",
        "answer": "Excessive alcohol can interfere with nutrient absorption and add empty calories."
    },
    {
        "question": "What is the glycemic index?",
        "answer": "It's a measure of how quickly a food raises blood sugar levels."
    },
    {
        "question": "Are energy drinks safe?",
        "answer": "They can be high in caffeine and sugar; moderation is key."
    },
    {
        "question": "What is mindful eating?",
        "answer": "Mindful eating involves paying full attention to the experience of eating and drinking."
    },
    {
        "question": "How can I read nutrition labels effectively?",
        "answer": "Focus on serving size, calories, and nutrients like fiber, sugars, and fats."
    },
    {
        "question": "What are whole foods?",
        "answer": "Whole foods are minimally processed and free from additives or artificial substances."
    },
    {
        "question": "Can I trust nutrition information online?",
        "answer": "Always verify information with reputable sources or consult healthcare professionals."
    },
    {
        "question": "What is the DASH diet?",
        "answer": "The DASH diet emphasizes fruits, vegetables, whole grains, and lean proteins to lower blood pressure."
    },
    {
        "question": "How does stress affect eating habits?",
        "answer": "Stress can lead to overeating or undereating and cravings for unhealthy foods."
    },
    {
        "question": "What are empty calories?",
        "answer": "Foods high in calories but low in nutrients, like sugary drinks and snacks."
    },
    {
        "question": "How can I ensure I'm getting enough calcium?",
        "answer": "Consume dairy products, leafy greens, and fortified foods."
    },
    {
        "question": "What is intermittent fasting?",
        "answer": "It's an eating pattern that cycles between periods of fasting and eating."
    },
    {
        "question": "Are plant-based diets healthy?",
        "answer": "Yes, when well-planned, they can provide all necessary nutrients."
    },
    {
        "question": "How does sleep affect nutrition?",
        "answer": "Poor sleep can disrupt hunger hormones and lead to unhealthy eating habits."
    },
    {
        "question": "What are the benefits of meal prepping?",
        "answer": "Meal prepping saves time, ensures balanced meals, and helps control portions."
    },
    {
        "question": "Can I eat healthy on a budget?",
        "answer": "Yes, by planning meals, buying in bulk, and choosing seasonal produce."
    },
    {
        "question": "What is the role of water in the body?",
        "answer": "Water regulates body temperature, transports nutrients, and removes waste."
    },
    {
        "question": "How can I reduce cholesterol through diet?",
        "answer": "Eat more fiber, reduce saturated fats, and include healthy fats like omega-3s."
    },
    {
        "question": "What are the signs of a nutrient deficiency?",
        "answer": "Symptoms vary but can include fatigue, weakness, and poor immune function."
    },
    {
        "question": "Is it better to eat three meals or several small meals a day?",
        "answer": "It depends on individual preferences and health goals; both can be effective."
    },
    {
        "question": "What are superfoods?",
        "answer": "Superfoods are nutrient-rich foods considered beneficial for health, like berries and leafy greens."
    },
    {
        "question": "How does fiber aid in weight loss?",
        "answer": "Fiber increases satiety, helping you feel full longer and reducing overall calorie intake."
    },
    {
        "question": "What is the Mediterranean diet?",
        "answer": "A diet emphasizing fruits, vegetables, whole grains, olive oil, and lean proteins."
    },
    {
        "question": "Are smoothies healthy?",
        "answer": "They can be, if made with whole fruits, vegetables, and without added sugars."
    },
    {
        "question": "How can I boost my immune system through diet?",
        "answer": "Eat a variety of fruits, vegetables, lean proteins, and stay hydrated."
    },
    {
        "question": "What is the role of vitamin D?",
        "answer": "Vitamin D helps with calcium absorption and supports bone health."
    },
    {
        "question": "Can diet affect mental health?",
        "answer": "Yes, a balanced diet can improve mood and cognitive function."
    },
    {
        "question": "What are the benefits of eating breakfast?",
        "answer": "Breakfast provides energy, improves concentration, and can aid in weight management."
    },
    {
        "question": "How can I reduce sugar cravings?",
        "answer": "Eat balanced meals, manage stress, and get adequate sleep."
    },
    {
        "question": "What is a low-carb diet?",
        "answer": "A diet that limits carbohydrates, focusing on proteins and fats."
    },
    {
        "question": "Are dairy products necessary for calcium intake?",
        "answer": "Not necessarily; leafy greens, fortified foods, and other sources can provide calcium."
    }
]

def preprocess_text(text):
    # Lowercase and remove punctuation
    text = text.lower()
    return text.translate(str.maketrans('', '', string.punctuation))

def get_answer(question):
    questions = [preprocess_text(item["question"]) for item in faq_data]
    answers = [item["answer"] for item in faq_data]

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(questions)

    user_query = preprocess_text(question)
    user_vec = vectorizer.transform([user_query])

    similarities = cosine_similarity(user_vec, tfidf_matrix)
    max_sim_index = similarities.argmax()
    max_sim_score = similarities[0, max_sim_index]

    if max_sim_score < 0.1:
        return "Sorry, I don't have an answer for that. Please try asking something else."
    else:
        return answers[max_sim_index]

if __name__ == "__main__":
    print("Welcome to the Health & Nutrition Chatbot! Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        response = get_answer(user_input)
        print("Chatbot:", response)
