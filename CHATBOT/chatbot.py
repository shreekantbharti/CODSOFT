def get_response(user_input):
    user_input = user_input.lower()
    
    if "hello" in user_input or "hi" in user_input:
        return "Hello! How can I help you today?"
    
    elif "how are you" in user_input:
        return "I'm doing well, thank you for asking!"
    
    elif "bye" in user_input or "goodbye" in user_input:
        return "Goodbye! Have a great day!"
    
    elif "name" in user_input:
        return "My name is ChatBot!"
    
    elif "weather" in user_input:
        return "I'm sorry, I don't have access to weather information."
    
    elif "time" in user_input:
        return "I'm not configured to provide the current time."
    
    elif "help" in user_input:
        return "I can help you with basic conversation. Try asking about my name or saying hello!"
    
    else:
        return "I'm not sure how to respond to that. Try asking something else!"

def main():
    print("ChatBot: Hi! I'm your chatbot. Type 'bye' to exit.")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == "bye":
            print("ChatBot: Goodbye!")
            break
            
        response = get_response(user_input)
        print("ChatBot:", response)

if __name__ == "__main__":
    main()