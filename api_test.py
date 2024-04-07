import openai

# Set your OpenAI API key here
api_key = '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'


# Function to call OpenAI API and get response
def call_openai_api(input_text):
    openai.api_key = api_key
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # Use GPT-3.5 "turbo" model
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": input_text}
        ]
    )
    return response['choices'][0]['message']['content']


# Main function
def main():
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
        response = call_openai_api(user_input)
        print("AI: ", response)


if __name__ == "__main__":
    main()
