import json
import speech_recognition as sr
from gtts import gTTS
import os
from llama_cpp import Llama

def load_persona(file_path):
    with open(file_path, 'r') as file:
        persona_data = json.load(file)
    return persona_data

def listen_microphone():
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    with microphone as source:
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source)  # adjust for noise
        audio = recognizer.listen(source)

    try:
        print("Recognizing...")
        text = recognizer.recognize_google(audio)
        print("You said:", text)
        return text
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
        return ""
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))
        return ""

def text_to_speech_and_print(text):
    # Print the response text to the shell
    print("Response:", text)
    
    # Convert text to speech and play it
    tts = gTTS(text=text, lang='en')
    tts.save("output.mp3")
    os.system("start output.mp3")  # for Windows, plays the audio using the default player

def generate_response(input_text, llm, persona, history):
    # Concatenate persona data with input text
    input_text_with_persona = f"{persona['char_greeting']} {input_text}"
    context = ". ".join(history[-3:])  # Use last 3 interactions as context
    input_text_with_context = f"{context}. {input_text_with_persona}"
    response = llm(input_text_with_context, max_tokens=100, echo=False)
    # Extract the generated text from the response dictionary
    generated_text = response['choices'][0]['text']  # Access 'text' key within the first choice
    return generated_text

import os

def save_history(history):
    # Load existing history if available
    existing_history = load_history()

    with open("history.txt", "a") as file:
        for item in history:
            # Check if the item is not in existing history before writing
            if item + "\n" not in existing_history:
                file.write(item + "\n")
                # Append the new item to existing history
                existing_history.append(item + "\n")

def load_history():
    # Initialize an empty list to store the history
    history = []

    # Check if the history file exists
    if os.path.exists("history.txt"):
        # Open the history file in read mode
        with open("history.txt", "r") as file:
            # Read all lines from the history file
            history = file.readlines()

    # Return the unique history
    return history

if __name__ == "__main__":
    # Load the persona data
    persona_file_path = "./persona.json"
    persona = load_persona(persona_file_path)

    # Load the Mistral model
    llm = Llama(
        model_path="./mistral-7b-instruct-v0.2.Q8_0.gguf",
        n_ctx=32768,  # Set the maximum context length
        n_threads=8,  # Set the number of CPU threads to use
        n_gpu_layers=0  # Set the number of GPU layers to use (if GPU acceleration is available)
    )

    # Load conversation history
    history = load_history()

    # Enter the main conversation loop
    while True:
        print("Choose your input method:")
        print("1. Type your message")
        print("2. Speak into the microphone")
        choice = input("Enter your choice (1 or 2): ")

        if choice == "1":
            # Text input
            user_input = input("You: ")
            if user_input.lower() == "exit":
                break
            if user_input:
                response = generate_response(user_input, llm, persona, history)
                text_to_speech_and_print(response)
                # Append user input and response to history
                history.append(user_input)
                history.append(response)
                # Save history to file
                save_history(history)
        elif choice == "2":
            # Microphone input
            print("Speak now...")
            user_input = listen_microphone()
            if user_input.lower() == "exit":
                break
            if user_input:
                response = generate_response(user_input, llm, persona, history)
                text_to_speech_and_print(response)
                # Append user input and response to history
                history.append(user_input)
                history.append(response)
                # Save history to file
                save_history(history)
        else:
            print("Invalid choice. Please enter '1' or '2'.")
