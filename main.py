import json
import os
from AI.open_ai import OpenAIModel
from prompt.prompts import SYSTEM_MESSAGES, USER_MESSAGES

def load_json_data(filename: str) -> dict:
    """Load JSON data from file"""
    try:
        with open(filename, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        return {}

def get_summary_from_openai(json_data: dict) -> str:
    """Get summary of JSON data using OpenAI"""
    
    # Convert JSON data to string for the prompt
    json_string = json.dumps(json_data, indent=2)
    system_message = SYSTEM_MESSAGES
    user_message = USER_MESSAGES.format(json_string=json_string)

    try:
        ai_model = OpenAIModel.from_message(
            model_name="gpt-4.1-nano",
            system_message=system_message,
            user_message=user_message,
            temperature=0.3,
            max_tokens=1000
        )
        response = ai_model.execute_text_response()
        # print(response)
        return ai_model.extract_text_response()
        
    except Exception as e:
        return f"Error getting summary from OpenAI: {str(e)}"

def save_summary_to_md(summary: str, filename: str = "summary.md"):
    """Save summary to Markdown file"""
    # Ensure output directory exists
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, filename)
    
    with open(output_path, 'w', encoding='utf-8') as mdfile:
        mdfile.write("## AI Summary\n\n")
        mdfile.write(summary)

def main():
    """Main function to run the JSON summary application"""
    json_filename = "test.json"
    json_data = load_json_data(json_filename)
    
    if not json_data:
        return
    
    summary = get_summary_from_openai(json_data)

    save_summary_to_md(summary)

if __name__ == "__main__":
    main()
