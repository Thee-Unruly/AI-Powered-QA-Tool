import os
import requests

# Function to interact with Llama API using a provided API key
def generate_tests_with_llama(requirements, model="llama-3.2", api_key=None):
    """
    Generates test cases using the Llama model.

    Args:
        requirements (str): The requirements or code snippet for generating test cases.
        model (str): The name of the Llama model to use.
        api_key (str): The API key for authenticating with the Llama API.

    Returns:
        str: Generated test scripts or an error message.
    """
    url = "http://localhost:11434/api/completion"  # Ensure the Llama API is running locally on this port

    # API request payload
    payload = {
        "model": model,
        "prompt": f"Generate Python test cases for the following requirement:\n\n{requirements}\n\n"
                  f"Ensure the tests cover boundary conditions, happy paths, and error handling.",
    }

    # Add authorization header
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        result = response.json()
        return result.get("completion", "No test cases generated.")
    except requests.exceptions.RequestException as e:
        return f"Error communicating with Llama: {e}"


# Example usage
if __name__ == "__main__":
    # Input requirements
    requirements = input("Enter the requirement or code snippet for generating test cases:\n")

    # Name of the Llama model
    model_name = "llama-3.2"  # Replace with the actual model name if different.

    # API key
    api_key = "LA-98295d66869d4edf81462b1ddb89ed80a01ac5ff4d67456b83632e0dda9d4577"

    # Generate test cases using Llama
    print("\nGenerating test cases using Llama...")
    test_cases = generate_tests_with_llama(requirements, model=model_name, api_key=api_key)

    # Save the test cases to a file
    output_file = "llama_generated_test_cases.py"
    with open(output_file, "w") as file:
        file.write(test_cases)

    print(f"\nTest cases have been saved to {output_file}.")
