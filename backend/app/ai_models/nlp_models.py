from transformers import pipeline, BertTokenizer, BertForMaskedLM
import torch

# Load pre-trained BERT model and tokenizer for Masked Language Model task
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForMaskedLM.from_pretrained("bert-base-uncased")
nlp_fill_mask = pipeline("fill-mask", model=model, tokenizer=tokenizer)

def analyze_requirements(requirements_text):
    """
    Analyzes the system requirements text using NLP techniques to identify
    ambiguities, inconsistencies, and missing details.
    
    Parameters:
        requirements_text (str): Text of the system requirements document.
        
    Returns:
        dict: Contains identified issues like ambiguity, inconsistencies, and other suggestions.
    """
    issues = {'ambiguity': [], 'inconsistency': [], 'missing_details': []}
    
    # 1. Ambiguity detection: Check for vague terms
    ambiguous_terms = ['should', 'could', 'may', 'might', 'possibly', 'allow', 'may be']
    for term in ambiguous_terms:
        if term in requirements_text.lower():
            issues['ambiguity'].append(f"Potential ambiguity detected: '{term}'")

    # 2. Inconsistency detection: Look for contradictions
    # Example simple inconsistency detection for conflicting requirements
    inconsistent_statements = [
        ("should allow", "should not allow"),
        ("must be", "must not be")
    ]
    for inconsistent_pair in inconsistent_statements:
        if inconsistent_pair[0] in requirements_text.lower() and inconsistent_pair[1] in requirements_text.lower():
            issues['inconsistency'].append(f"Possible inconsistency between '{inconsistent_pair[0]}' and '{inconsistent_pair[1]}'")

    # 3. Missing details detection: Try to find common missing requirements
    missing_phrases = ["with no error", "in any condition", "within 24 hours", "without failure"]
    for phrase in missing_phrases:
        if phrase not in requirements_text.lower():
            issues['missing_details'].append(f"Missing detail suggestion: '{phrase}'")

    # 4. Use Masked Language Model to detect missing words (suggestions)
    masked_text = requirements_text + " The system should [MASK] the data."
    predictions = nlp_fill_mask(masked_text)
    suggestions = [pred['sequence'] for pred in predictions]

    # Adding the suggested completions as part of missing details or improvement
    if suggestions:
        issues['missing_details'].append(f"Suggested improvement: {suggestions[0]}")

    return issues

# Example usage
requirements = """
The system should allow users to create an account.
The system must be able to process data in real-time.
The system could allow third-party integrations.
The system should not allow invalid login attempts.
"""
analysis_result = analyze_requirements(requirements)
print("Analysis Report:")
for issue_type, issues_list in analysis_result.items():
    if issues_list:
        print(f"{issue_type.capitalize()}:")
        for issue in issues_list:
            print(f"  - {issue}")
