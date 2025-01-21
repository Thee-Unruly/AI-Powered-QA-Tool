import re
from nltk.corpus import wordnet as wn
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag
from textblob import TextBlob
from collections import defaultdict
import pandas as pd

# Load essential NLP models (if using NLTK, download required resources)
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

class RequirementAnalyzer:
    def __init__(self, requirements):
        """
        Initialize the RequirementAnalyzer with the given requirements.
        :param requirements: List of strings, where each string is a system requirement.
        """
        self.requirements = requirements

    @staticmethod
    def detect_ambiguity(requirement):
        """
        Detect ambiguous terms in the requirement using WordNet and predefined ambiguous terms.
        :param requirement: A single system requirement string.
        :return: List of ambiguous terms found in the requirement.
        """
        ambiguous_terms = ['may', 'should', 'can', 'could', 'might', 'possibly', 'typically']
        detected_terms = []

        words = word_tokenize(requirement)
        for word in words:
            if word.lower() in ambiguous_terms:
                detected_terms.append(word)
            elif len(wn.synsets(word)) > 1:  # WordNet detects multiple meanings
                detected_terms.append(word)

        return detected_terms

    @staticmethod
    def detect_inconsistencies(requirement):
        """
        Detect inconsistencies in the requirement by analyzing contradictions.
        :param requirement: A single system requirement string.
        :return: Boolean indicating whether an inconsistency is detected.
        """
        contradictory_pairs = [('must', 'should'), ('always', 'sometimes'), ('mandatory', 'optional')]
        tokens = word_tokenize(requirement.lower())
        for pair in contradictory_pairs:
            if pair[0] in tokens and pair[1] in tokens:
                return True
        return False

    @staticmethod
    def detect_passive_voice(requirement):
        """
        Detect if the requirement is written in passive voice.
        :param requirement: A single system requirement string.
        :return: Boolean indicating whether passive voice is detected.
        """
        blob = TextBlob(requirement)
        for sentence in blob.sentences:
            words = pos_tag(word_tokenize(str(sentence)))
            for i in range(1, len(words)):
                if words[i - 1][1] in ['VBN'] and words[i][1] in ['IN', 'TO', 'BY']:
                    return True
        return False

    def analyze(self):
        """
        Analyze all requirements for ambiguities, inconsistencies, and passive voice usage.
        :return: DataFrame with analysis results for each requirement.
        """
        analysis_results = []

        for idx, requirement in enumerate(self.requirements):
            ambiguous_terms = self.detect_ambiguity(requirement)
            inconsistent = self.detect_inconsistencies(requirement)
            passive_voice = self.detect_passive_voice(requirement)

            analysis_results.append({
                'Requirement ID': idx + 1,
                'Requirement': requirement,
                'Ambiguous Terms': ', '.join(ambiguous_terms),
                'Inconsistencies': inconsistent,
                'Passive Voice': passive_voice
            })

        return pd.DataFrame(analysis_results)

# Example Usage
if __name__ == "__main__":
    # Example requirements list
    requirements_list = [
        "The system should handle up to 1000 requests per second.",
        "Users must log in before accessing the system, but they can sometimes skip this step.",
        "The database might support multiple types of queries.",
        "All data backups are to be performed daily."
    ]

    # Initialize the analyzer and analyze requirements
    analyzer = RequirementAnalyzer(requirements_list)
    results = analyzer.analyze()

    # Save results to a CSV file
    results.to_csv("requirement_analysis_results.csv", index=False)

    # Display results
    print(results)
