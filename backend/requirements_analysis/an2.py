import re
import spacy
from textblob import TextBlob
from nltk.tokenize import word_tokenize
import nltk
import pandas as pd
import ollama  # Importing Ollama
import streamlit as st

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Load spaCy model for dependency parsing
nlp = spacy.load('en_core_web_sm')


class RequirementAnalyzer:
    def __init__(self, requirements):
        """
        Initialize the RequirementAnalyzer with the given requirements.
        :param requirements: List of strings, where each string is a system requirement.
        """
        self.requirements = requirements
        self.ollama_analyzer = ollama.chat  # Using Ollama for text classification

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

        return detected_terms

    @staticmethod
    def detect_vague_phrases(requirement):
        """
        Detect vague phrases in the requirement using predefined rules.
        :param requirement: A single system requirement string.
        :return: Boolean indicating whether vague phrases are detected.
        """
        vague_phrases = ['sufficiently', 'adequate', 'as required', 'as needed']
        doc = nlp(requirement)
        for token in doc:
            if token.text.lower() in vague_phrases:
                return True
        return False

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
            words = nltk.pos_tag(word_tokenize(str(sentence)))
            for i in range(1, len(words)):
                if words[i - 1][1] in ['VBN'] and words[i][1] in ['IN', 'TO', 'BY']:
                    return True
        return False

    def analyze(self):
        """
        Analyze all requirements for ambiguities, inconsistencies, passive voice usage, and vague phrases.
        :return: DataFrame with analysis results for each requirement.
        """
        analysis_results = []

        for idx, requirement in enumerate(self.requirements):
            ambiguous_terms = self.detect_ambiguity(requirement)
            vague_phrases = self.detect_vague_phrases(requirement)
            inconsistent = self.detect_inconsistencies(requirement)
            passive_voice = self.detect_passive_voice(requirement)

            # Using Ollama for transformer analysis (assuming chat function or similar works)
            ollama_response = self.ollama_analyzer(f"Analyze this requirement: {requirement}")
            transformer_analysis = ollama_response.get('response', 'No analysis')

            analysis_results.append({
                'Requirement ID': idx + 1,
                'Requirement': requirement,
                'Ambiguous Terms': ', '.join(ambiguous_terms),
                'Vague Phrases': vague_phrases,
                'Inconsistencies': inconsistent,
                'Passive Voice': passive_voice,
                'Transformer Analysis': transformer_analysis
            })

        return pd.DataFrame(analysis_results)


# Function to generate Requirements Traceability Matrix (RTM)
def generate_rtm(requirements_df, test_cases, designs, implementations):
    rtm = requirements_df.copy()
    rtm['Test Cases'] = test_cases
    rtm['Designs'] = designs
    rtm['Implementations'] = implementations
    return rtm


# Streamlit Interface
def requirement_analyzer_streamlit():
    st.title("Enhanced Requirement Analyzer")

    uploaded_file = st.file_uploader("Upload Requirements Document (Text File)", type="txt")

    if uploaded_file:
        requirements = uploaded_file.read().decode("utf-8").splitlines()
        analyzer = RequirementAnalyzer(requirements)
        results = analyzer.analyze()

        st.subheader("Analysis Results")
        st.dataframe(results)

        st.subheader("Download Results")
        csv = results.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", data=csv, file_name="requirement_analysis_results.csv")

        # Example RTM data
        test_cases = [f"Test_{i}" for i in range(1, len(requirements) + 1)]
        designs = [f"Design_{i}" for i in range(1, len(requirements) + 1)]
        implementations = [f"Implementation_{i}" for i in range(1, len(requirements) + 1)]

        rtm = generate_rtm(results, test_cases, designs, implementations)
        st.subheader("Requirements Traceability Matrix (RTM)")
        st.dataframe(rtm)


# Run Streamlit app
if __name__ == "__main__":
    requirement_analyzer_streamlit()
