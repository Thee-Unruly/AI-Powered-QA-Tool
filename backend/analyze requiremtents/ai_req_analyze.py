import re
import os
import nltk
import requests
from nltk.corpus import wordnet as wn
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag
from textblob import TextBlob
import pandas as pd
import spacy
import streamlit as st
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Constants
NLTK_DATA_DIR = os.path.join(os.path.expanduser('~'), 'nltk_data')
NLTK_RESOURCES = {
    'punkt': 'tokenizers/punkt',
    'averaged_perceptron_tagger': 'taggers/averaged_perceptron_tagger',
    'wordnet': 'corpora/wordnet',
    'omw-1.4': 'corpora/omw-1.4'
}

def verify_wordnet():
    """Special verification for WordNet as it has multiple files"""
    try:
        required_files = [
            'corpora/wordnet/lexnames',
            'corpora/wordnet/index.sense',
            'corpora/wordnet/data.verb'
        ]
        for file in required_files:
            nltk.data.find(file)
        return True
    except LookupError:
        return False

def setup_nltk():
    """Configure NLTK data path and verify resources"""
    try:
        os.makedirs(NLTK_DATA_DIR, exist_ok=True)
        if NLTK_DATA_DIR not in nltk.data.path:
            nltk.data.path.insert(0, NLTK_DATA_DIR)
        
        missing = []
        for resource, path in NLTK_RESOURCES.items():
            try:
                if resource == 'wordnet':
                    if not verify_wordnet():
                        missing.append(resource)
                else:
                    nltk.data.find(path)
            except LookupError:
                missing.append(resource)
        
        if missing:
            with st.spinner(f"Downloading NLTK resources: {', '.join(missing)}..."):
                for resource in missing:
                    nltk.download(resource, download_dir=NLTK_DATA_DIR)
                    if resource == 'wordnet' and not verify_wordnet():
                        st.warning("WordNet download incomplete, retrying...")
                        nltk.download('wordnet', download_dir=NLTK_DATA_DIR)
                    
        return True
    except Exception as e:
        st.error(f"NLTK setup failed: {e}")
        return False

# Initialize NLTK at startup
if not setup_nltk():
    st.error("""
    Failed to initialize NLTK. Please try manually:
    ```python
    import nltk
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    ```
    Then restart the application.
    """)
    st.stop()

# Load spaCy model
try:
    nlp = spacy.load('en_core_web_sm')
except Exception as e:
    st.error(f"Failed to load spaCy model: {e}")
    st.error("Please install it first: `python -m spacy download en_core_web_sm`")
    st.stop()

# Ollama integration for transformer-based analysis
def ollama_analyze_requirement(req, model='mistral'):
    try:
        prompt = f"""Analyze the following software requirement for clarity, ambiguity, vagueness, and passive voice:

        "{req}"

        Provide a brief assessment of quality issues and suggestions for improvement.
        """
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=15
        )
        response.raise_for_status()
        return response.json()['response'].strip()
    except Exception as e:
        return f"Ollama error: {e}"

class RequirementAnalyzer:
    def __init__(self, requirements):
        self.requirements = requirements

    def detect_ambiguity(self, requirement):
        ambiguous_terms = ['may', 'should', 'can', 'could', 'might', 'possibly', 'typically']
        detected_terms = []
        
        try:
            if not wn.get_version():
                raise LookupError("WordNet not properly initialized")
                
            words = word_tokenize(requirement)
            for word in words:
                if word.lower() in ambiguous_terms:
                    detected_terms.append(word)
                elif len(wn.synsets(word)) > 1:
                    detected_terms.append(word)
        except Exception as e:
            st.warning(f"Ambiguity detection error: {e}")
            
        return detected_terms

    def detect_vague_phrases(self, requirement):
        vague_phrases = ['sufficiently', 'adequate', 'as required', 'as needed']
        try:
            doc = nlp(requirement)
            return any(token.text.lower() in vague_phrases for token in doc)
        except Exception as e:
            st.warning(f"Vague phrase detection error: {e}")
            return False

    def detect_inconsistencies(self, requirement):
        contradictory_pairs = [('must', 'should'), ('always', 'sometimes'), ('mandatory', 'optional')]
        try:
            tokens = word_tokenize(requirement.lower())
            return any(pair[0] in tokens and pair[1] in tokens for pair in contradictory_pairs)
        except Exception as e:
            st.warning(f"Inconsistency detection error: {e}")
            return False

    def detect_passive_voice(self, requirement):
        try:
            blob = TextBlob(requirement)
            for sentence in blob.sentences:
                words = pos_tag(word_tokenize(str(sentence)))
                for i in range(1, len(words)):
                    if words[i - 1][1] in ['VBN'] and words[i][1] in ['IN', 'TO', 'BY']:
                        return True
            return False
        except Exception as e:
            st.warning(f"Passive voice detection error: {e}")
            return False

    def analyze(self):
        analysis_results = []
        
        for idx, requirement in enumerate(self.requirements):
            if not requirement.strip():
                continue
                
            try:
                analysis_results.append({
                    'Requirement ID': idx + 1,
                    'Requirement': requirement,
                    'Ambiguous Terms': ', '.join(self.detect_ambiguity(requirement)) or 'None',
                    'Vague Phrases': 'Yes' if self.detect_vague_phrases(requirement) else 'No',
                    'Inconsistencies': 'Yes' if self.detect_inconsistencies(requirement) else 'No',
                    'Passive Voice': 'Yes' if self.detect_passive_voice(requirement) else 'No',
                    'Transformer Analysis': ollama_analyze_requirement(requirement)
                })
            except Exception as e:
                st.error(f"Failed to analyze requirement {idx+1}: {e}")
                analysis_results.append({
                    'Requirement ID': idx + 1,
                    'Requirement': requirement,
                    **{k: 'Error' for k in [
                        'Ambiguous Terms', 'Vague Phrases', 
                        'Inconsistencies', 'Passive Voice', 
                        'Transformer Analysis'
                    ]}
                })
                
        return pd.DataFrame(analysis_results)

def main():
    st.title("💬 AI Requirement Analyzer")
    
    with st.expander("ℹ️ How to use"):
        st.markdown("""
        1. Upload a `.txt` file with one requirement per line.
        2. The analyzer will check for:
           - Ambiguity
           - Vague phrases
           - Logical inconsistencies
           - Passive voice
           - **AI-powered insights using Ollama**
        3. Download the full analysis as CSV.
        """)
    
    uploaded_file = st.file_uploader("📄 Upload Requirements File", type="txt")
    
    if uploaded_file:
        try:
            requirements = [
                line.strip() 
                for line in uploaded_file.read().decode("utf-8").splitlines() 
                if line.strip()
            ]
            
            if not requirements:
                st.warning("⚠️ No valid requirements found in the file.")
                return
                
            with st.spinner("🔍 Analyzing requirements..."):
                analyzer = RequirementAnalyzer(requirements)
                results = analyzer.analyze()
                
                st.subheader("✅ Analysis Results")
                st.dataframe(results)
                
                st.download_button(
                    "⬇️ Download CSV",
                    data=results.to_csv(index=False).encode('utf-8'),
                    file_name="requirement_analysis.csv",
                    mime="text/csv"
                )
                
        except Exception as e:
            st.error(f"Processing failed: {e}")

if __name__ == "__main__":
    main()
