import nltk
import os
import sys
from pathlib import Path

def setup_nltk_data():
    """Set up NLTK data and ensure proper environment."""
    # Print Python environment info
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version}")
    
    # Create NLTK data directory in the virtual environment
    venv_path = os.path.dirname(os.path.dirname(sys.executable))
    nltk_data_path = os.path.join(venv_path, 'nltk_data')
    os.makedirs(nltk_data_path, exist_ok=True)
    
    # Add the NLTK data path
    nltk.data.path.append(nltk_data_path)
    print(f"NLTK data path: {nltk.data.path}")
    
    # Download required NLTK data
    required_packages = [
        'punkt',
        'averaged_perceptron_tagger',
        'averaged_perceptron_tagger_eng',
        'maxent_ne_chunker',
        'words',
        'stopwords'
    ]
    
    for package in required_packages:
        print(f"Downloading {package}...")
        nltk.download(package, download_dir=nltk_data_path)
    
    # Create punkt_tab directory and files
    punkt_dir = Path(nltk_data_path) / 'tokenizers' / 'punkt_tab' / 'english'
    punkt_dir.mkdir(parents=True, exist_ok=True)
    
    # Create required files
    files_to_create = {
        'collocations.tab': '',
        'sentence_starters.tab': '',
        'abbreviations.tab': '',
        'abbrev_types.txt': '',
        'ortho_context.tab': ''
    }
    
    for filename, content in files_to_create.items():
        file_path = punkt_dir / filename
        if not file_path.exists():
            file_path.write_text(content)
            print(f"Created {filename}")
    
    # Copy punkt data to punkt_tab
    punkt_source = Path(nltk_data_path) / 'tokenizers' / 'punkt' / 'english.pickle'
    if punkt_source.exists():
        import shutil
        shutil.copy(punkt_source, punkt_dir / 'english.pickle')
        print("Copied punkt data to punkt_tab directory")

if __name__ == "__main__":
    setup_nltk_data()
    print("NLTK setup completed!")
