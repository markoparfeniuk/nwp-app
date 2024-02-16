#!/bin/bash
echo "Installing Python dependencies..."
pip install -r requirements.txt

echo "Installing spaCy's language model..."
python -m spacy download en_core_web_sm