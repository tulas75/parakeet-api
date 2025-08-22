#!/usr/bin/env python3
"""
API Test Examples for Language Detection

This script demonstrates the expected behavior of the enhanced Parakeet API
with automatic language detection and OpenAI-compatible responses.
"""

import json

def test_cases():
    """
    Test cases demonstrating the API behavior
    """
    print("Parakeet API Language Detection Test Cases")
    print("=" * 50)
    
    # Test Case 1: Explicit supported language
    print("\n1. Explicit supported language (English):")
    print("   Request: language='en'")
    print("   Expected: Success, uses 'en' in response")
    print("   Response: {'text': '...', 'language': 'en'}")
    
    # Test Case 2: Explicit unsupported language
    print("\n2. Explicit unsupported language (Chinese):")
    print("   Request: language='zh'")
    print("   Expected: Error response")
    expected_error = {
        "error": {
            "message": "Unsupported language: zh",
            "type": "invalid_request_error",
            "param": "language",
            "code": "unsupported_language"
        }
    }
    print(f"   Response: {json.dumps(expected_error, indent=2)}")
    
    # Test Case 3: No language provided, auto-detect supported
    print("\n3. No language provided, auto-detect supported language:")
    print("   Request: language=None (audio contains English speech)")
    print("   Expected: Auto-detect English, proceed with transcription")
    print("   Response: {'text': '...', 'language': 'en'}")
    
    # Test Case 4: No language provided, auto-detect unsupported
    print("\n4. No language provided, auto-detect unsupported language:")
    print("   Request: language=None (audio contains Chinese speech)")
    print("   Expected: Depends on ENABLE_AUTO_LANGUAGE_REJECTION setting")
    print("   - If True: Error response (same as case 2)")
    print("   - If False: Default to English, proceed with transcription")
    
    # Test Case 5: Verbose JSON response
    print("\n5. Verbose JSON response with detected language:")
    print("   Request: language=None, response_format='verbose_json'")
    print("   Expected: Detailed response with detected language")
    expected_verbose = {
        "task": "transcribe",
        "language": "es",  # Auto-detected Spanish
        "duration": 30.5,
        "text": "Hola, esta es una transcripción de ejemplo.",
        "segments": [
            {
                "id": 0,
                "start": 0.0,
                "end": 3.5,
                "text": "Hola, esta es una transcripción de ejemplo.",
                # ... more fields
            }
        ]
    }
    print(f"   Response: {json.dumps(expected_verbose, indent=2)}")

def supported_languages():
    """
    Display the list of supported languages
    """
    print("\n\nSupported Languages (25 total):")
    print("-" * 30)
    
    languages = {
        'bg': 'Bulgarian',
        'hr': 'Croatian', 
        'cs': 'Czech',
        'da': 'Danish',
        'nl': 'Dutch',
        'en': 'English',
        'et': 'Estonian',
        'fi': 'Finnish',
        'fr': 'French',
        'de': 'German',
        'el': 'Greek',
        'hu': 'Hungarian',
        'it': 'Italian',
        'lv': 'Latvian',
        'lt': 'Lithuanian',
        'mt': 'Maltese',
        'pl': 'Polish',
        'pt': 'Portuguese',
        'ro': 'Romanian',
        'sk': 'Slovak',
        'sl': 'Slovenian',
        'es': 'Spanish',
        'sv': 'Swedish',
        'ru': 'Russian',
        'uk': 'Ukrainian'
    }
    
    for code, name in sorted(languages.items()):
        print(f"   {code}: {name}")

if __name__ == "__main__":
    test_cases()
    supported_languages()
    
    print("\n\nConfiguration Options:")
    print("-" * 30)
    print("ENABLE_AUTO_LANGUAGE_REJECTION: Controls whether to reject unsupported languages")
    print("LID_CLIP_SECONDS: Duration of audio clip used for language detection (default: 45)")
    print("\nFor more details, see the API documentation and source code.")