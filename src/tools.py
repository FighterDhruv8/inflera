import re
import math
import requests
from typing import Dict, Any

class Calculator:
    def __init__(self):
        """Initialize the calculator tool."""
        self.operators = {
            '+': lambda x, y: x + y,
            '-': lambda x, y: x - y,
            '*': lambda x, y: x * y,
            '/': lambda x, y: x / y,
            '^': lambda x, y: x ** y,
            'sqrt': lambda x: math.sqrt(x),
            'sin': lambda x: math.sin(math.radians(x)),
            'cos': lambda x: math.cos(math.radians(x)),
            'tan': lambda x: math.tan(math.radians(x))
        }
        
    def extract_math_expression(self, query: str) -> str:
        """Extract a mathematical expression from the query.
        
        Args:
            query: User query string
            
        Returns:
            Extracted math expression or empty string if none found
        """
        patterns = [
            r'calculate\s+([\d\s\+\-\*\/\^\(\)\.\,]+)',
            r'what is\s+([\d\s\+\-\*\/\^\(\)\.\,]+)',
            r'result of\s+([\d\s\+\-\*\/\^\(\)\.\,]+)',
            r'compute\s+([\d\s\+\-\*\/\^\(\)\.\,]+)',
            r'solve\s+([\d\s\+\-\*\/\^\(\)\.\,]+)',
            r'([\d\s\+\-\*\/\^\(\)\.\,]+)\s*=\s*\?'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query.lower())
            if match:
                return match.group(1).strip()
                
        return ""
    
    def calculate(self, query: str) -> Dict[str, Any]:
        """Perform calculation based on the query.
        
        Args:
            query: User query string
            
        Returns:
            Dictionary with calculation result and process
        """
        expression = self.extract_math_expression(query)
        
        if not expression:
            return {
                "success": False,
                "message": "No valid mathematical expression found"
            }
        
        try:
            result = eval(expression)
            
            return {
                "success": True,
                "expression": expression,
                "result": result,
                "explanation": f"The result of {expression} is {result}"
            }
        except Exception as e:
            return {
                "success": False,
                "expression": expression,
                "message": f"Error calculating expression: {str(e)}"
            }


class Dictionary:
    def __init__(self):
        """Initialize the dictionary tool."""
        self.api_url = "https://api.dictionaryapi.dev/api/v2/entries/en/"
        
    def extract_word(self, query: str) -> str:
        """Extract a word to define from the query.
        
        Args:
            query: User query string
            
        Returns:
            Word to define or empty string if none found
        """
        patterns = [
            r'define\s+(?:the\s+word\s+)?([a-zA-Z]+)',
            r'what does\s+([a-zA-Z]+)\s+mean',
            r'meaning of\s+([a-zA-Z]+)',
            r'definition of\s+([a-zA-Z]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query.lower())
            if match:
                return match.group(1).strip()
                
        return ""
    
    def define(self, query: str) -> Dict[str, Any]:
        """Look up the definition of a word.
        
        Args:
            query: User query string
            
        Returns:
            Dictionary with word definition information
        """
        word = self.extract_word(query)
        
        if not word:
            return {
                "success": False,
                "message": "No word to define found in the query"
            }
        
        try:
            response = requests.get(f"{self.api_url}{word}")
            
            if response.status_code == 200:
                data = response.json()
                
                if data and isinstance(data, list) and len(data) > 0:
                    result = {
                        "success": True,
                        "word": word,
                        "definitions": []
                    }
                    
                    for entry in data:
                        for meaning in entry.get('meanings', []):
                            for definition in meaning.get('definitions', []):
                                result["definitions"].append({
                                    "part_of_speech": meaning.get('partOfSpeech', ''),
                                    "definition": definition.get('definition', ''),
                                    "example": definition.get('example', '')
                                })
                    
                    return result
                else:
                    return {
                        "success": False,
                        "word": word,
                        "message": "No definitions found"
                    }
            else:
                return {
                    "success": False,
                    "word": word,
                    "message": f"Error: {response.status_code}"
                }
        except Exception as e:
            return {
                "success": False,
                "word": word,
                "message": f"Error looking up definition: {str(e)}"
            }