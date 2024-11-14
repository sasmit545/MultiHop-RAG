import json
import ast
from pathlib import Path
from typing import Dict, List, Optional
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

# Constants
QUERY_FILTERS_FILE = Path("query_fin_metadata_filters.json")
DATASET_FILE = Path("MultiHop-RAG/dataset/finance.json")
MODEL_NAME = "gpt-3.5-turbo"

EXTRACT_FILTER_TEMPLATE = """Some questions will be provided below. 
Given the question, extract metadata that includes the document name ("doc_name") and the evidence page number ("evidence_page_num").
Only include these two fields if present in the data.
-----------------------------------------------------------------------------
Examples:

Question: What was the revenue reported by 3M in its 2018 10-K document?
Answer: {{'doc_name': '3M_2018_10K', 'evidence_page_num': 59}}

Question: Refer to the "Tesla_2020_Annual_Report" on page 34 to find capital expenditures.
Answer: {{'doc_name': 'Tesla_2020_Annual_Report', 'evidence_page_num': 34}}

If no doc_name or evidence_page_num is found, answer with an empty dictionary.
-----------------------------------------------------------------------------
Now it is your turn:

Question: {query}
Answer:
"""

def clean_filter(filter_dict: dict) -> dict:
    """Clean the extracted filter dictionary.
    
    Args:
        filter_dict (dict): Raw filter dictionary from API response
        
    Returns:
        dict: Cleaned filter with only valid keys
    """
    valid_keys = {"doc_name", "evidence_page_num"}
    return {k: filter_dict[k] for k in valid_keys if k in filter_dict}

def load_json_file(filepath: Path) -> List[dict]:
    """Load and parse a JSON file.
    
    Args:
        filepath (Path): Path to JSON file
        
    Returns:
        List[dict]: Parsed JSON content
        
    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file contains invalid JSON
    """
    try:
        if filepath.exists() and filepath.stat().st_size > 0:
            with open(filepath) as f:
                return json.load(f)
        return []
    except json.JSONDecodeError as e:
        print(f"Error loading {filepath}: {e}")
        return []

def save_json_file(data: List[dict], filepath: Path) -> None:
    """Save data to a JSON file.
    
    Args:
        data (List[dict]): Data to save
        filepath (Path): Output file path
    """
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4, sort_keys=True)

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def extract_filter_from_openai(client: OpenAI, query: str) -> Optional[Dict]:
    """Extract filter from query using OpenAI API with retry logic.
    
    Args:
        client (OpenAI): OpenAI client instance
        query (str): Query to process
        
    Returns:
        Optional[Dict]: Extracted filter dictionary or None if extraction failed
    """
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": EXTRACT_FILTER_TEMPLATE.format(query=query),
                }
            ],
            temperature=0.1,
        )
        
        filter_str = completion.choices[0].message.content
        print(f"Received filter string: {filter_str}")
        
        filter_dict = ast.literal_eval(filter_str)
        return clean_filter(filter_dict)
        
    except Exception as e:
        print(f"Error extracting filter for query '{query}': {e}")
        return None

def main():
    """Main function to process queries and extract filters."""
    # Initialize OpenAI client
    client = OpenAI()
    
    # Load existing data
    query_filters = load_json_file(QUERY_FILTERS_FILE)
    query_data_list = load_json_file(DATASET_FILE)
    
    if not query_data_list:
        print("Error: Could not load query dataset")
        return
    
    # Track processed queries
    processed_queries = {item["query"] for item in query_filters}
    updates = 0
    
    # Process each query
    for query_data in query_data_list:
        question = query_data["question"]
        if question not in processed_queries:
            filter_dict = extract_filter_from_openai(client, question)
            
            if filter_dict:
                query_filters.append({
                    "query": question,
                    "filter": filter_dict
                })
                updates += 1
                
                # Save periodically to prevent data loss
                if updates % 10 == 0:
                    save_json_file(query_filters, QUERY_FILTERS_FILE)
                    print(f"Saved {updates} updates")
    
    # Final save
    if updates > 0:
        save_json_file(query_filters, QUERY_FILTERS_FILE)
        print(f"Completed processing with {updates} total updates")

if __name__ == "__main__":
    main()