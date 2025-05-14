#!/usr/bin/env python3
import os
import sys
import json
import requests
from pathlib import Path
import glob
from dotenv import load_dotenv

env_path = os.path.join(os.getcwd(), '.env')
load_dotenv(dotenv_path=env_path, override=True)

API_URL = os.getenv('OPENWEBUI_API_URL', 'http://localhost:3000')
API_KEY = os.getenv('OPENWEBUI_API_KEY')
REQUIRED_METADATA = ['title', 'author', 'version']

class FunctionNotFoundError(Exception):
    pass

def extract_metadata(content):
    lines = content.split('\n')
    metadata = {}
    if lines[0].strip() == '"""':
        for line in lines[1:]:
            if line.strip() == '"""':
                break
            key, value = line.split(':', 1)
            metadata[key.strip()] = value.strip()
    return metadata

def validate_metadata(metadata):
    missing = [attr for attr in REQUIRED_METADATA if attr not in metadata]
    if missing:
        raise ValueError(f"Missing required metadata: {', '.join(missing)}")

def get_existing_function(function_id, token):
    url = f"{API_URL}/api/v1/functions/id/{function_id}"
    headers = {'Authorization': f'Bearer {token}'}
    response = requests.get(url, headers=headers)
    if response.status_code in (401, 404):
        raise FunctionNotFoundError(f"Function with ID '{function_id}' not found. Will create new function. API Error code: {response.status_code}")
    
    response.raise_for_status()
    res_json = response.json()
    
    return res_json

def deploy_function(filename, token):
    with open(filename, 'r') as file:
        content = file.read()
    
    metadata = extract_metadata(content)
    validate_metadata(metadata)
    function_id = Path(filename).stem

    manifest = {
        "title": metadata.get('title', ''),
        "author": metadata.get('author', ''),
        "author_url": metadata.get('author_url', ''),
        "funding_url": metadata.get('funding_url', ''),
        "version": metadata.get('version', ''),
        "description": metadata.get('description', metadata.get('title', ''))
    }

    headers = {
        'Accept': 'application/json',
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {token}'
    }

    try:
        existing_function = get_existing_function(function_id, token)
        # Update existing function
        url = f"{API_URL}/api/v1/functions/id/{function_id}/update"
        data = {
            "id": function_id,
            "name": existing_function.get('name', metadata.get('title', function_id)),
            "meta": {
                "description": existing_function.get('meta', {}).get('description', metadata.get('description', metadata.get('title', ''))),
                "manifest": manifest
            },
            "content": content
        }
        action = "Updated"
    except FunctionNotFoundError:
        # Create new function
        url = f"{API_URL}/api/v1/functions/create"
        data = {
            "id": function_id,
            "name": metadata.get('title', function_id),
            "meta": {
                "description": metadata.get('description', metadata.get('title', '')),
                "manifest": manifest
            },
            "content": content
        }
        action = "Created"

    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    print(f"Function \"{function_id}\" {action} successfully.")
    return response.json()

def deploy_functions(file_pattern, token):
        
    if file_pattern.endswith('**'):
        base_pattern = file_pattern[:-2]
        files = glob.glob(f"{base_pattern}**/*.py", recursive=True)  # Root directory files
    else:
        files = glob.glob(file_pattern)  # Simple pattern matching

    results = []
    for filename in (f for f in files if os.path.isfile(f)):
        try:
            result = deploy_function(filename, token)
            results.append((filename, result))
        except ValueError as e:
            print(f"Metadata validation error in {filename}: {e}")
        except requests.exceptions.RequestException as e:
            print(f"Error deploying function {filename}: {e}")
        except Exception as e:
            print(f"Error deploying function {filename}: {e}")
    
    return results

if __name__ == "__main__":
    script_name = os.path.basename(sys.argv[0])
    if len(sys.argv) < 2:
        print(f"Usage: python {script_name} <file_pattern> [<file_pattern> ...]")
        sys.exit(1)

    print(f"Deploying to {API_URL}")
    file_patterns = sys.argv[1:]
    
    try:
        token = API_KEY
        total_deployed = 0
        for pattern in file_patterns:
            results = deploy_functions(pattern, token)
            total_deployed += len(results)
        print(f"Deployment completed. {total_deployed} functions deployed successfully.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)