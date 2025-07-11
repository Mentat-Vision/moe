import re
import json
import datetime
import os
import sys
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from llama_cpp import Llama

# === Config ===
MODEL_PATH = "modelsChat/llama-3.2-1b-instruct-q4_k_m.gguf"
LOG_FILE = "logs.txt"
QUESTION = "tell me everything that happened in camera 1"

class LogRetriever:
    """Efficient log retrieval system for JSON logs"""
    
    def __init__(self, log_file: str):
        self.log_file = log_file
        self.logs_cache = []
        self._load_logs()
    
    def _load_logs(self):
        """Load and parse JSON logs from file"""
        try:
            with open(self.log_file, 'r') as f:
                content = f.read().strip()
                if not content:
                    self.logs_cache = []
                    return
                
                # Split by double newlines to separate JSON objects
                json_blocks = content.split('\n\n')
                self.logs_cache = []
                
                for block in json_blocks:
                    block = block.strip()
                    if block:
                        try:
                            log_entry = json.loads(block)
                            self.logs_cache.append(log_entry)
                        except json.JSONDecodeError:
                            # Skip malformed JSON
                            continue
                            
        except FileNotFoundError:
            self.logs_cache = []
        except Exception as e:
            print(f"Error loading logs: {e}")
            self.logs_cache = []
    
    def _parse_timestamp(self, timestamp_str: str) -> Optional[datetime]:
        """Parse ISO timestamp string to datetime object"""
        try:
            return datetime.fromisoformat(timestamp_str)
        except ValueError:
            return None
    
    def _extract_time_constraints(self, question: str) -> Dict[str, Any]:
        """Extract time constraints from the question"""
        time_constraints = {
            'start_time': None,
            'end_time': None,
            'duration': None
        }
        
        # Time patterns for relative time references
        time_patterns = {
            r'last (\d+) seconds?': lambda x: timedelta(seconds=int(x)),
            r'last (\d+) minutes?': lambda x: timedelta(minutes=int(x)),
            r'last (\d+) hours?': lambda x: timedelta(hours=int(x)),
            r'(\d+) seconds? ago': lambda x: timedelta(seconds=int(x)),
            r'(\d+) minutes? ago': lambda x: timedelta(minutes=int(x)),
            r'in the last (\d+) seconds?': lambda x: timedelta(seconds=int(x)),
            r'in the last (\d+) minutes?': lambda x: timedelta(minutes=int(x)),
        }
        
        question_lower = question.lower()
        for pattern, time_func in time_patterns.items():
            match = re.search(pattern, question_lower)
            if match:
                duration = time_func(match.group(1))
                time_constraints['duration'] = duration
                time_constraints['end_time'] = datetime.now()
                time_constraints['start_time'] = datetime.now() - duration
                break
        
        return time_constraints
    
    def _filter_by_time(self, logs: List[Dict], time_constraints: Dict) -> List[Dict]:
        """Filter logs based on time constraints"""
        if not time_constraints['start_time'] and not time_constraints['end_time']:
            return logs
        
        filtered_logs = []
        for log in logs:
            timestamp = self._parse_timestamp(log.get('timestamp', ''))
            if not timestamp:
                continue
                
            # Apply time filters
            if time_constraints['start_time'] and timestamp < time_constraints['start_time']:
                continue
            if time_constraints['end_time'] and timestamp > time_constraints['end_time']:
                continue
                
            filtered_logs.append(log)
        
        return filtered_logs
    
    def _extract_camera_reference(self, question: str) -> Optional[str]:
        """Extract camera reference from question"""
        camera_patterns = [
            r'CAM(\d+)',
            r'IP(\d+)',
            r'camera (\d+)',
            r'(\w+)\s+webcam',
            r'(\w+)\s+door'
        ]
        
        for pattern in camera_patterns:
            match = re.search(pattern, question, re.IGNORECASE)
            if match:
                # For CAM1, IP101 patterns, construct the full camera name
                if 'CAM' in pattern:
                    return f"CAM{match.group(1)}"
                elif 'IP' in pattern:
                    return f"IP{match.group(1)}"
                else:
                    return match.group(0)
        return None
    
    def _filter_by_camera(self, logs: List[Dict], camera_id: str) -> List[Dict]:
        """Filter logs by specific camera"""
        if not camera_id:
            return logs
        
        # More flexible camera matching
        filtered_logs = []
        for log in logs:
            log_camera = log.get('camera', '').lower()
            search_camera = camera_id.lower()
            
            # Check for exact match or partial match
            if search_camera in log_camera or log_camera in search_camera:
                filtered_logs.append(log)
        
        return filtered_logs
    
    def _semantic_filter(self, logs: List[Dict], question: str) -> List[Dict]:
        """Filter logs based on semantic relevance to the question"""
        question_lower = question.lower()
        relevant_keywords = []
        
        # Extract keywords from question
        if 'man' in question_lower or 'person' in question_lower:
            relevant_keywords.extend(['person', 'man', 'woman', 'people'])
        if 'car' in question_lower or 'vehicle' in question_lower:
            relevant_keywords.extend(['car', 'vehicle', 'truck', 'motorcycle'])
        if 'door' in question_lower:
            relevant_keywords.extend(['door', 'entrance', 'exit'])
        
        if not relevant_keywords:
            return logs
        
        # Filter logs that contain relevant keywords in caption or objects
        filtered_logs = []
        for log in logs:
            caption = log.get('caption', '').lower()
            objects = [obj.lower() for obj in log.get('objects', [])]
            
            # Check if any keyword appears in caption or objects
            for keyword in relevant_keywords:
                if keyword in caption or any(keyword in obj for obj in objects):
                    filtered_logs.append(log)
                    break
        
        return filtered_logs
    
    def retrieve_relevant_logs(self, question: str, max_entries: int = 50) -> List[Dict]:
        """Main retrieval method that combines all filtering strategies"""
        # Extract constraints from question
        time_constraints = self._extract_time_constraints(question)
        camera_id = self._extract_camera_reference(question)
        
        # Debug information
        print(f"DEBUG: Question: '{question}'")
        print(f"DEBUG: Total logs loaded: {len(self.logs_cache)}")
        print(f"DEBUG: Camera ID extracted: '{camera_id}'")
        print(f"DEBUG: Time constraints: {time_constraints}")
        
        # Start with all logs
        relevant_logs = self.logs_cache.copy()
        print(f"DEBUG: Starting with {len(relevant_logs)} logs")
        
        # Apply time filtering
        if time_constraints['start_time'] or time_constraints['end_time']:
            relevant_logs = self._filter_by_time(relevant_logs, time_constraints)
            print(f"DEBUG: After time filtering: {len(relevant_logs)} logs")
        
        # Apply camera filtering
        if camera_id:
            relevant_logs = self._filter_by_camera(relevant_logs, camera_id)
            print(f"DEBUG: After camera filtering: {len(relevant_logs)} logs")
        
        # Apply semantic filtering
        relevant_logs = self._semantic_filter(relevant_logs, question)
        print(f"DEBUG: After semantic filtering: {len(relevant_logs)} logs")
        
        # Sort by timestamp (newest first)
        relevant_logs.sort(key=lambda x: self._parse_timestamp(x.get('timestamp', '')), reverse=True)
        
        # Limit results
        return relevant_logs[:max_entries]
    
    def format_logs_for_prompt(self, logs: List[Dict]) -> str:
        """Format logs for inclusion in prompt"""
        if not logs:
            return "No relevant log entries found."
        
        formatted_logs = []
        for log in logs:
            timestamp = log.get('timestamp', 'Unknown')
            camera = log.get('camera', 'Unknown')
            caption = log.get('caption', 'No caption')
            objects = ', '.join(log.get('objects', [])) if log.get('objects') else 'No objects'
            
            formatted_logs.append(f"Time: {timestamp} | Camera: {camera} | Caption: {caption} | Objects: {objects}")
        
        return '\n'.join(formatted_logs)

def get_relevant_log_entries(question: str, log_file: str) -> str:
    """Updated function to use the new LogRetriever"""
    retriever = LogRetriever(log_file)
    relevant_logs = retriever.retrieve_relevant_logs(question)
    return retriever.format_logs_for_prompt(relevant_logs)

def create_improved_prompt(question: str, context: str) -> str:
    """Create an improved prompt for better analysis"""
    
    return f"""You are an AI surveillance analyst. Analyze the provided surveillance log data and answer the user's question with specific, actionable information.

### SURVEILLANCE LOG DATA:
{context}

### USER QUESTION:
{question}

### ANALYSIS INSTRUCTIONS:
1. **Be Specific**: Mention exact timestamps, cameras, and events
2. **Identify Patterns**: Look for recurring objects, people, or activities
3. **Chronological Order**: Present events in time sequence when relevant
4. **Quantify**: Count objects, people, or events when possible
5. **Highlight Changes**: Focus on what changed between log entries
6. **Camera Context**: Distinguish between different camera locations
7. **Be Concise**: Provide clear, factual information without speculation

### RESPONSE FORMAT:
- Start with a brief summary of what was found
- List specific events with timestamps and cameras
- Mention any patterns or notable changes
- If nothing relevant found, clearly state this

### ANALYSIS:
"""

def main():
    # Completely suppress all output during model loading
    with open(os.devnull, 'w') as devnull:
        # Redirect both stdout and stderr
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        
        try:
            # === Load Llama ===
            llm = Llama(
                model_path=MODEL_PATH,
                n_ctx=8192,
                n_threads=4,
                n_gpu_layers=35,
                use_mlock=True,
                verbose=False,
                logits_all=False,
                embedding=False
            )
        finally:
            # Restore stdout and stderr
            sys.stdout = old_stdout
            sys.stderr = old_stderr
    
    # === Load relevant context ===
    context = get_relevant_log_entries(QUESTION, LOG_FILE)
    
    # Debug: Print what we found
    print(f"Found {len(context.split(chr(10)))} log entries")
    print("Context preview:")
    print(context[:500] + "..." if len(context) > 500 else context)
    print("\n" + "="*50 + "\n")
    
    # === Create improved prompt ===
    prompt = create_improved_prompt(QUESTION, context)
    
    # === Run inference with better parameters ===
    response = llm(
        prompt, 
        max_tokens=512,  # Increased for more detailed responses
        stop=["###", "\n\n\n"],  # Better stop sequences
        temperature=0.1,  # Lower temperature for more focused responses
        top_p=0.9,  # Add top_p for better quality
        repeat_penalty=1.1  # Prevent repetition
    )
    
    # Extract only the response text
    answer = response["choices"][0]["text"].strip()
    
    # Clean up any remaining artifacts
    if answer.startswith("###"):
        answer = answer.split("###", 1)[1].strip()
    
    # Print the final answer
    print("ANALYSIS RESULT:")
    print("="*50)
    print(answer)

if __name__ == "__main__":
    main()
