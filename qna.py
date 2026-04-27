"""
SJSU IT Service Desk - QnA Interface with Multi-Turn Support
=============================================================
Interactive chat with conversation memory and query rewriting.

Features:
- Remembers conversation history
- Rewrites follow-up questions to be standalone
- Handles pronouns like "it", "that", "them"
- Works with vague queries like "tell me more"

Usage:
    python3 qna.py

Prerequisites:
    - Corpus already created (run setup_it_corpus.py first)
    - Update CORPUS_NAME below with your actual corpus resource name
"""

import vertexai
from vertexai import rag
from vertexai.generative_models import GenerativeModel, Tool
from typing import List, Dict, Optional

# ============================================================================
# CONFIGURATION
# ============================================================================
PROJECT_ID = "sjsu-it-genai-poc"
LOCATION = "us-west1"

# Corpus name from setup_it_corpus.py
CORPUS_NAME = "projects/925509787316/locations/us-west1/ragCorpora/4035225266123964416"

# How many conversation turns to keep in memory (for context)
MAX_HISTORY_TURNS = 5

# ============================================================================
# Initialize Vertex AI
# ============================================================================
vertexai.init(project=PROJECT_ID, location=LOCATION)

# ============================================================================
# Conversation Memory
# ============================================================================
class ConversationMemory:
    """Stores conversation history for multi-turn context."""
    
    def __init__(self, max_turns: int = MAX_HISTORY_TURNS):
        self.history: List[Dict[str, str]] = []
        self.max_turns = max_turns
    
    def add_turn(self, user_message: str, assistant_message: str):
        """Add a Q&A turn to history."""
        self.history.append({
            'user': user_message,
            'assistant': assistant_message
        })
        
        # Keep only last N turns
        if len(self.history) > self.max_turns:
            self.history = self.history[-self.max_turns:]
    
    def get_history(self) -> List[Dict[str, str]]:
        """Get current conversation history."""
        return self.history
    
    def format_history_for_prompt(self) -> str:
        """Format history as text for LLM prompts."""
        if not self.history:
            return ""
        
        formatted = []
        for turn in self.history:
            formatted.append(f"User: {turn['user']}")
            formatted.append(f"Assistant: {turn['assistant']}")
        
        return "\n".join(formatted)
    
    def clear(self):
        """Clear conversation history (start fresh)."""
        self.history = []
    
    def is_empty(self) -> bool:
        """Check if there's any history."""
        return len(self.history) == 0

# ============================================================================
# Query Rewriting
# ============================================================================
def rewrite_query_with_context(current_query: str, memory: ConversationMemory) -> str:
    """
    Rewrite a follow-up question to be standalone using conversation history.
    
    Examples:
        Input: "can u tell me more about it?"
        History: [{"user": "what AI platforms?", "assistant": "ChatGPT, Gemini..."}]
        Output: "Tell me more about ChatGPT, Gemini, and Zoom AI Companion at SJSU"
    
    Args:
        current_query: The user's current question
        memory: ConversationMemory object with history
        
    Returns:
        Standalone query that includes necessary context
    """
    # If no history, return query as-is
    if memory.is_empty():
        return current_query
    
    # Format history for the rewrite prompt
    history_text = memory.format_history_for_prompt()
    
    # Create rewrite prompt
    rewrite_prompt = f"""Given this conversation history:

{history_text}

Rewrite this follow-up question as a standalone, self-contained question that includes all necessary context from the conversation history. The rewritten question should make sense even without seeing the conversation history.

Follow-up question: "{current_query}"

Rewritten standalone question (one sentence, no preamble):"""
    
    try:
        # Use a simple model for rewriting (no RAG needed)
        rewriter = GenerativeModel("gemini-2.5-flash")
        response = rewriter.generate_content(rewrite_prompt)
        
        standalone_query = response.text.strip()
        
        # Remove quotes if the model added them
        if standalone_query.startswith('"') and standalone_query.endswith('"'):
            standalone_query = standalone_query[1:-1]
        
        return standalone_query
        
    except Exception as e:
        print(f"⚠ Query rewriting failed: {e}")
        print(f"  Using original query instead.\n")
        return current_query

# ============================================================================
# Create RAG-enabled Gemini model
# ============================================================================
def create_rag_model(corpus_name, model_name="gemini-2.5-flash"):
    """Create a Gemini model with RAG retrieval from the specified corpus."""
    rag_tool = Tool.from_retrieval(
        retrieval=rag.Retrieval(
            source=rag.VertexRagStore(
                rag_resources=[rag.RagResource(rag_corpus=corpus_name)],
                rag_retrieval_config=rag.RagRetrievalConfig(
                    top_k=5,  # Retrieve top 5 most relevant chunks
                    filter=rag.Filter(vector_distance_threshold=0.5)
                ),
            )
        )
    )
    
    model = GenerativeModel(
        model_name=model_name,
        tools=[rag_tool],
        system_instruction="""You are the SJSU IT Service Desk assistant. 
Answer questions based only on the provided context from IT documentation.
Be concise and helpful. If the answer isn't in the context, say 
"I don't have that information in my knowledge base. Please contact 
the IT Service Desk at (408) 924-1530 or visit sjsu.edu/it for help."
"""
    )
    
    return model

# ============================================================================
# Ask a question with conversation context
# ============================================================================
def ask_question_with_context(model, question: str, memory: ConversationMemory, 
                              show_rewrite: bool = True) -> Optional[str]:
    """
    Send a question to the RAG model with conversation context.
    
    Args:
        model: The RAG-enabled Gemini model
        question: User's question
        memory: ConversationMemory object
        show_rewrite: Whether to display the rewritten query
        
    Returns:
        Assistant's answer, or None if error
    """
    print(f"\n{'='*70}")
    print(f"Your question: {question}")
    
    # Step 1: Rewrite query if there's conversation history
    standalone_query = rewrite_query_with_context(question, memory)
    
    if show_rewrite and standalone_query != question:
        print(f"{'='*70}")
        print(f"🔍 Contextualized: {standalone_query}")
    
    print(f"{'='*70}")
    
    try:
        # Step 2: Generate answer using rewritten query
        # The RAG tool will retrieve based on standalone_query
        response = model.generate_content(standalone_query)
        answer = response.text
        
        print(f"\nA: {answer}\n")
        
        # Step 3: Add to conversation memory
        memory.add_turn(question, answer)
        
        return answer
        
    except Exception as e:
        print(f"\n✗ Error: {e}\n")
        return None

# ============================================================================
# Interactive QnA loop with conversation memory
# ============================================================================
def interactive_mode(model):
    """Run an interactive question-answering session with conversation memory."""
    memory = ConversationMemory()
    
    print("\n" + "="*70)
    print("SJSU IT Service Desk - Interactive Chat (Multi-Turn Support)")
    print("="*70)
    print("Type your questions below.")
    print("Commands:")
    print("  'quit' or 'exit' - Exit the chat")
    print("  'new' or 'reset' - Start a new conversation")
    print("  'history' - Show conversation history")
    print("="*70 + "\n")
    
    while True:
        try:
            question = input("Your question: ").strip()
            
            # Handle commands
            if question.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break
            
            if question.lower() in ['new', 'reset', 'clear']:
                memory.clear()
                print("\n✓ Started new conversation. History cleared.\n")
                continue
            
            if question.lower() == 'history':
                if memory.is_empty():
                    print("\nNo conversation history yet.\n")
                else:
                    print("\n" + "="*70)
                    print("CONVERSATION HISTORY")
                    print("="*70)
                    print(memory.format_history_for_prompt())
                    print("="*70 + "\n")
                continue
            
            if not question:
                continue
            
            # Ask question with context
            ask_question_with_context(model, question, memory)
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\n✗ Error: {e}\n")

# ============================================================================
# Demo: Multi-turn conversation flow
# ============================================================================
def demo_multi_turn(model):
    """Demonstrate multi-turn conversation with follow-up questions."""
    memory = ConversationMemory()
    
    print("\n" + "="*70)
    print("DEMO: Multi-Turn Conversation")
    print("="*70)
    print("This demo shows how the system handles follow-up questions.\n")
    
    # Conversation scenario
    questions = [
        "What AI platforms do we have?",
        "Can you tell me more about it?",  # "it" refers to AI platforms
        "What about the other tools?",     # Continues the topic
        "How do I access them?",           # "them" refers to AI platforms
    ]
    
    for idx, question in enumerate(questions, 1):
        print(f"\n[Turn {idx}]")
        input("Press Enter to continue...")
        ask_question_with_context(model, question, memory, show_rewrite=True)

# ============================================================================
# Predefined test questions (single-turn, no context)
# ============================================================================
def run_test_questions(model):
    """Run a set of predefined test questions (no conversation context)."""
    test_questions = [
        "How do I reset my SJSU One password?",
        "How do I connect to SJSU WiFi?",
        "What VPN should I use for remote access?",
        "How do I get help from IT Service Desk?",
        "What are the IT Service Desk hours?",
    ]
    
    print("\n" + "="*70)
    print("Running test questions (single-turn, no context)...")
    print("="*70)
    
    memory = ConversationMemory()  # Fresh memory for each question
    
    for question in test_questions:
        ask_question_with_context(model, question, memory, show_rewrite=False)
        input("Press Enter for next question...")

# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    # Validate corpus name is set
    if CORPUS_NAME == "YOUR_CORPUS_NAME_HERE":
        print("\n" + "="*70)
        print("⚠ ERROR: CORPUS_NAME not set")
        print("="*70)
        print("\nPlease update the CORPUS_NAME variable in this script.")
        print("\nSteps:")
        print("1. Run setup_it_corpus.py if you haven't already")
        print("2. Copy the corpus name from the output")
        print("3. Paste it into the CORPUS_NAME variable at the top of this file")
        print("\nExample:")
        print('  CORPUS_NAME = "projects/sjsu-it-genai-poc/locations/us-west1/ragCorpora/123456789"')
        print("="*70 + "\n")
        exit(1)
    
    # Create the RAG model
    print(f"\nInitializing RAG model...")
    print(f"Project: {PROJECT_ID}")
    print(f"Region: {LOCATION}")
    print(f"Corpus: {CORPUS_NAME}")
    print(f"Model: gemini-2.5-flash")
    print(f"Multi-turn: Enabled ✓\n")
    
    model = create_rag_model(CORPUS_NAME)
    
    # Ask user what mode they want
    print("Choose mode:")
    print("  1. Interactive chat (with conversation memory)")
    print("  2. Demo multi-turn conversation")
    print("  3. Run predefined test questions (single-turn)")
    print("  4. Ask a single question and exit\n")
    
    choice = input("Enter choice (1/2/3/4): ").strip()
    
    if choice == "1":
        interactive_mode(model)
    elif choice == "2":
        demo_multi_turn(model)
    elif choice == "3":
        run_test_questions(model)
    elif choice == "4":
        question = input("\nYour question: ").strip()
        if question:
            memory = ConversationMemory()
            ask_question_with_context(model, question, memory, show_rewrite=False)
    else:
        print("Invalid choice. Running interactive chat...\n")
        interactive_mode(model)