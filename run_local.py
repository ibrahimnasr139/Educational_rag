#!/usr/bin/env python3
"""
Run script for local development (No Docker)
Usage: python run_local.py
"""

import os
import sys
import uvicorn
from pathlib import Path

# Colors for terminal output
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def check_environment():
    """Check if .env file exists and has required variables."""
    env_file = Path('.env')
    
    if not env_file.exists():
        print(f"{Colors.RED}✗ .env file not found!{Colors.ENDC}")
        print(f"\nPlease create .env file:")
        print(f"  cp .env.example .env")
        print(f"  nano .env  # Add your GOOGLE_API_KEY")
        return False
    
    # Check for API key
    with open('.env', 'r') as f:
        content = f.read()
        if 'GOOGLE_API_KEY=your_google_api_key_here' in content or \
           'GOOGLE_API_KEY=' not in content:
            print(f"{Colors.YELLOW}⚠ Warning: GOOGLE_API_KEY not set in .env{Colors.ENDC}")
            print(f"  Please edit .env and add your Google API key")
            return False
    
    return True

def check_dependencies():
    """Check if required packages are installed."""
    try:
        import fastapi
        import uvicorn
        import chromadb
        import whisper
        print(f"{Colors.GREEN}✓ Dependencies installed{Colors.ENDC}")
        return True
    except ImportError as e:
        print(f"{Colors.RED}✗ Missing dependency: {e.name}{Colors.ENDC}")
        print(f"\nPlease run setup first:")
        print(f"  bash local_setup.sh")
        return False

def check_dhakira():
    """Check if Dhakira is available."""
    try:
        repo_root = Path(__file__).resolve().parent
        dhakira_src = repo_root / 'Dhakira'
        if dhakira_src.exists() and str(dhakira_src) not in sys.path:
            sys.path.insert(0, str(dhakira_src))
        from dhakira.embeddings.huggingface_ import HuggingFaceEmbeddings
        print(f"{Colors.GREEN}✓ Dhakira available{Colors.ENDC}")
        return True
    except ImportError:
        print(f"{Colors.YELLOW}⚠ Dhakira not found, will use fallback model{Colors.ENDC}")
        return True  # Not critical, we have fallback

def create_directories():
    """Ensure all required directories exist."""
    dirs = ['data/chroma_db', 'data/uploads', 'data/temp', 'logs']
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    print(f"{Colors.GREEN}✓ Directories ready{Colors.ENDC}")

def print_banner():
    """Print startup banner."""
    print("\n" + "="*60)
    print(f"{Colors.BOLD}{Colors.BLUE}RAG Backend - Local Server{Colors.ENDC}")
    print("="*60 + "\n")

def main():
    """Main entry point."""
    print_banner()
    
    print("Checking environment...")
    
    # Run checks
    if not check_environment():
        sys.exit(1)
    
    if not check_dependencies():
        sys.exit(1)
    
    check_dhakira()
    create_directories()
    
    print(f"\n{Colors.GREEN}✓ All checks passed{Colors.ENDC}\n")
    
    # Configuration
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', 8000))
    
    print("="*60)
    print(f"{Colors.BOLD}Starting server...{Colors.ENDC}")
    print("="*60)
    print(f"\n{Colors.GREEN}Server URL:{Colors.ENDC} http://localhost:{port}")
    print(f"{Colors.GREEN}API Docs:{Colors.ENDC}  http://localhost:{port}/docs")
    print(f"{Colors.GREEN}Health Check:{Colors.ENDC} http://localhost:{port}/health")
    print(f"\n{Colors.YELLOW}Press CTRL+C to stop{Colors.ENDC}\n")
    print("="*60 + "\n")
    
    # Start server
    try:
        uvicorn.run(
            "main:app",
            host=host,
            port=port,
            reload=True,  # Auto-reload on code changes
            log_level="info"
        )
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}Server stopped{Colors.ENDC}")
    except Exception as e:
        print(f"\n{Colors.RED}Error: {e}{Colors.ENDC}")
        sys.exit(1)

if __name__ == "__main__":
    main()
