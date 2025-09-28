#!/usr/bin/env python3
"""
Main entry point for Magic Terminal
"""

import sys
import argparse
import os
from pathlib import Path

# Add the package to the path
sys.path.insert(0, str(Path(__file__).parent))

from core import EnhancedAITerminal


def create_parser():
    """Create command line argument parser"""
    parser = argparse.ArgumentParser(
        prog="magic-terminal",
        description="AI-powered terminal assistant with natural language processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  magic-terminal                    # Start interactive mode
  magic-terminal --version          # Show version
  magic-terminal --config           # Show configuration
  magic-terminal --setup            # Run initial setup

For more information, visit: https://github.com/yourusername/magic-terminal
        """
    )
    
    parser.add_argument(
        "--version", 
        action="version", 
        version="Magic Terminal v1.0.0"
    )
    
    parser.add_argument(
        "--config",
        action="store_true",
        help="Show current configuration"
    )
    
    parser.add_argument(
        "--setup",
        action="store_true",
        help="Run initial setup wizard"
    )
    
    parser.add_argument(
        "--ollama-url",
        default="http://localhost:11434",
        help="Ollama server URL (default: http://localhost:11434)"
    )
    
    parser.add_argument(
        "--enable-fallback",
        action="store_true",
        help="Enable fallback mode when LLM is unavailable"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    return parser


def show_config():
    """Show current configuration"""
    config_file = Path.home() / ".magic_terminal_config.json"
    log_file = Path.home() / ".magic_terminal_logs" / "enhanced_terminal.log"
    history_file = Path.home() / ".magic_terminal_history"
    
    print("Magic Terminal Configuration")
    print("=" * 40)
    print(f"Config file: {config_file}")
    print(f"Log file: {log_file}")
    print(f"History file: {history_file}")
    print(f"Ollama URL: {os.getenv('OLLAMA_URL', 'http://localhost:11434')}")
    print(f"Grok API Key: {'Set' if os.getenv('XAI_API_KEY') or os.getenv('GROK_API_KEY') else 'Not set'}")
    print(f"Fallback enabled: {os.getenv('AI_TERMINAL_ALLOW_FALLBACK', '0') == '1'}")


def run_setup():
    """Run initial setup wizard"""
    print("Magic Terminal Setup Wizard")
    print("=" * 40)
    
    # Check Ollama
    print("\n1. Checking Ollama installation...")
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=3)
        if response.status_code == 200:
            models = response.json().get('models', [])
            print(f"Ollama is running with {len(models)} models")
            for model in models[:3]:  # Show first 3 models
                print(f"   - {model['name']}")
        else:
            print("Ollama is running but returned an error")
    except Exception:
        print("Ollama is not running or not installed")
        print("   Install Ollama from: https://ollama.ai")
    
    # Check API keys
    print("\n2. Checking API keys...")
    api_keys_found = []
    
    if os.getenv('OPENAI_API_KEY'):
        api_keys_found.append("OpenAI")
        print("OpenAI API key is configured")
    
    if os.getenv('XAI_API_KEY') or os.getenv('GROK_API_KEY'):
        api_keys_found.append("Grok")
        print("Grok API key is configured")
    
    if not api_keys_found:
        print("No API keys found")
        print("   Available options:")
        print("   - Set OPENAI_API_KEY for OpenAI GPT")
        print("   - Set XAI_API_KEY for Grok")
        print("   - Install Ollama for local AI")
    
    # Check dependencies
    print("\n3. Checking dependencies...")
    try:
        import psutil
        import jsonschema
        print("All dependencies are installed")
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("   Run: pip install -r requirements.txt")
    
    print("\nSetup complete! Run 'magic-terminal' to start.")


def main():
    """Main entry point"""
    parser = create_parser()
    args = parser.parse_args()
    
    # Handle special commands
    if args.config:
        show_config()
        return
    
    if args.setup:
        run_setup()
        return
    
    # Set environment variables from args
    if args.ollama_url:
        os.environ['OLLAMA_URL'] = args.ollama_url
    
    if args.enable_fallback:
        os.environ['AI_TERMINAL_ALLOW_FALLBACK'] = '1'
    
    if args.debug:
        import logging
        logging.basicConfig(level=logging.DEBUG)
    
    # Start the terminal
    try:
        terminal = EnhancedAITerminal(enable_fallback=args.enable_fallback)
        terminal.run()
    except KeyboardInterrupt:
        print("\nGoodbye!")
    except Exception as e:
        print(f"Error starting Magic Terminal: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
