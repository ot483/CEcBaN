#!/bin/bash

# CEcBaN LLM Installation Script

set -e


RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' 

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}


SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

ENV_NAME="cecban_env"

print_status "Installing LLM packages for CEcBaN..."


if [ ! -d "$ENV_NAME" ]; then
    print_error "Virtual environment '$ENV_NAME' not found!"
    print_error "Please run './install.sh' first to set up the environment."
    exit 1
fi


print_status "Activating virtual environment..."
source "$ENV_NAME/bin/activate" || {
    print_error "Failed to activate virtual environment!"
    exit 1
}

print_success "Virtual environment activated"


print_status "Installing OpenAI package..."
pip install openai>=1.0.0 || {
    print_error "Failed to install OpenAI package"
    exit 1
}

print_status "Installing Google AI package..."
pip install google-generativeai>=0.3.0 || {
    print_error "Failed to install Google AI package"
    exit 1
}

print_success "LLM packages installed successfully!"


print_status "Testing LLM package imports..."
python3 -c "
try:
    import openai
    print('OpenAI package working')
except ImportError:
    print('OpenAI package failed')

try:
    import google.generativeai
    print('Google AI package working')  
except ImportError:
    print('Google AI package failed')

print('LLM packages ready!')
" || {
    print_warning "Some packages may not be working correctly"
}

deactivate

print_success "LLM installation complete!"
echo
print_status "LLM Features Now Available!"
echo
print_status "To use AI-powered interaction discovery:"
echo "1. Get API keys:"
echo "   • OpenAI GPT-4: https://platform.openai.com/api-keys"
echo "   • Google Gemini: https://makersuite.google.com/app/apikey"
echo
echo "2. Set environment variables (optional):"
echo "   export OPENAI_API_KEY='your-openai-key'"
echo "   export GOOGLE_API_KEY='your-google-key'"
echo
echo "3. Or enter API keys directly in the CEcBaN interface"
echo
print_status "Start the application with: ./run.sh"
print_status "Look for the 'AI Literature Discovery' section in the Known Interactions panel"
