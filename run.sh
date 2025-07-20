#!/bin/bash

# CEcBaN Run Script

set -e  # Exit on any error


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


command_exists() {
    command -v "$1" >/dev/null 2>&1
}


SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"


ENV_NAME="cecban_env"
APP_FILE="app.py"
HOST="127.0.0.1"
PORT="8050"
APP_URL="http://$HOST:$PORT"

print_status "Starting CEcBaN application..."


if [ ! -d "$ENV_NAME" ]; then
    print_error "Virtual environment '$ENV_NAME' not found!"
    print_error "Please run './install.sh' first to set up the environment."
    exit 1
fi


if [ ! -f "$APP_FILE" ]; then
    print_error "Application file '$APP_FILE' not found!"
    print_error "Please make sure the main application file is in the current directory."
    exit 1
fi


print_status "Activating virtual environment..."
source "$ENV_NAME/bin/activate" || {
    print_error "Failed to activate virtual environment!"
    exit 1
}

print_success "Virtual environment activated"


required_files=("1_CCM_ECCM.py" "2_SURR.py" "3_BN.py")
missing_files=()

for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        missing_files+=("$file")
    fi
done

if [ ${#missing_files[@]} -ne 0 ]; then
    print_warning "Some required files are missing:"
    for file in "${missing_files[@]}"; do
        echo "  - $file"
    done
    print_warning "The application may not work correctly without these files."
    read -p "Do you want to continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_status "Exiting..."
        exit 1
    fi
fi


wait_for_server() {
    local max_attempts=30
    local attempt=1
    
    print_status "Waiting for server to be ready..."
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s "$APP_URL" >/dev/null 2>&1; then
            print_success "Server is ready!"
            return 0
        fi
        
        echo -n "."
        sleep 1
        attempt=$((attempt + 1))
    done
    
    print_warning "Server didn't respond after $max_attempts seconds"
    print_warning "You may need to wait a bit longer for the app to load"
    return 1
}


open_browser() {
    print_status "Opening browser..."
    
    if command_exists xdg-open; then
        xdg-open "$APP_URL" >/dev/null 2>&1 &
    elif command_exists gnome-open; then
        gnome-open "$APP_URL" >/dev/null 2>&1 &
    elif command_exists firefox; then
        firefox "$APP_URL" >/dev/null 2>&1 &
    elif command_exists google-chrome; then
        google-chrome "$APP_URL" >/dev/null 2>&1 &
    elif command_exists chromium-browser; then
        chromium-browser "$APP_URL" >/dev/null 2>&1 &
    else
        print_warning "Could not find a suitable browser to open automatically."
        print_status "Please open your browser manually and go to: $APP_URL"
        return
    fi
    
    print_success "Browser opened to $APP_URL"
}


cleanup() {
    print_status "Shutting down CEcBaN application..."
    deactivate 2>/dev/null || true
    exit 0
}


trap cleanup SIGINT SIGTERM


if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
    print_warning "Port $PORT is already in use!"
    print_status "Trying to kill existing process..."
    sudo lsof -ti:$PORT | xargs sudo kill -9 2>/dev/null || true
    sleep 2
fi


export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1


print_status "Starting CEcBaN server on $APP_URL"
print_status "Press Ctrl+C to stop the application"
echo


python "$APP_FILE" &
APP_PID=$!


if wait_for_server; then
    open_browser
else
    print_status "Opening browser anyway - server might still be starting..."
    open_browser
fi

print_success "CEcBaN server is running!"
print_status "Access the application at: $APP_URL"
print_status "Use Ctrl+C to stop the server"
echo


wait $APP_PID || {
    print_error "Application stopped unexpectedly!"
    exit 1
}
