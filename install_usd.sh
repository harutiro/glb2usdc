#!/bin/bash
# Pixar USD Toolsのインストールスクリプト

echo "Installing Pixar USD Tools..."

# Homebrewがインストールされているか確認
if ! command -v brew &> /dev/null; then
    echo "Homebrew is not installed. Please install Homebrew first:"
    echo "/bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
    exit 1
fi

echo "Installing USD via Homebrew..."
brew install usd

# パスを設定
USD_PATH=$(brew --prefix)/lib/python$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)/site-packages

echo ""
echo "Installation complete!"
echo ""
echo "To use USD tools, you may need to add the following to your shell configuration:"
echo "export PYTHONPATH=\"\$PYTHONPATH:$USD_PATH\""
echo "export PATH=\"\$PATH:$(brew --prefix)/bin\""
echo ""
echo "After installation, the gltf2usd command should be available."
