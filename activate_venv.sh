#!/bin/bash
# Script to activate the MLX Audio virtual environment

echo "Activating MLX Audio virtual environment..."
source venv/bin/activate
echo "Virtual environment activated! You can now run MLX Audio commands."
echo ""
echo "To deactivate, run: deactivate"
echo "To run Python with MLX Audio: python"
echo ""
echo "Example usage:"
echo "  python -c \"import mlx_audio.tts; print('TTS module ready!')\""
echo ""
