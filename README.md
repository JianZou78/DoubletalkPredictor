# Doubletalk Five Predictor - PyTorch Edition

## Quick Start:
1. Double-click "DoubletalkPredictor.exe" to start
2. If you encounter issues, run "test_pytorch.bat" first

## Features:
- Advanced AI-powered counting of "five" occurrences in audio
- Drag & drop WAV files or click to select
- Results rounded up using ceiling function (np.ceil)
- Full file paths displayed during processing
- Export results to CSV format
- Optional source file for delay compensation

## Files Included:
- DoubletalkPredictor.exe (main application)  
- doubletalk_model_v250722.pth (pre-trained AI model)
- doubletalk_ai_trainer.py (trainer module)
- test_pytorch.bat (diagnostic tool)
- PyTorch libraries and dependencies (~400+ MB)

## System Requirements:
- Windows 10/11 (x64)
- 8GB+ RAM recommended
- No Python installation required
- All dependencies included

## Usage Instructions:
1. Start the application
2. Wait for "Model loaded. Ready to predict!" message
3. Select WAV files by:
   - Dragging and dropping files onto the drop zone
   - Clicking the drop zone to open file selector
4. Optionally select a source file for delay compensation
5. View results in the text area (shows full paths and ceiled predictions)
6. Export results to CSV if needed

## Troubleshooting:
- If app doesn't start: Run "test_pytorch.bat" for diagnostics
- For large files: Processing may take several minutes
- Memory issues: Close other applications, restart if needed
- Model loading errors: Ensure all files stay in the same folder

## Technical Details:
- Built with PyInstaller and advanced PyTorch compatibility fixes
- Includes runtime patches for source inspection issues
- Uses directory-based packaging for better stability
- Pre-trained model: doubletalk_model_v250722.pth
- Sample rate: 16kHz audio processing

## Updates:
- Results now use np.ceil() for rounded-up predictions
- Full file paths shown in processing and summary
- Enhanced error handling and user feedback
- Optimized for Windows distribution

For support or issues, ensure all files in this folder remain together.
Built with Python 3.13 + PyTorch + LibROSA + PyInstaller
