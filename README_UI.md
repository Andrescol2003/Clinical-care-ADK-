# 🏥 Medical Transcription Generator - Web UI

A beautiful, user-friendly web interface for generating medical transcriptions using your fine-tuned BioGPT model.

## ✨ Features

- 🎨 **Beautiful UI**: Modern, medical-themed interface
- 🚀 **Easy to Use**: Just enter keywords and generate transcriptions
- ⚙️ **Customizable**: Adjust generation parameters (temperature, length, etc.)
- 💡 **Examples**: Pre-loaded example keywords to get started
- 🔄 **Real-time**: Instant generation with progress feedback

## 📋 Prerequisites

- Python 3.8+
- Fine-tuned BioGPT model (or will use base model as fallback)
- Required packages (see installation)

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements_ui.txt
```

### 2. Run the Application

```bash
python app.py
```

### 3. Open in Browser

The app will automatically open in your browser at:
- **Local**: http://localhost:7860
- **Network**: http://0.0.0.0:7860

## 📖 How to Use

1. **Enter Keywords**: Type medical keywords, conditions, or terms (comma-separated)
   - Example: `hypertension, medication, follow-up, blood pressure`

2. **Adjust Parameters** (Optional):
   - **Max Length**: Maximum tokens to generate (100-1000)
   - **Temperature**: Controls creativity (0.1-2.0)
   - **Top-p**: Nucleus sampling parameter (0.1-1.0)
   - **Use Sampling**: Enable/disable sampling

3. **Generate**: Click "Generate Transcription" or press Enter

4. **Review**: The generated medical transcription will appear in the output box

## 🎯 Example Keywords

- `hypertension, medication, follow-up`
- `chest pain, cardiology consult, ECG`
- `diabetes mellitus, glucose monitoring, insulin`
- `fever, cough, respiratory infection`
- `abdominal pain, gastroenterology, endoscopy`

## 📁 File Structure

```
biogpt_medical_finetuned/
├── app.py                 # Main UI application
├── requirements_ui.txt    # UI dependencies
├── README_UI.md          # This file
└── biogpt_medical_finetuned/  # Fine-tuned model directory
    ├── config.json
    ├── pytorch_model.bin
    └── tokenizer files...
```

## ⚙️ Configuration

### Model Path

Edit `MODEL_PATH` in `app.py` if your model is in a different location:

```python
MODEL_PATH = "./biogpt_medical_finetuned"  # Change this path
```

### Server Settings

Modify the `app.launch()` call in `app.py`:

```python
app.launch(
    server_name="0.0.0.0",  # Change to "127.0.0.1" for local only
    server_port=7860,       # Change port if needed
    share=True,             # Set to True for public link
)
```

## 🔧 Troubleshooting

### Model Not Found

If you see "Using base model" message:
- Check that your fine-tuned model is in the correct directory
- Verify the `MODEL_PATH` in `app.py` matches your model location
- The app will still work with the base BioGPT model

### CUDA/GPU Issues

If you get GPU errors:
- The app will automatically fall back to CPU
- For MPS (Apple Silicon), it will use MPS if available

### Port Already in Use

If port 7860 is busy:
- Change `server_port` in `app.launch()`
- Or kill the process using that port

## 🎨 Customization

### Change Theme

Edit the `theme` parameter in `create_interface()`:

```python
with gr.Blocks(css=custom_css, theme=gr.themes.Monochrome()) as app:
```

Available themes:
- `gr.themes.Soft()` (default)
- `gr.themes.Monochrome()`
- `gr.themes.Glass()`
- `gr.themes.Default()`

### Modify CSS

Edit the `custom_css` variable in `app.py` to change colors, fonts, etc.

## 📝 Notes

- **Always Review**: AI-generated medical content should always be reviewed by medical professionals
- **Model Size**: The fine-tuned model is ~1.5GB, ensure you have enough disk space
- **Performance**: First generation may be slower as the model loads into memory

## 🚀 Deployment

### Local Network Access

To access from other devices on your network:
- Keep `server_name="0.0.0.0"`
- Find your IP address: `ipconfig` (Windows) or `ifconfig` (Mac/Linux)
- Access from other devices: `http://YOUR_IP:7860`

### Public Sharing

For a temporary public link:
```python
app.launch(share=True)
```

This creates a Gradio share link (expires after 72 hours).

### Production Deployment

For production, consider:
- Using a reverse proxy (nginx)
- Adding authentication
- Using a proper web server (FastAPI + Gradio)
- Setting up SSL/HTTPS

## 📞 Support

If you encounter issues:
1. Check the console output for error messages
2. Verify all dependencies are installed
3. Ensure the model path is correct
4. Check GPU/CPU availability

## 🎉 Enjoy!

Your medical transcription generator is ready to use! 🏥✨

