# Simple Segment Anything Demo

A streamlined demo of Meta's Segment Anything Model (SAM) with a simple web interface. This demo includes both the original SAM and SlimSAM models for comparison.

## Features

- Simple point-based segmentation
- Comparison between SAM and SlimSAM outputs
- Automatic random color generation for each segmentation
- Web-based interface accessible from any browser
- GPU acceleration when available

## Installation

1. Clone this repository:
```bash
git clone https://github.com/parsakhaz/sam-gradio.git
cd sam-gradio
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows, use: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the application:
```bash
python sam_point.py
```

2. Open your web browser and navigate to:
   - Local access: `http://localhost:7860`
   - The application will also generate a public URL for remote access

3. Using the interface:
   - Upload an image using the input area
   - Enter X and Y coordinates where you want to segment an object
   - Click "Segment Object" to see the results
   - The results show both SlimSAM (faster) and SAM (more accurate) outputs
   - Each segmentation gets a unique color pair (dark outline, light fill)

## Requirements

- Python 3.8+
- CUDA-capable GPU recommended (but not required)
- See `requirements.txt` for Python package dependencies

## Models Used

- Original SAM: `facebook/sam-vit-huge`
- SlimSAM: `nielsr/slimsam-50-uniform`

## Notes

- The interface runs on port 7860 by default
- First run will download the model weights (~2.4GB for SAM, ~140MB for SlimSAM)
- Segmentation results are visualized with:
  - Random color selection for each segment
  - Dark outline for clear boundaries
  - Light, semi-transparent fill (35% opacity) 