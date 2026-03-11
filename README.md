# Motion Tracking Studio

A professional Streamlit + OpenCV application for motion-based object tracking using background subtraction.

## Features

- Clean, modern interface with a dedicated control sidebar
- Two detector options: MOG2 and KNN
- Adjustable sensitivity, contour filtering, and model history
- Optional mask cleanup with morphology operations
- Side-by-side live views: original, tracked, and foreground mask
- Processing telemetry (detections, frame progress, FPS)
- Supports local sample videos and uploaded files

## Project Structure

- `app.py` - Main Streamlit application
- `requirements.txt` - Python dependencies
- `vtest.avi` and other video files - Sample inputs

## Setup

1. Create and activate your virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

## Usage

1. Choose a video source from the sidebar.
2. Tune detector and playback settings.
3. Click **Start Tracking**.
4. Monitor detections and progress in real time.

## Notes

- Higher playback FPS makes visualization faster, but may reduce UI smoothness on lower-end hardware.
- Morphology cleanup helps reduce noise and false detections in busy scenes.
