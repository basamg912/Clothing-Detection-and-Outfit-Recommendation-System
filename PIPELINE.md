# Project Pipeline: AI-Based Clothing Recommendation System

This document outlines the end-to-end pipeline of the project, which integrates computer vision (YOLO), a custom neural network for recommendations, and a web-based user interface.

## High-Level Architecture

The system operates as a web application where users upload images of clothing. The backend processes these images to detect, classify, and extract attributes (color). It then uses a recommendation model to suggest optimal outfit combinations ("cody"), potentially factoring in weather conditions.

### Core Components
1.  **Web Interface (Flask)**: Handles user interactions, image uploads, and result visualization.
2.  **Object Detection (YOLOv8)**: Detects and crops clothing items from uploaded images.
3.  **Feature Extraction**: Analyzes color properties of the detected items.
4.  **Recommendation Engine (ResNet + MLP)**: Scores the compatibility between different clothing items.
5.  **Context Provider (Weather API)**: Fetches real-time weather data to influence recommendations (Integrated/Planned).

---

## Detailed Pipeline Steps

### 1. User Input (Web UI)
*   **Interface**: `index.html`
*   **Action**: User uploads a raw image containing one or more clothing items.
*   **Handler**: `app.py` (`index` route).

### 2. Clothing Detection & Classification
*   **Model**: YOLOv8 (`fine_tune_yolo.pt`)
*   **Process**:
    1.  Image is preprocessed (EXIF orientation fix).
    2.  YOLO model detects objects in the image.
    3.  **Cropping**: Detected regions are cropped and saved as individual files in `static/uploads/`.
    4.  **Categorization**: Each crop is classified into broad categories (`top`, `bottom`, `outer`, `shoe`) based on the YOLO class labels.
*   **File**: `dummy_model.py` (`classify_clothes` function).

### 3. Attribute Extraction (Color)
*   **Method**: HSV Color Space Analysis.
*   **Process**:
    1.  Extracts center pixels of the cropped image.
    2.  Converts RGB to HSV.
    3.  Classifies the dominant color into discrete buckets (e.g., `navy`, `beige`, `white`, `khaki`) using threshold logic.
*   **File**: `dummy_model.py` (`detect_color` function).

### 4. Data Storage (Session-based)
*   Detected items are stored in an in-memory structure (`category_items` dictionary) within the Flask app, acting as a temporary "closet" for the user session.
*   **Interface**: Users can view their items via `closet.html`.

### 5. Outfit Recommendation
*   **Trigger**: User navigates to `/recommend`.
*   **Model Architecture**: Siamese-style Network
    *   **Backbone**: ResNet50 (pretrained) acts as a feature embedder (`ResNetEmbedder`).
    *   **Classifier**: Multi-Layer Perceptron (MLP) takes concatenated embeddings of two items (e.g., Top + Bottom).
    *   **Output**: A compatibility score (0-1).
*   **Process**:
    1.  The system iterates through all combinations of available 'Top' and 'Bottom' items.
    2.  Each pair is passed through the `ImageRelationClassifier`.
    3.  Pairs are ranked by their compatibility score.
    4.  The highest-scoring pair is selected.
*   **Weights**: `cody_recommend(second_train).pth`.
*   **File**: `dummy_model.py` (`recommend_cody`, `ImageRelationClassifier`, `make_model`).

### 6. Contextual Refinement (Weather)
*   **Source**: KMA (Korea Meteorological Administration) Ultra Short Term Forecast API.
*   **Function**: `weather_fetch.py` fetches real-time data (Temp, Sky condition, Rain probability).
*   **Role**: Used to filter or prioritize items (e.g., suggesting an 'Outer' if the temperature is low). *Note: Logic is integrated into the recommendation flow decision making.*

### 7. Output Presentation
*   **Interface**: `recommend.html`
*   **Display**: Shows the recommended Top, Bottom, and Shoes (plus Outer if applicable).
*   **Metrics**: Displays the compatibility score (scaled 0-100).

---

## Directory Structure Overview

*   `Project/app.py`: Main application controller.
*   `Project/dummy_model.py`: Contains inference logic for YOLO, Color detection, and Recommendation models.
*   `Project/weather_fetch.py`: Weather data retrieval module.
*   `Project/fine_tune_yolo.pt`: YOLOv8 weights for clothing detection.
*   `Project/cody_recommend*.pth`: Weights for the recommendation neural network.
*   `Project/templates/`: HTML views for the UI.
