from flask import Flask, request, jsonify
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import torch
import os

# Initialize the Flask app
app = Flask(__name__)

# Load the CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

# Confidence threshold
threshold = 0.7

# Unrelated product names for negative sampling
unrelated_names = [
    "Random Item 1", "Unrelated Product 2", "Example Product 3"
]

# Endpoint to process image and predict product name
@app.route('/predict_product', methods=['POST'])
def predict_product():
    try:
        # Get the product name and image file from the request
        product_name = request.form.get('product_name')
        image_file = request.files.get('image')

        if not product_name or not image_file:
            return jsonify({"error": "Product name and image are required"}), 400

        # Save the image to a temporary file
        temp_image_path = os.path.join("temp_image.jpg")
        image_file.save(temp_image_path)

        # Open the image
        image = Image.open(temp_image_path)

        # Prepare text inputs (product name + unrelated names)
        text_inputs = [product_name] + unrelated_names

        # Prepare inputs for the model
        inputs = processor(text=text_inputs, images=image, return_tensors="pt", padding=True)

        # Perform inference
        with torch.no_grad():
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image  # Image-text similarity scores
            probs = logits_per_image.softmax(dim=1)  # Convert to probabilities

        # Extract confidence scores
        confidences = probs[0].tolist()

        # Confidence for the actual product name
        actual_confidence = confidences[0]

        # Find the highest confidence score and its index
        max_confidence = max(confidences)
        max_idx = confidences.index(max_confidence)
        predicted_name = text_inputs[max_idx]

        # Determine if the match is correct
        is_correct = max_idx == 0 and actual_confidence > threshold

        # Clean up temporary image file
        os.remove(temp_image_path)

        # Return the results as a JSON response
        return jsonify({
            "product_name": product_name,
            "predicted_name": predicted_name,
            "actual_confidence": round(actual_confidence, 2),
            "highest_confidence": round(max_confidence, 2),
            "verification": "Correct" if is_correct else "Incorrect",
            "confidence_details": {text: round(score, 2) for text, score in zip(text_inputs, confidences)}
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Run the app
if __name__ == '__main__':
    app.run(debug=True,port=5001)
