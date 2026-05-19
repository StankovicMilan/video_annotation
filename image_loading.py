from datasets import load_dataset
from IPython.display import display

def display_celebrity_image():
    # Load the dataset from Hugging Face
    dataset = load_dataset("guloyy/celebrities_TOY", name="cropped", split="train")
    
    # Filter for rows where the label string or integer representation matches Barack Obama
    # The dataset uses a ClassLabel feature, so we can check the string name directly
    obama_images = []
    for row in dataset:
        # Convert label integer to its string name if it's a ClassLabel, or check raw string
        if row["label"] == 0:  
            obama_images.append(row["image"])
    obama_images[0].show()
        
        #label_str = dataset.features["label"].int2str(row["label"]) if hasattr(dataset.features["label"], "int2str") else str(row["label"])
        
    #     if "barack obama" in label_str.lower():
    #         obama_images.append(row["image"])
            
    # print(f"Found {len(obama_images)} images of Barack Obama.")
    
    # # Display the first matching image
    # if obama_images:
    #     obama_images[0].show()
    #     # If in a Jupyter Notebook, use: display(obama_images[0])
    # else:
    #     print("No images found matching the filter criteria.")

# Example usage for loading from Hugging Face Hub:
# display_celebrity_image("guloyy/celebrities_TOY", "Barack Obama")

# Example usage for loading from a local cloned repository:
display_celebrity_image()