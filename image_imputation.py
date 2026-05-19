import cv2
import os
import numpy as np
from datasets import load_dataset
import itertools
from tqdm import tqdm


def process_video_dataset(video_path, output_path, replacement_img_path):
    # 1. Load the video and the replacement image
    cap = cv2.VideoCapture(video_path)
    target_img = cv2.imread(replacement_img_path)
    
    # Decrease brightness and sharpness to blend better with the environment
    if target_img is not None:
        target_img = cv2.convertScaleAbs(target_img, alpha=0.85, beta=0)
        target_img = cv2.GaussianBlur(target_img, (3, 3), 0)
        
    # Get video properties for saving the output
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    # Target image dimensions (corners of the image we want to insert)
    th, tw, _ = target_img.shape
    src_pts = np.array([[0, 0], [tw - 1, 0], [tw - 1, th - 1], [0, th - 1]], dtype=np.float32)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # 2. Convert to HSV color space to easily isolate the green paper
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define range for chroma key green (adjust these values based on your lighting)
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        kernel = np.ones((5, 5), np.uint8)
        
        # Create a binary mask where green is white, everything else is black
        mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # 3. Find contours (the edges of the green papers)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        valid_contours = []
        for contour in contours:
            # Filter out tiny specks of noise by checking the area size
            if cv2.contourArea(contour) > 1000: 
                # Use convex hull to bridge gaps caused by occlusions on the edges
                hull = cv2.convexHull(contour)
                peri = cv2.arcLength(hull, True)
                approx = cv2.approxPolyDP(hull, 0.02 * peri, True)
                
                # If the polygon has 4 points, we found a card!
                if len(approx) == 4:
                    valid_contours.append(approx)
                    
        # Sort contours from left to right to accurately identify the rightmost paper
        valid_contours.sort(key=lambda c: cv2.boundingRect(c)[0])
        
        for i, approx in enumerate(valid_contours):
            # Reshape points to a standard 4x2 array
            dst_pts = approx.reshape(4, 2).astype(np.float32)
            
            # Sort the destination points clockwise
            center = dst_pts.mean(axis=0)
            angles = np.arctan2(dst_pts[:, 1] - center[1], dst_pts[:, 0] - center[0])
            ordered_dst = dst_pts[np.argsort(angles)]
            
            # Enforce the correct rotation by picking the proper top-left point
            # Heavily favor the top-most point for the rightmost paper to ensure rightward (CW) rotation
            if i == len(valid_contours) - 1:
                scores = ordered_dst[:, 0] + 5.0 * ordered_dst[:, 1]
            else:
                scores = ordered_dst[:, 0] + ordered_dst[:, 1]
            tl_idx = np.argmin(scores)
            ordered_dst = np.roll(ordered_dst, -tl_idx, axis=0)

            # 4. Calculate the Homography matrix and warp the image
            H, _ = cv2.findHomography(src_pts, ordered_dst)
            warped_img = cv2.warpPerspective(target_img, H, (frame_width, frame_height), borderMode=cv2.BORDER_REPLICATE)
            
            # 5. Overwrite the green paper area with our warped image
            # Create a mask for the warped image area
            poly_mask = cv2.warpPerspective(np.ones((th, tw), dtype=np.uint8) * 255, H, (frame_width, frame_height))
            mask_warped = cv2.bitwise_and(poly_mask, mask) # Intersect with the green mask
            mask_warped = cv2.dilate(mask_warped, kernel, iterations=1)
            
            # Feather the mask for seamless blending
            mask_blurred = cv2.GaussianBlur(mask_warped, (7, 7), 0)
            alpha = cv2.cvtColor(mask_blurred, cv2.COLOR_GRAY2BGR).astype(float) / 255.0
            
            # Blend the original frame and the warped image using the alpha mask
            frame = (frame * (1.0 - alpha) + warped_img * alpha).astype(np.uint8)
        
        # Write the processed frame to the output video
        out.write(frame)
        
    cap.release()
    out.release()
    cv2.destroyAllWindows()


def replace_green_papers_in_image(input_path, output_path, replacement_paths):
    # 1. Load the input image
    frame = cv2.imread(input_path)
    if frame is None:
        print(f"Error: Could not load input image at {input_path}")
        return
        
    frame_height, frame_width = frame.shape[:2]
    
    # Load the replacement images
    # Load the replacement images and decrease brightness and sharpness to blend better
    replacements = []
    for path in replacement_paths:
        img = cv2.imread(path)
        if img is not None:
            img = cv2.convertScaleAbs(img, alpha=0.9, beta=0)
            img = cv2.GaussianBlur(img, (3, 3), 0)
        replacements.append(img)
    
    # 2. Convert to HSV color space to isolate the green papers
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # 3. Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter and collect valid contours (having 4 corners and sufficient area)
    valid_contours = []
    for contour in contours:
        if cv2.contourArea(contour) > 1000:
            hull = cv2.convexHull(contour)
            peri = cv2.arcLength(hull, True)
            approx = cv2.approxPolyDP(hull, 0.02 * peri, True)
            if len(approx) == 4:
                valid_contours.append(approx)
                
    # Sort contours from left to right based on their x-coordinate
    # This ensures a predictable mapping of replacement images (left paper gets 1st image, etc.)
    valid_contours.sort(key=lambda c: cv2.boundingRect(c)[0])
    
    # 4. Map each replacement image to a detected paper
    for i, (approx, target_img) in enumerate(zip(valid_contours, replacements)):
        if target_img is None:
            continue
            
        th, tw, _ = target_img.shape
        src_pts = np.array([[0, 0], [tw - 1, 0], [tw - 1, th - 1], [0, th - 1]], dtype=np.float32)
        
        dst_pts = approx.reshape(4, 2).astype(np.float32)
        center = dst_pts.mean(axis=0)
        angles = np.arctan2(dst_pts[:, 1] - center[1], dst_pts[:, 0] - center[0])
        ordered_dst = dst_pts[np.argsort(angles)]
        
        # Enforce correct rotation (top-left starting point)
        if i == len(valid_contours) - 1:
            # Favor top-most point to ensure a rightward (clockwise) rotation
            scores = ordered_dst[:, 0] + 5.0 * ordered_dst[:, 1]
        else:
            scores = ordered_dst[:, 0] + ordered_dst[:, 1]
        tl_idx = np.argmin(scores)
        ordered_dst = np.roll(ordered_dst, -tl_idx, axis=0)

        # Calculate Homography and warp the image
        H, _ = cv2.findHomography(src_pts, ordered_dst)
        warped_img = cv2.warpPerspective(target_img, H, (frame_width, frame_height), borderMode=cv2.BORDER_REPLICATE)
        
        # Mask and overwrite
        poly_mask = cv2.warpPerspective(np.ones((th, tw), dtype=np.uint8) * 255, H, (frame_width, frame_height))
        mask_warped = cv2.bitwise_and(poly_mask, mask)
        mask_warped = cv2.dilate(mask_warped, kernel, iterations=1)
        
        # Feather the mask for seamless blending
        mask_blurred = cv2.GaussianBlur(mask_warped, (7, 7), 0)
        alpha = cv2.cvtColor(mask_blurred, cv2.COLOR_GRAY2BGR).astype(float) / 255.0
        
        frame = (frame * (1.0 - alpha) + warped_img * alpha).astype(np.uint8)
        
    # Save the output image
    cv2.imwrite(output_path, frame)
    print(f"Saved imputed image to {output_path}")

def replace_green_papers_in_video(input_video_path, output_video_path, replacement_paths, start_frame=0, end_frame=None):
    # Load the replacement images once to save I/O operations
    # Load the replacement images once to save I/O operations, adjusting brightness and sharpness
    replacements = []
    for path in replacement_paths:
        img = cv2.imread(path)
        if img is not None:
            img = cv2.convertScaleAbs(img, alpha=0.85, beta=0)
            img = cv2.GaussianBlur(img, (3, 3), 0)
        replacements.append(img)
    
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {input_video_path}")
        return
        
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    kernel = np.ones((5, 5), np.uint8)
    
    # Set the starting frame position
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Will hold the pre-calculated static warped images and their full masks
    static_data = [] 
    union_mask = None
    successful_frames = 0
    is_locked = False
    
    frame_count = start_frame
    while cap.isOpened():
        if end_frame is not None and frame_count > end_frame:
            break
            
        ret, frame = cap.read()
        if not ret:
            break
            
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # 1. Accumulate mask over the first 3 successful frames to get the maximum union area
        if not is_locked:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            valid_contours = []
            for contour in contours:
                if cv2.contourArea(contour) > 1000:
                    hull = cv2.convexHull(contour)
                    peri = cv2.arcLength(hull, True)
                    approx = cv2.approxPolyDP(hull, 0.02 * peri, True)
                    if len(approx) == 4:
                        valid_contours.append(approx)
                        
            # Accumulate mask if all papers are visible
            if len(valid_contours) == len(replacements):
                if union_mask is None:
                    union_mask = mask.copy()
                else:
                    union_mask = cv2.bitwise_or(union_mask, mask)
                    
                valid_contours.sort(key=lambda c: cv2.boundingRect(c)[0])
                
                temp_static_data = []
                for i, (approx, target_img) in enumerate(zip(valid_contours, replacements)):
                    if target_img is None:
                        continue
                        
                    th, tw, _ = target_img.shape
                    src_pts = np.array([[0, 0], [tw - 1, 0], [tw - 1, th - 1], [0, th - 1]], dtype=np.float32)
                    
                    dst_pts = approx.reshape(4, 2).astype(np.float32)
                    center = dst_pts.mean(axis=0)
                    angles = np.arctan2(dst_pts[:, 1] - center[1], dst_pts[:, 0] - center[0])
                    ordered_dst = dst_pts[np.argsort(angles)]
                    
                    if i == len(valid_contours) - 1:
                        # Heavily favor top-most point for rightmost image to enforce rightward (CW) rotation
                        scores = ordered_dst[:, 0] + 5.0 * ordered_dst[:, 1]
                    else:
                        scores = ordered_dst[:, 0] + ordered_dst[:, 1]
                    tl_idx = np.argmin(scores)
                    ordered_dst = np.roll(ordered_dst, -tl_idx, axis=0)

                    H, _ = cv2.findHomography(src_pts, ordered_dst)
                    if H is not None:
                        warped = cv2.warpPerspective(target_img, H, (frame_width, frame_height), borderMode=cv2.BORDER_REPLICATE)
                        poly = cv2.warpPerspective(np.ones((th, tw), dtype=np.uint8) * 255, H, (frame_width, frame_height))
                        temp_static_data.append((warped, poly))
                
                static_data = temp_static_data
                successful_frames += 1
                
                if successful_frames == 3:
                    # Use the combined union mask to find the maximum stable area of the green papers
                    union_contours, _ = cv2.findContours(union_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    union_valid = []
                    for contour in union_contours:
                        if cv2.contourArea(contour) > 1000:
                            hull = cv2.convexHull(contour)
                            peri = cv2.arcLength(hull, True)
                            approx = cv2.approxPolyDP(hull, 0.02 * peri, True)
                            if len(approx) == 4:
                                union_valid.append(approx)
                                
                    # Process the stable shapes. If the union fails to produce exactly 3 clean squares, 
                    # we safely fall back to keeping the `static_data` from the most recent successful frame.
                    if len(union_valid) == len(replacements):
                        union_valid.sort(key=lambda c: cv2.boundingRect(c)[0])
                        final_static_data = []
                        for i, (approx, target_img) in enumerate(zip(union_valid, replacements)):
                            if target_img is None:
                                continue
                            th, tw, _ = target_img.shape
                            src_pts = np.array([[0, 0], [tw - 1, 0], [tw - 1, th - 1], [0, th - 1]], dtype=np.float32)
                            
                            dst_pts = approx.reshape(4, 2).astype(np.float32)
                            center = dst_pts.mean(axis=0)
                            angles = np.arctan2(dst_pts[:, 1] - center[1], dst_pts[:, 0] - center[0])
                            ordered_dst = dst_pts[np.argsort(angles)]
                            
                            if i == len(union_valid) - 1:
                                scores = ordered_dst[:, 0] + 5.0 * ordered_dst[:, 1]
                            else:
                                scores = ordered_dst[:, 0] + ordered_dst[:, 1]
                            tl_idx = np.argmin(scores)
                            ordered_dst = np.roll(ordered_dst, -tl_idx, axis=0)

                            H, _ = cv2.findHomography(src_pts, ordered_dst)
                            if H is not None:
                                warped = cv2.warpPerspective(target_img, H, (frame_width, frame_height), borderMode=cv2.BORDER_REPLICATE)
                                poly = cv2.warpPerspective(np.ones((th, tw), dtype=np.uint8) * 255, H, (frame_width, frame_height))
                                final_static_data.append((warped, poly))
                        
                        if len(final_static_data) == len(replacements):
                            static_data = final_static_data
                            
                    is_locked = True
            else:
                frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                print(f"Warning: Could not detect all papers in frame {frame_number}!")
                
        # 2. Apply pre-calculated static warping intersected with the CURRENT frame's green mask
        if static_data:
            for warped_img, poly_mask in static_data:
                # mask is where green is visible *right now*. 
                # If the robot arm is over the paper, mask is 0 there, leaving the arm untouched!
                mask_warped = cv2.bitwise_and(poly_mask, mask) 
                mask_warped = cv2.dilate(mask_warped, kernel, iterations=1)
                
                # Feather the mask for seamless blending
                mask_blurred = cv2.GaussianBlur(mask_warped, (7, 7), 0)
                alpha = cv2.cvtColor(mask_blurred, cv2.COLOR_GRAY2BGR).astype(float) / 255.0
                
                frame = (frame * (1.0 - alpha) + warped_img * alpha).astype(np.uint8)
                
        out.write(frame)
        frame_count += 1
        
    cap.release()
    out.release()
    print(f"Saved imputed video to {output_video_path}")

def crop_to_4_3(img):
        w, h = img.size
        target_ratio = 3 / 4  # width / height (for a height:width = 4:3)
        if w / h > target_ratio:
            # Image is too wide
            new_w = int(h * target_ratio)
            new_h = h
        else:
            # Image is too tall
            new_w = w
            new_h = int(w / target_ratio)
            
        left = (w - new_w) // 2
        top = (h - new_h) // 2
        right = left + new_w
        bottom = top + new_h
        
        return img.crop((left, top, right, bottom))

def save_images():
    # Load the dataset from Hugging Face
    dataset = load_dataset("guloyy/celebrities_TOY", name="cropped", split="train")
    
    os.makedirs("obama_images", exist_ok=True)
    os.makedirs("taylor_images", exist_ok=True)
    os.makedirs("yann_images", exist_ok=True)
    
    obama_count = 0
    taylor_count = 0
    yann_count = 0
    
    for row in dataset:
        cropped_img = crop_to_4_3(row["image"])
        # Convert label integer to its string name if it's a ClassLabel, or check raw string
        if row["label"] == 0:  
            cropped_img.save(f"obama_images/image_{obama_count}.png")
            obama_count += 1
        elif row["label"] == 1:  
            cropped_img.save(f"taylor_images/image_{taylor_count}.png")
            taylor_count += 1
        elif row["label"] == 2:  
            cropped_img.save(f"yann_images/image_{yann_count}.png")
            yann_count += 1
    
    print(f"Saved {obama_count} Obama images, {taylor_count} Taylor images, {yann_count} Yann images.")

def generate_synthetic_images(n_samples = 3):
    # This function can be implemented to create synthetic variations of the images (e.g., rotations, brightness changes)
    template = "green_screen_template.png"
    combination_indexes = np.random.choice(range(27000), size=n_samples, replace=False)
    for i, idx in enumerate(tqdm(combination_indexes, desc="Generating synthetic images")):
        obama_idx = idx % 30
        taylor_idx = (idx // 30) % 30
        yann_idx = (idx // 900) % 30
        permutation_iter = itertools.permutations(["obama", "taylor", "yann"])
        for perm in permutation_iter:
            replacement_paths = [f"{perm[0]}_images/image_{obama_idx}.png", f"{perm[1]}_images/image_{taylor_idx}.png", f"{perm[2]}_images/image_{yann_idx}.png"]
            output_path = f"permutations/{perm[0][0]}{perm[1][0]}{perm[2][0]}/image_{i}.jpg"
            replace_green_papers_in_image(template, output_path, replacement_paths)

        
        

if __name__ == "__main__":
    # Example usage for video processing
    replace_green_papers_in_video("./teleop_videos/file-000.mp4", "output_video_occlusion_teleop.mp4", ["obama_images/image_0.png", "taylor_images/image_0.png", "yann_images/image_0.png"], start_frame=500, end_frame=1000)
    
    # Example usage for image processing
    #replace_green_papers_in_image("green_screen_template.png", "output_image.jpg", ["obama_images/image_0.png", "taylor_images/image_1.png", "yann_images/image_2.png"])
    #save_images()
    #generate_synthetic_images(n_samples=2000)