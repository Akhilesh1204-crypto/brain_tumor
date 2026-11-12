import os
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pydicom

def validate_brain_mri(img_pil):
    """
    Heuristic check to detect if the uploaded image is a brain MRI scan.
    Returns (is_valid: bool, reason: str or None)
    """
    arr = np.array(img_pil.convert('RGB'), dtype=np.float32)
    
    # 1. Color saturation/variance check (Medical MRIs are grayscale)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    rg_diff = np.mean(np.abs(r - g))
    gb_diff = np.mean(np.abs(g - b))
    color_diff = (rg_diff + gb_diff) / 2.0
    
    if color_diff > 24.0:
        return False, "The image appears to be a color photograph. Brain MRI neuro-scans are grayscale."
        
    # 2. Border darkness check (axial brain MRIs have dark outer background)
    h, w, _ = arr.shape
    border_pixels = np.concatenate([
        arr[:max(2, int(h * 0.06)), :, :].flatten(),
        arr[-max(2, int(h * 0.06)):, :, :].flatten(),
        arr[:, :max(2, int(w * 0.06)), :].flatten(),
        arr[:, -max(2, int(w * 0.06)):, :].flatten()
    ])
    border_mean = np.mean(border_pixels)
    if border_mean > 95.0:
        return False, "The image lacks the dark background characteristic of cranial MRI scans."
        
    # 3. Minimum contrast / non-blank check
    if np.std(arr) < 8.0:
        return False, "The image appears blank or has insufficient contrast for cranial analysis."
        
    return True, None

def process_dicom(file_path, output_jpg_path):
    """
    Read DICOM (.dcm) file, extract clinical metadata, and convert to 8-bit RGB JPEG.
    Returns (pil_image, metadata_dict)
    """
    dcm = pydicom.dcmread(file_path)
    
    # Extract metadata tags
    metadata = {
        'modality': str(getattr(dcm, 'Modality', 'MR')),
        'patient_age': str(getattr(dcm, 'PatientAge', 'N/A')),
        'patient_sex': str(getattr(dcm, 'PatientSex', 'N/A')),
        'magnetic_field_strength': f"{getattr(dcm, 'MagneticFieldStrength', 'N/A')}T" if hasattr(dcm, 'MagneticFieldStrength') else '1.5T',
        'slice_thickness': f"{getattr(dcm, 'SliceThickness', 'N/A')} mm" if hasattr(dcm, 'SliceThickness') else 'N/A',
        'study_date': str(getattr(dcm, 'StudyDate', 'N/A')),
        'series_description': str(getattr(dcm, 'SeriesDescription', 'Axial Brain MRI'))
    }
    
    pixel_array = dcm.pixel_array.astype(np.float32)
    slope = getattr(dcm, 'RescaleSlope', 1.0)
    intercept = getattr(dcm, 'RescaleIntercept', 0.0)
    pixel_array = pixel_array * slope + intercept
    
    # Normalize to [0, 255]
    p_min = np.min(pixel_array)
    p_max = np.max(pixel_array)
    if p_max > p_min:
        norm_pixels = np.uint8(255 * (pixel_array - p_min) / (p_max - p_min))
    else:
        norm_pixels = np.zeros_like(pixel_array, dtype=np.uint8)
        
    if len(norm_pixels.shape) == 2:
        img = Image.fromarray(norm_pixels).convert('RGB')
    else:
        img = Image.fromarray(norm_pixels).convert('RGB')
        
    img.save(output_jpg_path, format='JPEG')
    return img, metadata

def generate_gradcam(model, img_array, predicted_class_idx, orig_pil_img, output_path):
    """
    Generate Grad-CAM heatmap overlay for the nested VGG16 model.
    """
    try:
        vgg16 = model.layers[0]
        last_conv = vgg16.get_layer('block5_conv3')
        
        # Intermediate model extracting last conv activations and vgg16 output
        grad_model = tf.keras.Model(vgg16.inputs, [last_conv.output, vgg16.output])
        
        with tf.GradientTape() as tape:
            conv_outputs, vgg_outputs = grad_model(img_array)
            tape.watch(conv_outputs)
            
            # Pass through the rest of the classifier layers
            y = vgg_outputs
            for layer in model.layers[1:]:
                y = layer(y)
            loss = y[:, predicted_class_idx]
            
        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        conv_outputs_val = conv_outputs[0]
        heatmap = conv_outputs_val @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0.0) / (tf.math.reduce_max(heatmap) + 1e-10)
        heatmap_np = heatmap.numpy()
        
        # Resize heatmap to match original image dimensions
        orig_w, orig_h = orig_pil_img.size
        heatmap_img = Image.fromarray(np.uint8(255 * heatmap_np)).resize((orig_w, orig_h), Image.Resampling.BILINEAR)
        heatmap_resized = np.array(heatmap_img) / 255.0
        
        # Colorize using Jet colormap
        cmap = plt.colormaps.get_cmap('jet')
        heatmap_colored = cmap(heatmap_resized)[:, :, :3]  # Drop alpha
        
        # Blend original image and heatmap
        orig_np = np.array(orig_pil_img.convert('RGB'), dtype=np.float32) / 255.0
        superimposed = 0.55 * orig_np + 0.45 * heatmap_colored
        superimposed = np.uint8(255 * np.clip(superimposed, 0.0, 1.0))
        
        overlay_pil = Image.fromarray(superimposed)
        overlay_pil.save(output_path, format='JPEG', quality=95)
        return True
    except Exception as e:
        print(f"Error generating Grad-CAM: {e}")
        return False
