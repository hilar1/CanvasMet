import numpy as np
from skimage.measure import regionprops
from skimage.segmentation import clear_border

METRICS_REGISTRY = []

def register_metric(func):
    METRICS_REGISTRY.append(func)
    return func

@register_metric
def calculate_astm_e112(mask_array, pixel_size_um):
    clean_masks = clear_border(mask_array)
    props = regionprops(clean_masks)
    areas_pixels = np.array([prop.area for prop in props])
    
    if len(areas_pixels) == 0:
        return {"Valid Grains Count": 0, "Mean Area (mm^2)": 0.0, "ASTM Grain Size (G)": 0.0, "Mean Intercept (um)": 0.0}

    conversion_factor_mm2 = (pixel_size_um / 1000.0) ** 2
    areas_mm2 = areas_pixels * conversion_factor_mm2
    mean_area_mm2 = np.mean(areas_mm2)
    astm_g = -3.3219 * np.log10(mean_area_mm2) - 2.954
    
    h_intersections = np.sum(np.diff(mask_array, axis=1) != 0)
    v_intersections = np.sum(np.diff(mask_array, axis=0) != 0)
    total_intersections = h_intersections + v_intersections
    
    height, width = mask_array.shape
    total_line_length_mm = (height * width * 2 * pixel_size_um) / 1000.0
    mean_intercept_um = (total_line_length_mm / total_intersections) * 1000.0 if total_intersections > 0 else 0.0

    return {
        "Valid Grains Count": len(areas_pixels),
        "Mean Area (mm^2)": mean_area_mm2,
        "ASTM Grain Size (G)": astm_g,
        "Mean Intercept (um)": mean_intercept_um
    }