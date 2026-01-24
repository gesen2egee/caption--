from PIL import Image, ImageFilter, ImageChops

def process_mask_channel(mask: Image.Image, shrink: int, blur: float, min_alpha: int) -> Image.Image:
    """
    Process a single channel mask (L mode).
    1. Shrink (Erode) - Simulated by MinFilter? Or negative expand?
       Actually, standard erosion shrinks white regions.
       If mask is Alpha channel where 255=Opaque (Keep), 0=Transparent (Remove).
       Start with full opaque (255) and carve out transparent parts?
       Usually Unmask/MaskText returns an Image with Alpha.
       The "Mask" is the Alpha channel.
       
       If shrinking the "Mask" (the opaque part), we use MinFilter.
       BUT, "Mask Text" usually makes text transparent (0). So we want to Expand the transparent region?
       If we shrink the opaque region (255), the transparent region (0) grows.
       
       Let's assume:
       - shrink_size > 0: Erode the opaque area (255 area gets smaller, 0 area gets bigger).
       
    2. Gaussian Blur.
    3. Clamp Min (Ensure alpha >= min_alpha).
    """
    processed = mask.copy()
    
    # 1. Shrink (Erode opaque area)
    if shrink > 0:
        # MinFilter erodes bright areas (255)
        # Using an odd kernel size ~ 2*shrink + 1 is common, or iteration.
        # PIL MinFilter size is diameter.
        processed = processed.filter(ImageFilter.MinFilter(size=shrink * 2 + 1))
        
    # 2. Blur
    if blur > 0:
        processed = processed.filter(ImageFilter.GaussianBlur(radius=blur))
        
    # 3. Clamp Min
    if min_alpha > 0:
        # pixel = max(pixel, min_alpha)
        # Use point operation
        processed = processed.point(lambda x: max(x, min_alpha))
        
    return processed

def apply_advanced_mask_processing(original_img: Image.Image, 
                                   mask_img: Image.Image, 
                                   shrink: int, 
                                   blur: float, 
                                   min_alpha: int,
                                   is_original_alpha_processing: bool = False,
                                   original_bg_fill_color: int = 255) -> Image.Image:
    """
    Apply mask processing to an image.
    
    Args:
        original_img: Source RGB/RGBA image.
        mask_img: The mask to apply (L mode, 0=Transparent, 255=Opaque).
                  OR an RGBA image where Alpha is the mask.
        shrink: Shrink pixels.
        blur: Blur radius.
        min_alpha: Minimum alpha value (clamp).
        is_original_alpha_processing: If True, fill 0-alpha pixels in original with white before processing?
                                      User said: "原始alpha alpha=0像素填補白色"
    
    Returns:
        RGBA image with processed alpha.
    """
    img = original_img.convert("RGBA")
    
    # Extract alpha from mask_img
    if mask_img.mode == 'RGBA':
        target_alpha = mask_img.getchannel('A')
    elif mask_img.mode == 'L':
        target_alpha = mask_img
    else:
        target_alpha = mask_img.convert('L')
        
    # Process the target alpha
    processed_alpha = process_mask_channel(target_alpha, shrink, blur, min_alpha)
    
    # Logic for merging?
    # User said: "注意RGB值不要影響 先在alpha操作並合併取最小值"
    # Merging with original alpha?
    
    current_alpha = img.getchannel('A')
    
    # "原始alpha alpha=0像素填補白色" -> This implies modifying the RGB channels of the original image
    # where it was transparent, to be White, so that if we unmask it later (via blur?), it shows white?
    # Or just for calculation?
    # User: "原始alpha alpha=0像素填補白色  內縮 > 高斯模糊 > clamp最低值(用背景的設定值)"
    # This likely refers to specific processing of the *Original Image's* existing alpha channel.
    
    # But usually we are applying a NEW mask (from Unmask or TextMask).
    # The final alpha = min(Original Alpha, New Mask Alpha).
    
    # Let's just return the processed alpha for now, caller handles merging.
    return processed_alpha
