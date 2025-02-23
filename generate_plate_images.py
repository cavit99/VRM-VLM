#!/usr/bin/env python3
"""
This script generates a set of UK car number plates (VRNs) and then renders
each plate as two JPEG images (front and rear) according to the following rules:

• Plate dimensions are 450×96px.
• The front plate uses a white background.
• The rear plate uses yellow – randomly either "#FFFF00" or "#FEDD00" (50-50 weighted).
• An optional flash strip is drawn on the left if any of the VRN's optional options 
  ('flash', 'flag', or 'country') are not "NONE". In the flash area the flag (if any) 
  is drawn on top and the optional country identifier below that.
• The VRN is drawn with fixed spacing and grouped as per the specification:
    • First group (the regional tag and age identifier) of 4 characters,
    • A larger gap,
    • A second group of 3 random letters.
• The font used is "Charles Wright 2001" (if available) at a size which gives a character
  height of approximately 68px and all other spacing is computed from the provided specs.
  
Usage:
    - Import and call generate_plates from the VRN module as:
          from generate_VRN import generate_plates
    - Run this script (by default 5 plates are generated; the number can be provided as an argument).

Make sure you have Pillow installed:
    pip install Pillow
"""

import os
import random
import sys

from PIL import Image, ImageDraw, ImageFont

# Import the VRN generator (change the module name/path as needed)
from generate_VRN import generate_plates  # our module that returns the plate data dictionaries

# ---------------------------
# Define conversion and layout constants
# ---------------------------
# The number plate spec is defined in mm for a plate of 520mm x 111mm.
# Our output image is 450x96px so we use a scale factor.
IMAGE_WIDTH = 450
IMAGE_HEIGHT = 96
ORIG_WIDTH_MM = 520  # original width in mm
ORIG_HEIGHT_MM = 111  # original height in mm

scale = IMAGE_WIDTH / ORIG_WIDTH_MM  # roughly 0.865

# Character specs (converted to px):
CHAR_HEIGHT = int(79 * scale)       # approx 68px character height
CHAR_WIDTH  = int(50 * scale)       # approx 50px character width (for most characters)
CHAR_SPACING = int(11 * scale)      # spacing in-between characters ~10px
GROUP_SPACING = int(33 * scale)     # spacing between the two groups ~29px
SIDE_MARGIN = int(11 * scale)       # left/right margin ~10px

# Vertical positioning for VRN text (centered vertically)
TEXT_Y = (IMAGE_HEIGHT - CHAR_HEIGHT) // 2  # ~14px top margin

# For flash area (if present) the width is randomly chosen between 40mm and 50mm, converted:
def get_flash_width():
    flash_mm = random.randint(40, 50)
    return int(flash_mm * scale)

def get_font_size_for_height(draw, font_path, target_height):
    """Calculate the font size needed to achieve a specific rendered height."""
    size = target_height  # Start with target height as initial guess
    test_font = ImageFont.truetype(font_path, size)
    test_text = "A"  # Use capital A as reference
    
    # Get the actual rendered height
    bbox = draw.textbbox((0, 0), test_text, font=test_font)
    actual_height = bbox[3] - bbox[1]
    
    # Adjust font size proportionally
    adjusted_size = int(size * (target_height / actual_height))
    return adjusted_size

def get_country_font_size(draw, font_path, text, max_width):
    """Calculate the largest font size that will fit the text within max_width."""
    size = 20  # Start with current size
    while size > 8:  # Don't go smaller than 8pt
        test_font = ImageFont.truetype(font_path, size)
        bbox = draw.textbbox((0, 0), text, font=test_font)
        text_width = bbox[2] - bbox[0]
        if text_width <= max_width:
            return size
        size -= 1
    return 8  # Minimum size if text is still too wide

# ---------------------------
# Render a single plate image. 
# This function draws the flash (if any) and the VRN text.
# ---------------------------
def render_plate(plate, bg_color, flash_width, font_path, small_font):
    img = Image.new("RGB", (IMAGE_WIDTH, IMAGE_HEIGHT), bg_color)
    draw = ImageDraw.Draw(img)
    
    # Calculate the font size needed for our target height
    font_size = get_font_size_for_height(draw, font_path, CHAR_HEIGHT)
    main_font = ImageFont.truetype(font_path, font_size)

    # Flash area logic 
    flash_present = (plate['flash'] != "NONE" or plate['flag'] != "NONE" or plate['country'] != "NONE")
    if flash_present and flash_width > 0:
        if plate['flash'] == "blue":
            flash_color = "#003DA5"
        elif plate['flash'] == "green":
            flash_color = "#00B74F"
        else:
            flash_color = bg_color
        draw.rectangle([0, 0, flash_width, IMAGE_HEIGHT], fill=flash_color)
        
        # Prepare variables for grouping flag and country text
        flag_img_resized = None
        flag_draw_width = 0
        flag_draw_height = 0
        
        # Try to load and resize flag image if exists
        if plate['flag'] != "NONE":
            flag_mapping = {
                "UK": "media/uk.png",
                "ENG": "media/england.png",
                "SCO": "media/scotland.png"
            }
            flag_path = flag_mapping.get(plate['flag'])
            if flag_path and os.path.exists(flag_path):
                try:
                    flag_img = Image.open(flag_path).convert("RGBA")
                    margin = 2
                    target_width = max(flash_width - 2 * margin, 1)
                    target_height = max((IMAGE_HEIGHT // 2) - 2 * margin, 1)
                    aspect_ratio = flag_img.width / flag_img.height
                    new_height = target_height
                    new_width = int(new_height * aspect_ratio)
                    if new_width > target_width:
                        new_width = target_width
                        new_height = int(new_width / aspect_ratio)
                    # Scale down the width by 75%
                    width_scale = 0.75
                    new_width = int(new_width * width_scale)
                    new_height = int(new_width / aspect_ratio)
    
                    flag_img_resized = flag_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                    flag_draw_width = new_width
                    flag_draw_height = new_height
                except Exception as e:
                    print("Error loading flag image:", e)
        
        # Prepare country text details if available
        country_text = None
        country_text_width = 0
        country_text_height = 0
        country_font_chosen = None
        if plate['country'] != "NONE":
            country_text = plate['country']
            max_text_width = flash_width - 4  # 2px margin on each side
            country_font_size = get_country_font_size(draw, font_path, country_text, max_text_width)
            country_font_chosen = ImageFont.truetype(font_path, country_font_size)
            bbox = draw.textbbox((0, 0), country_text, font=country_font_chosen)
            country_text_width = bbox[2] - bbox[0]
            country_text_height = bbox[3] - bbox[1]
        
        # Calculate the total group height and starting y for the group (centered within flash area)
        spacing = 8  # spacing between flag and text if both exist
        group_total_height = 0
        if flag_img_resized is not None and country_text is not None:
            group_total_height = flag_draw_height + spacing + country_text_height
        elif flag_img_resized is not None:
            group_total_height = flag_draw_height
        elif country_text is not None:
            group_total_height = country_text_height
        
        if flag_img_resized is None and country_text is not None:
            # When only country text is present, offset the text 15px down from center.
            group_start_y = (IMAGE_HEIGHT - group_total_height) // 2 + 15
        else:
            group_start_y = (IMAGE_HEIGHT - group_total_height) // 2
        
        # Draw flag image if present
        if flag_img_resized is not None:
            flag_x = (flash_width - flag_draw_width) // 2
            flag_y = group_start_y
            img.paste(flag_img_resized, (flag_x, flag_y), flag_img_resized)
        
        # Draw country text if present
        if country_text is not None:
            country_x = (flash_width - country_text_width) // 2
            if flag_img_resized is not None:
                country_y = group_start_y + flag_draw_height + spacing
            else:
                country_y = group_start_y
            draw.text((country_x, country_y), country_text, fill="black", font=country_font_chosen)
    
    # Compute the total width of the VRN text according to the drawing rules.
    def compute_vrn_text_width(vrn):
        if " " in vrn:
            groups = vrn.split(" ")
            group1 = groups[0]
            group2 = groups[1] if len(groups) > 1 else ""
        else:
            group1 = vrn[:4]
            group2 = vrn[4:]
        total_width = 0
        # Process the first group
        for ch in group1:
            if ch in ('i', 'I', '1'):
                w = int(round(draw.textlength(ch, font=main_font)))
            else:
                w = CHAR_WIDTH
            total_width += w + CHAR_SPACING
        # Process the second group, if any, with adjusted spacing.
        if group2:
            total_width += (GROUP_SPACING - CHAR_SPACING)
            for ch in group2:
                if ch in ('i', 'I', '1'):
                    w = int(round(draw.textlength(ch, font=main_font)))
                else:
                    w = CHAR_WIDTH
                total_width += w + CHAR_SPACING
        # Remove trailing spacing added after the last character.
        if total_width > 0:
            total_width -= CHAR_SPACING
        return total_width

    # Calculate the available width (excluding the flash strip) and center the VRN text.
    vrn_text = plate['VRN']
    available_width = IMAGE_WIDTH - flash_width
    text_width = compute_vrn_text_width(vrn_text)
    text_start_x = flash_width + (available_width - text_width) // 2
    curr_x = text_start_x

    # Vertical positioning for VRN text (centered vertically)
    TEXT_Y = (IMAGE_HEIGHT - CHAR_HEIGHT) // 2

    # Split VRN into two groups
    if " " in vrn_text:
        groups = vrn_text.split(" ")
        group1 = groups[0]
        group2 = groups[1] if len(groups) > 1 else ""
    else:
        group1 = vrn_text[:4]
        group2 = vrn_text[4:]

    # Draw the first group using fixed or natural width as needed.
    for ch in group1:
        if ch in ('i', 'I', '1'):
            actual_char_width = int(round(draw.textlength(ch, font=main_font)))
            draw.text((curr_x, TEXT_Y), ch, fill="black", font=main_font)
            curr_x += actual_char_width + CHAR_SPACING
        else:
            actual_char_width = int(round(draw.textlength(ch, font=main_font)))
            offset_x = (CHAR_WIDTH - actual_char_width) // 2
            draw.text((curr_x + offset_x, TEXT_Y), ch, fill="black", font=main_font)
            curr_x += CHAR_WIDTH + CHAR_SPACING

    # Add group spacing between groups.
    curr_x += (GROUP_SPACING - CHAR_SPACING)

    # Draw the second group.
    for ch in group2:
        if ch in ('i', 'I', '1'):
            actual_char_width = int(round(draw.textlength(ch, font=main_font)))
            draw.text((curr_x, TEXT_Y), ch, fill="black", font=main_font)
            curr_x += actual_char_width + CHAR_SPACING
        else:
            actual_char_width = int(round(draw.textlength(ch, font=main_font)))
            offset_x = (CHAR_WIDTH - actual_char_width) // 2
            draw.text((curr_x + offset_x, TEXT_Y), ch, fill="black", font=main_font)
            curr_x += CHAR_WIDTH + CHAR_SPACING

    # Draw a 2mm black border (converted to pixels)
    border_thickness = int(round(2 * scale))
    draw.rectangle(
         [0, 0, IMAGE_WIDTH - 1, IMAGE_HEIGHT - 1],
         outline="black",
         width=border_thickness
    )

    return img

# ---------------------------
# Main routine to generate and render plates.
# ---------------------------
def main():
    # Get the number of plates to generate. Default is 1.
    num_plates = 1
    if len(sys.argv) > 1:
        try:
            num_plates = int(sys.argv[1])
        except ValueError:
            print("Invalid number provided, using default (5).")
    
    # Generate VRN dictionaries using our VRN generator module.
    plates = generate_plates(num_plates)
    
    # Create and/or ensure output directory exists.
    output_dir = "generated"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load fonts. We try to load the specified typeface ("CharlesWright2001.ttf").
    # If not available, we fallback to the default PIL font.
    font_path = "media/CharlesWright-Bold.otf"
    try:
        # We'll pass the font path instead of the loaded font
        main_font = font_path
    except Exception as e:
        print("Could not access 'CharlesWright-Bold.otf'. Using default font.", e)
        main_font = ImageFont.load_default()
    # A smaller font for drawing the country identifier in the flash panel.
    try:
        small_font = ImageFont.truetype("media/CharlesWright-Bold.otf", 20)
    except Exception:
        small_font = ImageFont.load_default()

    for plate in plates:
        # Clean the VRN for file naming (remove any spaces)
        vrn_clean = plate['VRN'].replace(" ", "")
        
        # Decide front and rear background colours.
        front_bg = "#FFFFFF"  # pure white
        
        # Rear plate: choose one of two yellow shades (50-50 weighted).
        rear_bg = random.choice(["#FFFF00", "#FEDD00"])
        
        # Determine whether there is a flash area.
        flash_present = (plate['flash'] != "NONE" or plate['flag'] != "NONE" or plate['country'] != "NONE")
        flash_width = get_flash_width() if flash_present else 0

        # Render the front plate.
        front_plate_img = render_plate(plate, front_bg, flash_width, main_font, small_font)
        front_filename = os.path.join(output_dir, f"{vrn_clean}_f.jpg")
        front_plate_img.save(front_filename, "JPEG")
        
        # Render the rear plate.
        rear_plate_img = render_plate(plate, rear_bg, flash_width, main_font, small_font)
        rear_filename = os.path.join(output_dir, f"{vrn_clean}_r.jpg")
        rear_plate_img.save(rear_filename, "JPEG")
        
        print(f"Generated plate {plate['VRN']}: {front_filename} and {rear_filename}")
    

if __name__ == '__main__':
    main() 