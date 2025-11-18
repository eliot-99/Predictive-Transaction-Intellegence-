try:
    from PIL import Image, ImageDraw
    
    png_path = r'WebApp\static\apple-touch-icon.png'
    ico_path = r'WebApp\static\favicon.ico'
    
    img = Image.new('RGB', (256, 256), color='white')
    draw = ImageDraw.Draw(img)
    
    draw.polygon([
        (128 - 50, 40),
        (128 + 50, 55),
        (128 + 50, 110),
        (128, 155),
        (128 - 50, 110),
    ], fill=(0, 48, 135), outline=(0, 30, 80), width=3)
    
    draw.text((128, 105), 'F', fill='white', anchor='mm')
    
    img.save(png_path)
    
    img_ico = img.resize((64, 64), Image.Resampling.LANCZOS)
    img_ico.save(ico_path)
    
    print(f"Created {ico_path}")
    print(f"Created {png_path}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
