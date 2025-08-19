import os
import json
import shutil
from flask import Flask, render_template, request, jsonify
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

app = Flask(__name__)

BASE_DIR = 'data'
STATIC_DATA_DIR = 'static/data'

# --- بارگذاری مدل هوش مصنوعی در زمان اجرای برنامه ---
# این کار فقط یک بار انجام می‌شود تا سرعت پردازش بالا بماند
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

# --- سایر توابع (find_files, copy_new_files_to_static و ...) بدون تغییر ---
def find_files():
    data_without_caption = []
    data_with_caption = []

    for root, dirs, files in os.walk(BASE_DIR):
        for file in files:
            if file.endswith('.json'):
                json_path = os.path.join(root, file)
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if all(key in data for key in ['frame1_image', 'frame2_image']):
                            # , 'source_points', 'target_points'
                            data['frame1_image'] = os.path.relpath(os.path.join(root, data['frame1_image']), start=BASE_DIR).replace('\\', '/')
                            data['frame2_image'] = os.path.relpath(os.path.join(root, data['frame2_image']), start=BASE_DIR).replace('\\', '/')
                            data['json_file_path'] = json_path
                            data['directory_name'] = os.path.basename(root)

                            if data.get('caption'):
                                data_with_caption.append(data)
                            else:
                                data_without_caption.append(data)
                except (json.JSONDecodeError, FileNotFoundError):
                    continue
    return data_without_caption, data_with_caption

def copy_new_files_to_static():
    os.makedirs(STATIC_DATA_DIR, exist_ok=True)
    
    for root, dirs, files in os.walk(BASE_DIR):
        relative_path = os.path.relpath(root, BASE_DIR)
        destination_dir = os.path.join(STATIC_DATA_DIR, relative_path)
        
        if files:
            os.makedirs(destination_dir, exist_ok=True)
            for file in files:
                source_file_path = os.path.join(root, file)
                destination_file_path = os.path.join(destination_dir, file)
                
                if not os.path.exists(destination_file_path):
                    shutil.copy2(source_file_path, destination_file_path)
                    

def remove_empty_dirs():
    root_dir = BASE_DIR
    for dirpath, dirnames, filenames in os.walk(root_dir, topdown=False):
        if not dirnames and not filenames:
            print(f"Removing empty directory: {dirpath}")
            os.rmdir(dirpath)


# --- Endpoint جدید برای توصیف تصاویر ---
@app.route('/describe_images', methods=['POST'])
def describe_images():
    data = request.get_json()
    image1_path = data.get('image1_path')
    image2_path = data.get('image2_path')

    full_image1_path = os.path.join(STATIC_DATA_DIR, image1_path)
    full_image2_path = os.path.join(STATIC_DATA_DIR, image2_path)

    try:
        raw_image1 = Image.open(full_image1_path).convert('RGB')
        raw_image2 = Image.open(full_image2_path).convert('RGB')

        # توصیف تصویر اول
        inputs1 = processor(raw_image1, return_tensors="pt")
        out1 = model.generate(**inputs1)
        description1 = processor.decode(out1[0], skip_special_tokens=True)

        # توصیف تصویر دوم
        inputs2 = processor(raw_image2, return_tensors="pt")
        out2 = model.generate(**inputs2)
        description2 = processor.decode(out2[0], skip_special_tokens=True)

        return jsonify({
            'success': True,
            'description1': description1,
            'description2': description2,
            'combined_description': f"تصویر اول: {description1}. تصویر دوم: {description2}"
        })

    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

def find_no_json_images():
    no_json_images = []
    for root, dirs, files in os.walk(BASE_DIR):
        has_json = any(f.endswith('.json') for f in files)
        if not has_json:
            image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif'))]
            if image_files:
                images = [os.path.relpath(os.path.join(root, f), start=BASE_DIR).replace('\\', '/') for f in image_files]
                no_json_images.append({
                    'directory_name': os.path.basename(root),
                    'images': images
                })
    return no_json_images

# --- سایر Route ها (index, with_caption, no_json_images) بدون تغییر ---
@app.route('/')
def index():
    data_without_caption, data_with_caption = find_files()
    no_json_images = find_no_json_images()
    return render_template('index.html', 
                           files=data_without_caption, 
                           no_caption_count=len(data_without_caption),
                           has_caption_count=len(data_with_caption), 
                           no_json_count=len(no_json_images))
    
@app.route('/no_json_images')
def no_json_images():
    data_without_caption, data_with_caption = find_files()
    no_json_images_data = find_no_json_images()
    return render_template('no_json_images.html', 
                           dirs=no_json_images_data,
                           no_caption_count=len(data_without_caption),
                           has_caption_count=len(data_with_caption), 
                           no_json_count=len(no_json_images_data))

@app.route('/with_caption')
def with_caption():
    data_without_caption, data_with_caption = find_files()
    no_json_images = find_no_json_images()
    return render_template('with_caption.html', 
                           files=data_with_caption,
                           no_caption_count=len(data_without_caption),
                           has_caption_count=len(data_with_caption), 
                           no_json_count=len(no_json_images))

@app.route('/update_caption', methods=['POST'])
def update_caption():
    data = request.get_json()
    json_file_path = data.get('json_file_path')
    caption = data.get('caption')

    if not json_file_path or caption is None:
        return jsonify({'success': False, 'message': 'Invalid data'}), 400

    try:
        if not os.path.abspath(json_file_path).startswith(os.path.abspath(BASE_DIR)):
            return jsonify({'success': False, 'message': 'Access denied'}), 403

        with open(json_file_path, 'r+', encoding='utf-8') as f:
            file_data = json.load(f)
            file_data['caption'] = caption
            f.seek(0)
            json.dump(file_data, f, indent=4, ensure_ascii=False)
            f.truncate()
        return jsonify({'success': True, 'message': 'Caption updated successfully'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

if __name__ == '__main__':
    copy_new_files_to_static()
    remove_empty_dirs()
    app.run(debug=True)