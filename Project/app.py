from flask import Flask, render_template, request, redirect, url_for
import os
import re
from dummy_model import classify_clothes, detect_color, make_model, recommend_cody
import weather_fetch # main 함수 실행시 weather csv 파일 생성 및, return csv 파일명

from ultralytics import YOLO
app = Flask(__name__)

PAGE_MAP = {
    '1/2 pants':    'bottom',
    'pants':        'bottom',
    'skirt':        'bottom',
    '1/2 shirts':   'top',
    'shirts':       'top',
    'hoodie':       'top',
    'outer':        'outer',
    'dres shoes':   'shoe',
    'shoes':        'shoe',
    'slipper':      'shoe',
    'women shoes':  'shoe'
}

cls_model = YOLO("/Users/basamg/KW_2025/ML/fine_tune_yolo.pt")
recommend_model = make_model('/Users/basamg/KW_2025/ML/Project/cody_recommend(second_train).pth')

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# 페이지별 저장된 파일 리스트
category_items = {'top': [], 'bottom': [], 'outer': [], 'shoe': []}

clothes_list = []
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # 업로드 처리
        if 'file' in request.files and request.files['file'].filename:
            f = request.files['file']
            fp = os.path.join(app.config['UPLOAD_FOLDER'], f.filename)
            f.save(fp) 
            detections = classify_clothes(fp, cls_model)
            if not detections:
                return render_template('index.html', show_retry=True) 
            for idx, (crop, raw_category) in enumerate(detections):
                crop_filename = f"{os.path.splitext(f.filename)[0]}_{idx+1}{os.path.splitext(f.filename)[1]}"
                crop_path = os.path.join(app.config['UPLOAD_FOLDER'], crop_filename)
                crop.save(crop_path)    
                color = detect_color(crop)
                cloth = {'filepath':crop_path,'category':raw_category,'color':color}
                clothes_list.append(cloth)
                
                page = PAGE_MAP.get(raw_category)    
                if page in category_items:
                    category_items[page].append({"filename":crop_filename,"raw":raw_category,"color":color})
            
            
            first_cat = detections[0][1] #첫번째 인식에서의 카테고리로 이동
            first_page = PAGE_MAP.get(first_cat)            
            outfit = {
                "top" : "recommend Top . jpg",
                "bottom" : "reccomend Bottomg . jpg",
                "shoe" : "recommend shoe"
            }
            return render_template('index.html',redirect_to=first_page, outfit = outfit)
        # 수동 수정 처리
        elif 'category_override' in request.form:
            filename = request.form['filename']
            old_page = request.form['old_page']
            new_page = request.form['category_override']
            old_color = request.form['old_color']
            raw = request.form['raw']
            color = None
            if 'color_override' in request.form:
                color = request.form['color_override']
            # 기존 카테고리에서 제거
            category_items[old_page] = [
                it for it in category_items[old_page]
                if it['filename'] != filename
            ]
            # 새 카테고리에 추가
            raw = request.form.get('raw', filename.split('.')[0])  # 대체 로직
            new_item = {"filename": filename, "raw":raw,"color":old_color}
            if color:
                new_item['color'] = color
            category_items[new_page].append(new_item)
            # 수정 완료 팝업 & 새 페이지로 이동
            return redirect(url_for(new_page, corrected='1'))
        else:
            return redirect(url_for('index'))
    return render_template('index.html')

@app.route('/closet', methods=['GET','POST'])
def closet():
    return render_template('closet.html', clothes=category_items)

@app.route('/recommend')
def recommend():
    top = [it['filename'] for it in category_items['top']]
    bottom = [it['filename'] for it in category_items['bottom']]
    shoe = [it['filename'] for it in category_items['shoe']]
    outer = [it['filename'] for it in category_items['outer']]
    # ---------
    # 날씨 기반 옷 색깔 결정 후 해당 색에 알맞는 의류를 걸러내서 recommend_cody 에 input
    # top / bottom 재구성
    # ---------
    
    score, cody_top, cody_bottom, sorted_cody = recommend_cody(top,bottom,recommend_model)
    print(sorted_cody)
    if not top or not bottom:
        return render_template('recommend.html', top = None, bottom= None, shoe=None, outer=None)
    
    top_img = cody_top
    bottom_img = cody_bottom
    score = int(score*10)
    
    if shoe:
        shoe_img = shoe[0]    
        if outer:
            outer_img = outer[0] # 만약 날씨가 춥다면 outer 까지 출력
            return render_template('recommend.html', top=top_img, bottom= bottom_img, shoe=shoe_img, outer = outer_img, score = score)
        else:
            return render_template('recommend.html', top=top_img, bottom= bottom_img, shoe=shoe_img, outer = None, score =score)
    else:
        return render_template('recommend.html', top=top_img, bottom= bottom_img, shoe=None, outer = None, score=score)
@app.route('/top', methods=['GET','POST'])
def top():
    if request.method == "POST" and 'delete_filename' in request.form:
        filename = request.form['delete_filename']
        category = request.form['delete_category'].strip()
        category_items[category] = [
            it for it in category_items[category] 
            if it['filename'] != filename
            ]
        print(category)
        return redirect(url_for(('top')))
    return render_template('cloth/top.html',items=category_items['top'])
@app.route('/bottom', methods=['GET','POST'])
def bottom():
    if request.method == "POST" and 'delete_filename' in request.form:
        filename = request.form['delete_filename']
        category = request.form['delete_category'].strip()
        category_items[category] = [
            it for it in category_items[category] 
            if it['filename'] != filename
            ]
        print(category)
        return redirect(url_for(('bottom')))
    return render_template('cloth/bottom.html',items=category_items['bottom'])
@app.route('/outer', methods=['GET','POST'])
def outer():
    if request.method == "POST" and 'delete_filename' in request.form:
        filename = request.form['delete_filename']
        category = request.form['delete_category'].strip()
        category_items[category] = [
            it for it in category_items[category] 
            if it['filename'] != filename
            ]
        print(category)
        return redirect(url_for(('outer')))
    return render_template('cloth/outer.html',items=category_items['outer'])
@app.route('/shoe', methods=['GET','POST'])
def shoe():
    if request.method == "POST" and 'delete_filename' in request.form:
        filename = request.form['delete_filename']
        category = request.form['delete_category'].strip()
        category_items[category] = [
            it for it in category_items[category] 
            if it['filename'] != filename
            ]
        print(category)
        return redirect(url_for(('shoe')))
    return render_template('cloth/shoe.html',items=category_items['shoe'])
if __name__ == "__main__":
    app.run(debug=True)

