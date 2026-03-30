#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PCB Defect Detection API
RESTful API للكشف عن عيوب لوحات الدوائر المطبوعة باستخدام YOLOv8
بدون OpenCV - يعمل على Render و Hugging Face Spaces
"""

import os
import sys
import base64
import numpy as np
from datetime import datetime
from typing import List, Optional
import io
from PIL import Image, ImageDraw, ImageFont
import uvicorn

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from ultralytics import YOLO

# ============================================
# تحميل النموذج
# ============================================

MODEL_PATH = "last.pt"

if not os.path.exists(MODEL_PATH):
    print(f"⚠️ النموذج غير موجود في {MODEL_PATH}")
    print("يرجى رفع ملف last.pt إلى المجلد")
    sys.exit(1)

model = YOLO(MODEL_PATH)
print("✅ تم تحميل النموذج بنجاح")

# ============================================
# تعريف الفئات
# ============================================

CLASS_NAMES_EN = {
    0: "missing_hole",
    1: "mouse_bite",
    2: "open_circuit",
    3: "short_circuit",
    4: "spur",
    5: "spurious_copper"
}

CLASS_NAMES_AR = {
    "missing_hole": "ثقب مفقود",
    "mouse_bite": "عضّة فأر",
    "open_circuit": "دارة مفتوحة",
    "short_circuit": "دارة قصيرة",
    "spur": "نتوء نحاسي",
    "spurious_copper": "نحاس زائد"
}

# ============================================
# نماذج البيانات (Pydantic)
# ============================================

class Detection(BaseModel):
    """معلومات عيب واحد"""
    defect_type: str = Field(..., description="نوع العيب (بالإنجليزية)")
    defect_type_ar: str = Field(..., description="نوع العيب (بالعربية)")
    confidence: float = Field(..., description="نسبة الثقة (0-1)", ge=0, le=1)
    bbox: List[int] = Field(..., description="مربع الإحاطة [x1, y1, x2, y2]")

class DetectResponse(BaseModel):
    """استجابة API للكشف"""
    success: bool = Field(..., description="نجاح العملية")
    num_defects: int = Field(..., description="عدد العيوب المكتشفة")
    detections: List[Detection] = Field(..., description="قائمة العيوب")
    annotated_image_base64: Optional[str] = Field(None, description="الصورة مع المربعات (Base64)")
    processing_time_ms: float = Field(..., description="زمن المعالجة بالمللي ثانية")
    timestamp: str = Field(..., description="وقت المعالجة")
    parameters_used: dict = Field(..., description="البارامترات المستخدمة في الكشف")

class HealthResponse(BaseModel):
    """استجابة فحص الصحة"""
    status: str
    model_loaded: bool
    model_path: str
    timestamp: str

# ============================================
# دوال مساعدة (بدون OpenCV)
# ============================================

def image_from_bytes(contents: bytes) -> np.ndarray:
    """تحويل bytes إلى numpy array باستخدام PIL"""
    img = Image.open(io.BytesIO(contents))
    return np.array(img)

def image_to_base64(image_array: np.ndarray) -> str:
    """تحويل numpy array إلى Base64"""
    img = Image.fromarray(image_array)
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG', quality=95)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

def draw_defects_with_language(image_array: np.ndarray, results, lang: str = "en") -> np.ndarray:
    """
    رسم العيوب على الصورة مع تسميات باللغة المحددة
    باستخدام PIL بدلاً من OpenCV
    """
    # تحويل إلى PIL Image
    img = Image.fromarray(image_array)
    draw = ImageDraw.Draw(img)
    
    # محاولة تحميل خط عربي (إذا لم يوجد، يستخدم الخط الافتراضي)
    try:
        # محاولة استخدام خط يدعم العربية
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except:
        font = ImageFont.load_default()
    
    for r in results:
        if r.boxes is not None:
            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                class_name_en = CLASS_NAMES_EN.get(cls, "unknown")
                
                # اختيار اللغة
                if lang == "ar":
                    label = f"{CLASS_NAMES_AR.get(class_name_en, class_name_en)} {conf:.2f}"
                else:
                    label = f"{class_name_en} {conf:.2f}"
                
                # إحداثيات المربع
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                
                # رسم المربع
                draw.rectangle([x1, y1, x2, y2], outline="green", width=3)
                
                # رسم النص (مع خلفية شفافة)
                bbox_text = draw.textbbox((x1, y1 - 20), label, font=font)
                draw.rectangle(bbox_text, fill="green")
                draw.text((x1, y1 - 20), label, fill="white", font=font)
    
    return np.array(img)

# ============================================
# إنشاء تطبيق FastAPI
# ============================================

app = FastAPI(
    title="PCB Defect Detection API",
    description="""
    # 🖥️ نظام كشف عيوب لوحات الدوائر المطبوعة (PCB)
    
    هذا API يوفر كشفاً تلقائياً لـ 6 أنواع من العيوب في PCB:
    - **missing_hole**: ثقب مفقود
    - **mouse_bite**: عضّة فأر
    - **open_circuit**: دارة مفتوحة
    - **short_circuit**: دارة قصيرة
    - **spur**: نتوء نحاسي
    - **spurious_copper**: نحاس زائد
    
    ## 📌 كيفية الاستخدام
    
    1. استخدم endpoint `/detect` مع رفع صورة PCB
    2. يمكنك ضبط البارامترات المختلفة للتحكم في دقة الكشف
    3. استخدم `/docs` لاستعراض وثائق API وتجربته مباشرة
    
    ## 📊 أداء النموذج
    
    | المقياس | القيمة |
    |---------|--------|
    | mAP50 | 99.01% |
    | Precision | 98.05% |
    | Recall | 98.46% |
    """,
    version="2.0.0",
    contact={
        "name": "PCB Defect Detection Team",
    },
    license_info={
        "name": "MIT",
    }
)

# إضافة CORS للسماح بالاتصال من أي نطاق
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# نقاط النهاية (Endpoints)
# ============================================

@app.get("/", response_model=dict, tags=["General"])
async def root():
    """المعلومات الأساسية عن API"""
    return {
        "service": "PCB Defect Detection API",
        "version": "2.0.0",
        "model": "YOLOv8n",
        "status": "online",
        "docs": "/docs",
        "redoc": "/redoc",
        "health": "/health",
        "detect": "POST /detect"
    }

@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """فحص صحة الخدمة والنموذج"""
    return HealthResponse(
        status="healthy",
        model_loaded=True,
        model_path=MODEL_PATH,
        timestamp=datetime.now().isoformat()
    )

@app.post(
    "/detect",
    response_model=DetectResponse,
    summary="كشف العيوب في صورة PCB",
    description="""
    ## كشف العيوب في صورة PCB
    
    يرسل صورة PCB ويستعيد:
    - قائمة العيوب المكتشفة مع مواقعها ونسبة الثقة
    - الصورة مع المربعات المرسومة حول العيوب (بتنسيق Base64)
    
    ### البارامترات المتاحة:
    
    | البارامتر | الوصف | النطاق | الافتراضي |
    |-----------|-------|--------|-----------|
    | `file` | صورة PCB (jpg, png, jpeg) | - | مطلوب |
    | `conf_threshold` | عتبة الثقة | 0.1 - 0.9 | 0.5 |
    | `iou_threshold` | عتبة IoU | 0.1 - 0.9 | 0.45 |
    | `max_det` | الحد الأقصى للعيوب | 1 - 1000 | 300 |
    | `imgsz` | حجم الصورة المدخلة | 320 - 1280 | 640 |
    | `classes_filter` | تصفية الفئات (مثال: "0,2,4") | 0-5 | الكل |
    | `agnostic_nms` | NMS بغض النظر عن الفئة | true/false | false |
    | `half_precision` | استخدام دقة float16 | true/false | false |
    | `return_annotated` | إرجاع الصورة مع المربعات | true/false | true |
    | `lang` | اللغة (ar/en) للتسميات على الصورة | ar/en | en |
    """,
    tags=["Detection"]
)
async def detect_defects(
    file: UploadFile = File(..., description="صورة PCB (jpg, png, jpeg)"),
    conf_threshold: float = Query(0.5, ge=0.1, le=0.9, description="عتبة الثقة"),
    iou_threshold: float = Query(0.45, ge=0.1, le=0.9, description="عتبة IoU - إزالة المربعات المتداخلة"),
    max_det: int = Query(300, ge=1, le=1000, description="الحد الأقصى للعيوب"),
    imgsz: int = Query(640, ge=320, le=1280, description="حجم الصورة المدخلة"),
    classes_filter: Optional[str] = Query(None, description="تصفية الفئات (مثال: '0,2,4')"),
    agnostic_nms: bool = Query(False, description="NMS بغض النظر عن الفئة"),
    half_precision: bool = Query(False, description="استخدام دقة float16"),
    return_annotated: bool = Query(True, description="إرجاع الصورة مع المربعات (Base64)"),
    lang: str = Query("en", description="لغة التسميات على الصورة (ar/en)", pattern="^(ar|en)$")
):
    """
    كشف العيوب في صورة PCB مع إمكانية إرجاع الصورة مع المربعات
    """
    import time
    start_time = time.time()
    
    # التحقق من نوع الملف
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="الملف المرفق ليس صورة")
    
    try:
        # قراءة الصورة (بدون OpenCV)
        contents = await file.read()
        image_array = image_from_bytes(contents)
        
        if image_array is None or image_array.size == 0:
            raise HTTPException(status_code=400, detail="لم يتمكن النظام من قراءة الصورة")
        
        # تحويل classes_filter إلى قائمة أرقام
        classes_list = None
        if classes_filter:
            try:
                classes_list = [int(c.strip()) for c in classes_filter.split(",")]
                for c in classes_list:
                    if c not in CLASS_NAMES_EN:
                        raise HTTPException(status_code=400, detail=f"رقم الفئة {c} غير صالح")
            except ValueError:
                raise HTTPException(status_code=400, detail="classes_filter يجب أن يكون أرقاماً مفصولة بفواصل")
        
        # إعداد بارامترات التنبؤ
        predict_kwargs = {
            "conf": conf_threshold,
            "iou": iou_threshold,
            "max_det": max_det,
            "imgsz": imgsz,
            "verbose": False,
            "agnostic_nms": agnostic_nms,
            "half": half_precision
        }
        
        if classes_list:
            predict_kwargs["classes"] = classes_list
        
        # التنبؤ
        results = model.predict(image_array, **predict_kwargs)
        
        # استخراج النتائج
        detections = []
        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    class_name_en = CLASS_NAMES_EN.get(cls, "unknown")
                    class_name_ar = CLASS_NAMES_AR.get(class_name_en, class_name_en)
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    
                    detections.append(Detection(
                        defect_type=class_name_en,
                        defect_type_ar=class_name_ar,
                        confidence=conf,
                        bbox=[int(x1), int(y1), int(x2), int(y2)]
                    ))
        
        # رسم الصورة مع المربعات إذا طلب المستخدم
        annotated_image_base64 = None
        if return_annotated and results:
            annotated_img = draw_defects_with_language(image_array, results, lang)
            annotated_image_base64 = image_to_base64(annotated_img)
        
        processing_time = (time.time() - start_time) * 1000
        
        # تسجيل البارامترات المستخدمة
        parameters_used = {
            "conf_threshold": conf_threshold,
            "iou_threshold": iou_threshold,
            "max_det": max_det,
            "imgsz": imgsz,
            "classes_filter": classes_filter if classes_filter else "all",
            "agnostic_nms": agnostic_nms,
            "half_precision": half_precision,
            "return_annotated": return_annotated,
            "lang": lang
        }
        
        return DetectResponse(
            success=True,
            num_defects=len(detections),
            detections=detections,
            annotated_image_base64=annotated_image_base64,
            processing_time_ms=round(processing_time, 2),
            timestamp=datetime.now().isoformat(),
            parameters_used=parameters_used
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"خطأ في المعالجة: {str(e)}")

# ============================================
# واجهة HTML بسيطة للاختبار
# ============================================

@app.get("/test", response_class=HTMLResponse, tags=["General"], include_in_schema=False)
async def test_ui():
    """واجهة بسيطة لاختبار API"""
    return """
    <!DOCTYPE html>
    <html lang="ar" dir="rtl">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>PCB Defect Detection API - اختبار</title>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
                margin: 0;
            }
            .container {
                max-width: 1000px;
                margin: 0 auto;
                background: white;
                border-radius: 20px;
                padding: 30px;
                box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            }
            h1, h2 { color: #333; text-align: center; }
            .upload-area {
                border: 2px dashed #ccc;
                border-radius: 15px;
                padding: 40px;
                text-align: center;
                margin: 20px 0;
                cursor: pointer;
                transition: all 0.3s ease;
            }
            .upload-area:hover, .upload-area.dragover {
                border-color: #667eea;
                background: #f5f3ff;
            }
            .btn {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                padding: 12px 30px;
                border-radius: 25px;
                cursor: pointer;
                font-size: 16px;
                margin: 10px;
            }
            .result {
                margin-top: 20px;
                padding: 15px;
                background: #f8f9fa;
                border-radius: 10px;
            }
            pre {
                white-space: pre-wrap;
                word-wrap: break-word;
                background: #f1f1f1;
                padding: 10px;
                border-radius: 5px;
                max-height: 300px;
                overflow: auto;
            }
            .badge {
                display: inline-block;
                padding: 5px 12px;
                border-radius: 20px;
                margin: 5px;
                font-size: 12px;
            }
            .badge-short_circuit { background: #fee2e2; color: #991b1b; }
            .badge-open_circuit { background: #fed7aa; color: #9b2c1d; }
            .badge-missing_hole { background: #fef3c7; color: #92400e; }
            .badge-mouse_bite { background: #d1fae5; color: #065f46; }
            .badge-spur { background: #e0e7ff; color: #1e40af; }
            .badge-spurious_copper { background: #fce7f3; color: #9d174d; }
            .params-panel {
                background: #f0f0f0;
                padding: 15px;
                border-radius: 10px;
                margin: 15px 0;
            }
            .params-panel label {
                display: inline-block;
                width: 150px;
                margin: 5px;
            }
            .params-panel input, .params-panel select {
                padding: 5px;
                margin: 5px;
                border-radius: 5px;
                border: 1px solid #ccc;
            }
            .docs-link { text-align: center; margin-top: 20px; }
            .docs-link a { color: #667eea; text-decoration: none; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🖥️ نظام كشف عيوب PCB</h1>
            <p style="text-align: center;">رفع صورة لاكتشاف العيوب مع إمكانية ضبط البارامترات</p>
            
            <div class="upload-area" id="uploadBox">
                <div style="font-size: 48px;">📸</div>
                <p>اسحب الصورة هنا أو انقر للاختيار</p>
                <input type="file" id="fileInput" accept="image/*" style="display: none;">
                <button class="btn" id="uploadBtn">اختر صورة</button>
            </div>
            
            <div class="params-panel">
                <h3>⚙️ بارامترات الكشف</h3>
                <label>عتبة الثقة (0.1-0.9):</label>
                <input type="range" id="conf" min="0.1" max="0.9" step="0.05" value="0.5">
                <span id="confValue">0.5</span><br>
                
                <label>عتبة IoU (0.1-0.9):</label>
                <input type="range" id="iou" min="0.1" max="0.9" step="0.05" value="0.45">
                <span id="iouValue">0.45</span><br>
                
                <label>الحد الأقصى للعيوب:</label>
                <input type="number" id="maxDet" value="300" min="1" max="1000"><br>
                
                <label>حجم الصورة:</label>
                <select id="imgsz">
                    <option value="320">320x320 (أسرع)</option>
                    <option value="640" selected>640x640 (افتراضي)</option>
                    <option value="1280">1280x1280 (أدق)</option>
                </select><br>
                
                <label>لغة التسميات:</label>
                <select id="langSelect">
                    <option value="en">English</option>
                    <option value="ar">العربية</option>
                </select><br>
                
                <label>تصفية الفئات:</label>
                <input type="text" id="classesFilter" placeholder="مثال: 0,2,4" style="width: 200px;">
                <small>(اتركها فارغة للكل)</small><br>
                
                <label>NMS بغض النظر عن الفئة:</label>
                <input type="checkbox" id="agnosticNms"><br>
                
                <label>نصف الدقة (أسرع):</label>
                <input type="checkbox" id="halfPrecision">
            </div>
            
            <div id="result" class="result" style="display: none;"></div>
            <div id="loading" style="display: none; text-align: center;">
                <div style="width: 40px; height: 40px; border: 4px solid #f3f3f3; border-top: 4px solid #667eea; border-radius: 50%; animation: spin 1s linear infinite; margin: 0 auto;"></div>
                <p>جاري تحليل الصورة...</p>
            </div>
            
            <div class="docs-link">
                <hr>
                <p>📚 <a href="/docs" target="_blank">استعراض وثائق API (Swagger UI)</a></p>
                <p>📖 <a href="/redoc" target="_blank">استعراض الوثائق بتنسيق ReDoc</a></p>
            </div>
        </div>
        
        <style>
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
        </style>
        
        <script>
            const uploadBox = document.getElementById('uploadBox');
            const fileInput = document.getElementById('fileInput');
            const uploadBtn = document.getElementById('uploadBtn');
            const resultDiv = document.getElementById('result');
            const loadingDiv = document.getElementById('loading');
            
            document.getElementById('conf').oninput = () => {
                document.getElementById('confValue').innerText = document.getElementById('conf').value;
            };
            document.getElementById('iou').oninput = () => {
                document.getElementById('iouValue').innerText = document.getElementById('iou').value;
            };
            
            uploadBtn.onclick = () => fileInput.click();
            uploadBox.onclick = () => fileInput.click();
            
            uploadBox.ondragover = (e) => {
                e.preventDefault();
                uploadBox.classList.add('dragover');
            };
            
            uploadBox.ondragleave = () => uploadBox.classList.remove('dragover');
            
            uploadBox.ondrop = (e) => {
                e.preventDefault();
                uploadBox.classList.remove('dragover');
                const file = e.dataTransfer.files[0];
                if (file?.type.startsWith('image/')) processImage(file);
            };
            
            fileInput.onchange = (e) => {
                if (e.target.files[0]) processImage(e.target.files[0]);
            };
            
            async function processImage(file) {
                loadingDiv.style.display = 'block';
                resultDiv.style.display = 'none';
                
                const formData = new FormData();
                formData.append('file', file);
                formData.append('conf_threshold', document.getElementById('conf').value);
                formData.append('iou_threshold', document.getElementById('iou').value);
                formData.append('max_det', document.getElementById('maxDet').value);
                formData.append('imgsz', document.getElementById('imgsz').value);
                formData.append('lang', document.getElementById('langSelect').value);
                formData.append('classes_filter', document.getElementById('classesFilter').value);
                formData.append('agnostic_nms', document.getElementById('agnosticNms').checked);
                formData.append('half_precision', document.getElementById('halfPrecision').checked);
                formData.append('return_annotated', 'true');
                
                try {
                    const response = await fetch('/detect', { method: 'POST', body: formData });
                    const data = await response.json();
                    
                    loadingDiv.style.display = 'none';
                    resultDiv.style.display = 'block';
                    
                    let html = '<h3>📊 النتائج</h3>';
                    html += `<p><strong>🔍 تم اكتشاف ${data.num_defects} عيب/عيوب</strong></p>`;
                    html += `<p>⏱️ زمن المعالجة: ${data.processing_time_ms} مللي ثانية</p>`;
                    
                    if (data.annotated_image_base64) {
                        html += `<div style="margin: 15px 0; text-align: center;">
                                    <img src="data:image/jpeg;base64,${data.annotated_image_base64}" style="max-width: 100%; border-radius: 10px;">
                                 </div>`;
                    }
                    
                    if (data.detections.length > 0) {
                        html += '<div>';
                        data.detections.forEach(d => {
                            html += `<span class="badge badge-${d.defect_type}">${d.defect_type_ar} (${(d.confidence*100).toFixed(1)}%)</span>`;
                        });
                        html += '</div>';
                    } else {
                        html += '<p>✅ لم يتم اكتشاف أي عيوب</p>';
                    }
                    
                    html += '<details><summary>⚙️ البارامترات المستخدمة</summary>';
                    html += '<pre>' + JSON.stringify(data.parameters_used, null, 2) + '</pre>';
                    html += '</details>';
                    
                    resultDiv.innerHTML = html;
                    
                } catch (err) {
                    loadingDiv.style.display = 'none';
                    resultDiv.style.display = 'block';
                    resultDiv.innerHTML = `<p style="color: red;">❌ حدث خطأ: ${err.message}</p>`;
                }
            }
        </script>
    </body>
    </html>
    """

# ============================================
# تشغيل التطبيق
# ============================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    print(f"🚀 تشغيل PCB Defect Detection API")
    print(f"📍 Swagger UI: http://localhost:{port}/docs")
    print(f"📍 ReDoc: http://localhost:{port}/redoc")
    print(f"📍 صفحة الاختبار: http://localhost:{port}/test")
    uvicorn.run(app, host="0.0.0.0", port=port)
