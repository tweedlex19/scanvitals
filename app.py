import io
import os
import re

import cv2
import fitz  # PyMuPDF
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO

# ----------------------------------------------------------------------------
# Localization & Translations
# ----------------------------------------------------------------------------
LANG = {
    "tr": {
        "app_name": "ScanVitals",
        "subtitle": "Çok Modüllü Tıbbi Sinyal ve Görüntü Analiz Platformu",
        "mod_active": "Aktif Modül: EKG Ritim Analizi",
        "mod_future": "Ultrason • Tomografi (BT) • MR (Yakında)",
        "nav_center": "🫀 EKG Analiz Merkezi",
        "nav_legal": "⚖️ Kanuni & Hukuki Bilgilendirme",
        "welcome_title": "🚀 Nasıl Kullanılır?",
        "step1": "<b>1. Dosya Seç</b><br>JPG, PNG veya PDF raporunuzu yükleyin.",
        "step2": "<b>2. AI Analizi</b><br>Sistem şeritleri ayırıp ritimleri belirlesin.",
        "step3": "<b>3. Sonuçları İncele</b><br>Nabız ve ritim detaylarını görüntüleyin.",
        "ritms_title": "📚 Tespit Edilen Ritmler",
        "upload_lbl": "EKG dosyası yükle",
        "err_model": "Model dosyası bulunamadı",
        "info_wait": "Analize başlamak için lütfen yukarıdan bir dosya seçin.",
        "res_summary": "🎯 Genel Analiz Özeti",
        "qual_lbl": "Sinyal Kalitesi",
        "rhythm_lbl": "Genel Ritim",
        "bpm_lbl": "Ortalama Nabız",
        "scale_lbl": "Nabız Skalası (40-140 BPM)",
        "noise_warn": "💡 Sinyalde kısmi parazit saptandı. Toplam {} şeritten {} tanesi temiz kabul edilip analiz edildi.",
        "clean_success": "✨ Tüm şeritler yüksek kalitededir ve analiz edilebilir durumda.",
        "tab_strip": "📋 Şerit Bazlı Analiz",
        "tab_table": "📊 Özet Tablo & Rapor",
        "tab_tech": "🔍 Ham Görüntü & Teknik",
        "met_method": "Bölümleme yöntemi",
        "expand_noisy": " | Parazitli",
        "expand_normal": " | ~{} bpm ({})",
        "btn_sample": "Modelin İncelediği Atım Kesitini Gör (224x224)",
        "cap_sample": "Otomatik Hizalanan Örnek Atım",
        "prob_title": "Sınıf Benzerlikleri:",
        "summary_title": "Detaylı Karşılaştırma Tablosu",
        "btn_csv": "CSV Rapor İndir",
        "tech_title": "Teknik Bilgiler",
        "tech_meth": "- Bölümleme Metodu",
        "pool_title": "🧪 AI Geliştirme Katkısı",
        "pool_info": "Anonimleştirilmiş verileriniz, modellerimizin gelecekte daha doğru sonuçlar vermesini sağlar.",
        "pool_consent": "Bu kaydın anonim olarak V5 yapay zekâ geliştirme havuzuna eklenmesine izin veriyorum",
        "pool_select": "Gerçek / Onaylanan Durum (İsteğe bağlı düzeltme):",
        "pool_hint": "Sistemin tahmini otomatik seçilidir; saat/cihaz raporunda farklı bir sonuç yazıyorsa buradan düzelterek V5 eğitimine katkı sağlayabilirsiniz.",
        "btn_save": "Onayla ve Havuza Kaydet",
        "saved_msg": "Veri havuzuna başarıyla kaydedildi!",
        "pool_cnt": "📦 V5 Eğitim Havuzu: {} / 50 onaylı örnek",
        "legal_disclaimer": "🛡️ Tıbbi Sorumluluk Reddi",
        "legal_disclaimer_txt": "Bu platform kesinlikle bir tıbbi teşhis, tanı veya tedavi cihazı değildir. Sunulan analizler yalnızca araştırma, akademik eğitim ve ön bilgilendirme amacı taşımaktadır. Klinik kararlar sadece uzman hekimler tarafından verilmelidir.",
        "legal_kvkk": "🔐 KVKK ve Anonim Veri İşleme",
        "legal_kvkk_txt": "Yüklenen kayıtlar kullanıcının açık rızası olmaksızın sisteme kaydedilmez. Onay verildiğinde, kişisel kimlik bilgileri içermeden yalnızca yapay zekâ model iyileştirmesi amacıyla anonim işlenmektedir.",
        "legal_terms": "📜 Kullanım Koşulları ve Fikri Mülkiyet",
        "legal_terms_txt": "Sistem çıktılarının klinik karar yerine geçmeyeceği taahhüt edilir. Tüm fikri mülkiyet الحقوق ScanVitals şirketine aittir.",
        "footer": "© 2026 ScanVitals — Tüm hakları saklıdır. | Çok Modüllü Tıbbi Sinyal ve Görüntü Analiz Platformu",
        "med_warning": "ℹ️ Bu uygulama tıbbi teşhis aracı değildir; sonuçlar yalnızca bilgilendirme ve araştırma/deneme amaçlıdır.",
        "accept_model": "Model Tahmini Doğru (Değişiklik Yok)",
        "too_few": "Şeritler tespit edilemedi. Sadece genel görünüm sonucu geçerlidir."
    },
    "en": {
        "app_name": "ScanVitals",
        "subtitle": "Multi-Modular Medical Signal & Imaging Analysis Platform",
        "mod_active": "Active Module: ECG Rhythm Analysis",
        "mod_future": "Ultrasound • CT • MRI (Coming Soon)",
        "nav_center": "🫀 ECG Analysis Center",
        "nav_legal": "⚖️ Legal & Regulatory Information",
        "welcome_title": "🚀 How to Use?",
        "step1": "<b>1. Upload File</b><br>Upload JPG, PNG or PDF reports.",
        "step2": "<b>2. AI Analysis</b><br>System segments strips and detects rhythms.",
        "step3": "<b>3. Review Results</b><br>Check heart rate and rhythm details.",
        "ritms_title": "📚 Detected Rhythms",
        "upload_lbl": "Upload ECG File",
        "err_model": "Model file not found",
        "info_wait": "Please select a file above to begin analysis.",
        "res_summary": "🎯 General Analysis Summary",
        "qual_lbl": "Signal Quality",
        "rhythm_lbl": "General Rhythm",
        "bpm_lbl": "Average Heart Rate",
        "scale_lbl": "Heart Rate Scale (40-140 BPM)",
        "noise_warn": "💡 Partial noise detected. {} out of {} strips were analyzed as clean.",
        "clean_success": "✨ All strips are high quality and analyzable.",
        "tab_strip": "📋 Strip-based Analysis",
        "tab_table": "📊 Summary Table & Report",
        "tab_tech": "🔍 Raw View & Technical",
        "met_method": "Segmentation Method",
        "expand_noisy": " | Noisy",
        "expand_normal": " | ~{} bpm ({})",
        "btn_sample": "View Model Segment (224x224)",
        "cap_sample": "Automatically Aligned Beat Sample",
        "prob_title": "Class Probabilities:",
        "summary_title": "Detailed Comparison Table",
        "btn_csv": "Download CSV Report",
        "tech_title": "Technical Details",
        "tech_meth": "- Segmentation Method",
        "pool_title": "🧪 AI Development Contribution",
        "pool_info": "Your anonymized data helps our models achieve higher accuracy in future versions.",
        "pool_consent": "I consent to anonymously adding this record to the V5 development pool",
        "pool_select": "Actual / Verified Status (Optional correction):",
        "pool_hint": "The model's prediction is selected by default; if the device report says otherwise, you can correct it here to contribute to V5 training.",
        "btn_save": "Confirm and Save to Pool",
        "saved_msg": "Successfully saved to the data pool!",
        "pool_cnt": "📦 V5 Training Pool: {} / 50 verified samples",
        "legal_disclaimer": "🛡️ Medical Disclaimer",
        "legal_disclaimer_txt": "This platform is NOT a medical diagnostic or treatment device. Analyses are for research, education, and preliminary info only. Clinical decisions must be made solely by qualified physicians.",
        "legal_kvkk": "🔐 Privacy & Data Processing (GDPR/KVKK)",
        "legal_kvkk_txt": "Records are not stored without explicit consent. Upon consent, data is processed anonymously, stripped of PII, strictly for improving AI model robustness.",
        "legal_terms": "📜 Terms of Use & Intellectual Property",
        "legal_terms_txt": "Outputs shall not replace clinical judgment. All intellectual property rights belong to ScanVitals.",
        "footer": "© 2026 ScanVitals — All rights reserved. | Multi-Modular Medical Signal & Imaging Analysis Platform",
        "med_warning": "ℹ️ This application is not a medical diagnostic tool; results are for informational and research purposes only.",
        "accept_model": "Accept Model Prediction (No Change)",
        "too_few": "Strips not detected. Only general view result is applicable."
    }
}

def t(key):
    lang = st.session_state.get("lang", "tr")
    return LANG[lang].get(key, key)

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yolo11m-cls_best.pt")
WHITE_THRESHOLD = 245
MIN_ROW_DENSITY = 0.02
MERGE_GAP_RATIO = 0.004
MIN_BLOCK_RATIO = 0.03
PDF_ZOOM = 3.0

CLASS_DESCRIPTIONS = {
    "NORMAL": " Düzenli kalp atımı ve sağlıklı dalga formu.",
    "DUZENSİZ": " Atımlar arası süre değişkenliği mevcut.",
    "EFOR": " Hızlı nabız ve yüksek aktivite profili.",
    "ISKEMI": " Kan akışı yetersizliğini işaret eden formlar.",
    "GURULTULU": " Parazitli veri, güvenilir değil."
}


def get_class_description(label: str) -> str:
    key = label.strip().upper()
    if key in CLASS_DESCRIPTIONS:
        return CLASS_DESCRIPTIONS[key]
    for k, desc in CLASS_DESCRIPTIONS.items():
        if k in key:
            return desc
    return "Model tarafından tespit edilen ritim sınıfı."


def evaluate_bpm(bpm: int, is_noisy: bool = False) -> tuple[str, str]:
    """BPM değerini değerlendirip durum ve kısa yorum döndürür."""
    if is_noisy:
        return "Hesaplanmadı", "Sinyal parazitli olduğu için nabız hesabı devre dışı bırakıldı."
    if bpm <= 0:
        return "Ölçülemedi", "Bu bölümde net R-dalgası tepesi ayrılamadı."
    if bpm < 60:
        return "Düşük (Bradikardi eğilimi)", "Dakikadaki atım sayısı 60'ın altında görünüyor."
    if bpm <= 100:
        return "Normal aralık", "Dakikadaki atım sayısı standart dinlenim aralığında (60–100 bpm)."
    return "Yüksek (Taşikardi eğilimi)", "Dakikadaki atım sayısı 100'ün üzerinde görünüyor."


@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)


def is_valid_ecg_image(img_rgb: np.ndarray) -> tuple[bool, str]:
    """Görselin boş, tek renk veya kayıt dışı olup olmadığını kontrol eder."""
    if img_rgb is None or img_rgb.size == 0:
        return False, "Görsel verisi boş veya bozuk."

    h, w = img_rgb.shape[:2]
    if h < 40 or w < 40:
        return False, "Görsel çözünürlüğü analiz için çok küçük."

    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    if float(np.std(gray)) < 5.0:
        return False, "Görsel neredeyse tek renk veya boş; okunabilir EKG sinyali bulunamadı."

    if float(np.mean(gray < WHITE_THRESHOLD)) < 0.005:
        return False, "Sayfada yeterli sinyal veya çizgi içeriği bulunamadı."

    return True, ""


def classify(model, img_rgb: np.ndarray):
    result = model.predict(Image.fromarray(img_rgb), verbose=False)[0]
    probs = result.probs.data.cpu().numpy()
    names = result.names
    prob_dict = {names[i]: float(probs[i]) for i in range(len(probs))}
    top = int(np.argmax(probs))
    return names[top], float(probs[top]), prob_dict


def extract_pdf_reported_bpm(text: str) -> int | None:
    """PDF raporunun metninde yazılı olan cihaz nabzını yakalar."""
    patterns = [
        r"(?:ortalama\s*kalp\s*h[ıi]z[ıi]|heart\s*rate|kalp\s*h[ıi]z[ıi])[\s\S]{0,25}?(\d{2,3})\s*(?:bpm|at[ıi]m)",
        r"\b(\d{2,3})\s*(?:bpm|at[ıi]m/dk|at[ıi]m/dakika)\b",
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            val = int(m.group(1))
            if 35 <= val <= 220:
                return val
    return None


def pdf_to_images(pdf_bytes: bytes, zoom: float = PDF_ZOOM):
    pages = []
    reported_bpms = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        matrix = fitz.Matrix(zoom, zoom)
        for page in doc:
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            pages.append(img[:, :, :3].copy())
            reported_bpms.append(extract_pdf_reported_bpm(page.get_text("text")))
    return pages, reported_bpms


def load_uploaded_file(uploaded):
    data = uploaded.read()
    name = uploaded.name.lower()
    is_pdf = name.endswith(".pdf")
    if is_pdf:
        pages, reported_bpms = pdf_to_images(data)
        return pages, True, reported_bpms
    img = Image.open(io.BytesIO(data)).convert("RGB")
    return [np.array(img)], False, [None]


def extract_1d_signal_from_strip(strip_rgb: np.ndarray) -> np.ndarray:
    """Hem siyah çizgili (hastane EKG) hem de kırmızı çizgili (akıllı saat EKG) şeritlerden 1D sinyal çıkarır."""
    h, w = strip_rgb.shape[:2]
    if h < 10 or w < 10:
        return np.zeros(max(w, 10), dtype=np.float32)

    r = strip_rgb[:, :, 0].astype(np.float32)
    g = strip_rgb[:, :, 1].astype(np.float32)
    b = strip_rgb[:, :, 2].astype(np.float32)
    gray = cv2.cvtColor(strip_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)

    dark_score = 255.0 - gray
    red_score = r - 0.5 * (g + b)

    if np.percentile(dark_score, 98) > 125:
        score_map = np.maximum(0.0, dark_score - 85.0)
    else:
        score_map = np.maximum(0.0, red_score - 9.0)

    margin_y = max(2, int(h * 0.05))
    score_map[:margin_y, :] = 0
    score_map[h - margin_y:, :] = 0

    y_indices = np.arange(h, dtype=np.float32)
    signal = np.full(w, np.nan, dtype=np.float32)

    for x in range(w):
        col = score_map[:, x]
        max_val = np.max(col)
        if max_val > 3.5:
            weights = np.maximum(0.0, col - (max_val * 0.50))
            w_sum = np.sum(weights)
            if w_sum > 0:
                signal[x] = np.sum(y_indices * weights) / w_sum

    valid = ~np.isnan(signal)
    if np.sum(valid) < 10:
        return np.zeros(w, dtype=np.float32)

    x_all = np.arange(w)
    signal = np.interp(x_all, x_all[valid], signal[valid])
    signal = cv2.GaussianBlur(signal.reshape(1, -1), (1, 3), 0).flatten()
    return -signal


def detect_r_peaks(signal: np.ndarray, strip_duration_sec: float = 10.0, detected_class: str = "NORMAL"):
    """Sivrilik filtresiyle peak tespiti yapar ancak BPM hesabını GELİŞMİŞ OTOKORELASYON ile gerçekleştirir."""
    n = len(signal)
    if n < 20 or float(np.std(signal)) < 1e-4:
        return [], 0

    # Kalibrasyon Karesini Kırp (İlk %8)
    crop_idx = int(n * 0.08)
    sig_cropped = signal[crop_idx:]
    n_eff = len(sig_cropped)
    
    if n_eff < 20: return [], 0

    # --- Peak Tespiti (Konumlandırma için) ---
    narrow_win = max(5, int(n_eff * 0.012) | 1)
    narrow_smooth = cv2.GaussianBlur(sig_cropped.reshape(1, -1), (1, narrow_win), 0).flatten()
    high_freq = sig_cropped - narrow_smooth
    pos_peak = float(np.percentile(high_freq, 99.6))
    neg_peak = float(np.abs(np.percentile(high_freq, 0.4)))
    sharp_r = np.maximum(0.0, high_freq if pos_peak >= neg_peak * 0.85 else -high_freq)
    top_val = float(np.percentile(sharp_r, 99.6))
    thresh = 0.68 * top_val
    min_dist = max(10, int(n_eff * (0.52 / strip_duration_sec)))

    peaks = []
    for i in range(1, n_eff - 1):
        if sharp_r[i] >= thresh and sharp_r[i] >= sharp_r[i - 1] and sharp_r[i] >= sharp_r[i + 1]:
            if not peaks or (i - peaks[-1]) >= min_dist:
                peaks.append(i)
            elif sharp_r[i] > sharp_r[peaks[-1]]:
                peaks[-1] = i

    # --- NABIZ HESABI: Karesi Alınmış Zarfla Otokorelasyon ---
    rectified_sig = np.square(sig_norm := (sig_cropped - np.mean(sig_cropped)) / (np.std(sig_cropped) + 1e-6))
    corr = np.correlate(rectified_sig, rectified_sig, mode='full')
    corr = corr[len(corr)//2:] 
    
    time_axis = np.linspace(0, strip_duration_sec * (n_eff/n), len(corr))
    
    # Dinamik Arama Aralığı: EFOR ise alt sınır 0.35, değilse 0.58
    low_bound = 0.35 if "EFOR" in detected_class.upper() else 0.58
    mask = (time_axis >= low_bound) & (time_axis <= 1.50)
    
    if np.any(mask):
        search_area = corr[mask]
        best_lag_idx = np.argmax(search_area)
        t_rr = time_axis[mask][best_lag_idx]
        est_bpm = int(round(60.0 / t_rr))
    else:
        est_bpm = 0

    est_bpm = int(np.clip(est_bpm, 0, 200))
    final_peaks = [p + crop_idx for p in peaks]
    return final_peaks, est_bpm



def render_clean_ecg_window(signal_1d: np.ndarray, global_span: float, size: int = 224) -> np.ndarray:
    canvas = np.full((size, size, 3), 255, dtype=np.uint8)
    if len(signal_1d) < 5:
        return canvas

    x_old = np.linspace(0, 1, len(signal_1d))
    x_new = np.linspace(0, 1, size)
    sig = np.interp(x_new, x_old, signal_1d)

    s_min, s_max = np.min(sig), np.max(sig)
    local_span = s_max - s_min
    effective_span = max(local_span, global_span * 0.65)

    if effective_span < 1e-5:
        y_norm = np.full(size, size // 2, dtype=np.int32)
    else:
        pad = int(size * 0.08)
        center = 0.5 * (s_max + s_min)
        scaled = 0.5 + (sig - center) / effective_span
        y_norm = (size - pad) - (scaled * (size - 2 * pad))
        y_norm = np.clip(y_norm, pad, size - pad).astype(np.int32)

    pts = np.column_stack((np.arange(size, dtype=np.int32), y_norm)).reshape((-1, 1, 2))
    cv2.polylines(canvas, [pts], isClosed=False, color=(0, 0, 0), thickness=2, lineType=cv2.LINE_AA)
    return canvas


def classify_strip_auto(model, strip_rgb: np.ndarray):
    sig = extract_1d_signal_from_strip(strip_rgb)
    n = len(sig)
    p2, p98 = np.percentile(sig, [2, 98])
    global_span = float(p98 - p2)

    half_win = max(12, int(n * 0.038))
    
    # First Pass: Quick Classification to decide BPM boundaries
    temp_peaks, temp_bpm = detect_r_peaks(sig, strip_duration_sec=10.0, detected_class="UNKNOWN")
    
    valid_centers = [p for p in temp_peaks if half_win <= p <= (n - half_win)]
    if len(valid_centers) < 3:
        valid_centers = np.linspace(half_win, max(half_win + 1, n - half_win), 8, dtype=int).tolist()

    all_probs = []
    sample_renders = []

    for c in valid_centers:
        seg = sig[max(0, c - half_win):min(n, c + half_win)]
        rendered = render_clean_ecg_window(seg, global_span=global_span, size=224)
        sample_renders.append(rendered)
        _, _, p_dict = classify(model, rendered)
        all_probs.append(p_dict)

    keys = list(all_probs[0].keys())
    avg_probs = {k: float(np.mean([p[k] for p in all_probs])) for k in keys}
    top_class = max(avg_probs, key=avg_probs.get)
    
    # Second Pass: Refined BPM calculation based on identified class
    _, refined_bpm = detect_r_peaks(sig, strip_duration_sec=10.0, detected_class=top_class)
    
    return top_class, avg_probs[top_class], avg_probs, sample_renders[len(sample_renders) // 2], refined_bpm


def find_vertical_blocks(gray: np.ndarray):
    h, w = gray.shape
    profile = np.sum(gray < WHITE_THRESHOLD, axis=1)
    active = profile > max(3, MIN_ROW_DENSITY * w)

    blocks = []
    start = None
    for y, is_active in enumerate(active):
        if is_active and start is None:
            start = y
        elif not is_active and start is not None:
            blocks.append([start, y])
            start = None
    if start is not None:
        blocks.append([start, h])

    gap = max(3, int(h * MERGE_GAP_RATIO))
    merged = []
    for b in blocks:
        if merged and b[0] - merged[-1][1] <= gap:
            merged[-1][1] = b[1]
        else:
            merged.append(b)

    min_h = int(h * MIN_BLOCK_RATIO)
    return [(s, e) for s, e in merged if (e - s) >= min_h]


def crop_horizontal_bounds(gray: np.ndarray, y0: int, y1: int, margin: int = 4):
    band = gray[y0:y1] < WHITE_THRESHOLD
    cols = np.where(band.sum(axis=0) > 0)[0]
    if len(cols) == 0:
        return 0, gray.shape[1]
    return max(0, cols[0] - margin), min(gray.shape[1], cols[-1] + 1 + margin)


def split_into_strips(img_rgb: np.ndarray):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    h = gray.shape[0]
    blocks = find_vertical_blocks(gray)
    margin_y = max(2, int(h * 0.002))

    if len(blocks) == 1 and (blocks[0][1] - blocks[0][0]) > h * 0.65:
        top, bottom = blocks[0]
        n_rows = 4
        step = (bottom - top) / n_rows
        chosen = [(int(top + i * step), int(top + (i + 1) * step)) for i in range(n_rows)]
        labels = [f"Kanal {i+1} (Üst-Alt)" for i in range(n_rows)]
        method = "Tam sayfa ızgara tespiti (4 satır)"
    elif len(blocks) >= 3:
        sorted_by_h = sorted(blocks, key=lambda b: b[1] - b[0], reverse=True)
        n_take = 4 if len(blocks) >= 4 and (sorted_by_h[3][1] - sorted_by_h[3][0]) > h * 0.12 else 3
        chosen = sorted(sorted_by_h[:n_take], key=lambda b: b[0])
        labels = ["0-10 sn", "10-20 sn", "20-30 sn"] if n_take == 3 else [f"Satır {i+1}" for i in range(n_take)]
        method = f"Otomatik şerit tespiti ({n_take} satır)"
    else:
        mask_rows = np.where(np.sum(gray < WHITE_THRESHOLD, axis=1) > 3)[0]
        if len(mask_rows) == 0:
            return [], [], "İçerik bulunamadı"
        top, bottom = int(mask_rows[0]), int(mask_rows[-1]) + 1
        step = (bottom - top) / 3
        chosen = [(int(top + i * step), int(top + (i + 1) * step)) for i in range(3)]
        labels = ["0-10 sn", "10-20 sn", "20-30 sn"]
        method = "Yedek yöntem (3 eşit bölme)"

    strips = []
    for y0, y1 in chosen:
        y0m, y1m = max(0, y0 - margin_y), min(h, y1 + margin_y)
        x0, x1 = crop_horizontal_bounds(gray, y0m, y1m)
        strips.append(img_rgb[y0m:y1m, x0:x1].copy())
    return strips, labels, method


def is_multi_strip_report(img_rgb: np.ndarray, is_pdf: bool) -> bool:
    h, w = img_rgb.shape[:2]
    if is_pdf:
        return True
    if h <= 400 and w <= 400:
        return False
    aspect = w / max(h, 1)
    if 0.75 <= aspect <= 1.35:
        return False
    return True


def show_probabilities(prob_dict: dict):
    """Olasılıkları %0 - %100 ölçeğinde çizer."""
    pct_dict = {k: round(v * 100.0, 1) for k, v in prob_dict.items()}
    df = pd.DataFrame({"Benzerlik (%)": pct_dict}).sort_values("Benzerlik (%)", ascending=False)
    st.bar_chart(df)


def show_status_box(label: str, conf: float, bpm: int | None = None):
    desc = get_class_description(label)
    upper = label.strip().upper()
    is_noisy = "GURULTU" in upper

    bpm_part = ""
    if bpm is not None:
        bpm_status, bpm_note = evaluate_bpm(bpm, is_noisy=is_noisy)
        if is_noisy or bpm <= 0:
            bpm_part = f"\n\n**Kalp Atımı (BPM):** {bpm_status} — *{bpm_note}*"
        else:
            bpm_part = f"\n\n**Kalp Atımı (BPM):** ~{bpm} bpm — *{bpm_status}* ({bpm_note})"

    if is_noisy:
        st.warning(f"**Tespit Edilen Durum:** {label} (**%{conf * 100:.1f}**)\n\n{desc}{bpm_part}")
    elif "NORMAL" in upper:
        st.success(f"**Tespit Edilen Durum:** {label} (**%{conf * 100:.1f}**)\n\n{desc}{bpm_part}")
    else:
        st.error(f"**Tespit Edilen Durum:** {label} (**%{conf * 100:.1f}**)\n\n{desc}{bpm_part}")


def show_page_analysis(model, img_rgb: np.ndarray, page_no: int):
    st.subheader(f"Sayfa / Görüntü {page_no} — Genel Görünüm")
    col_img, col_res = st.columns([1.1, 1])
    with col_img:
        st.image(img_rgb, caption=f"Yüklenen Kayıt ({img_rgb.shape[1]}x{img_rgb.shape[0]} px)", use_container_width=True)
    with col_res:
        label, conf, probs = classify(model, img_rgb)
        show_status_box(label, conf)
        show_probabilities(probs)


def show_strip_analysis(model, img_rgb: np.ndarray, page_no: int, reported_bpm: int | None = None):
    st.subheader(f"Sayfa {page_no} — Şerit Bazlı Detaylı Analiz")
    strips, strip_labels, method = split_into_strips(img_rgb)
    st.caption(f"Bölümleme yöntemi: {method}")

    if not strips:
        st.warning("Şeritler tespit edilemedi. Sadece genel görünüm sonucu geçerlidir.")
        return

    summary_rows = []
    clean_labels = []
    bpm_values = []
    strip_results = []

    for label_text, strip in zip(strip_labels, strips):
        label, conf, probs, sample_224, est_bpm = classify_strip_auto(model, strip)
        is_noisy = "GURULTU" in label.strip().upper()
        if not is_noisy:
            clean_labels.append(label)
            if est_bpm > 0:
                bpm_values.append(est_bpm)
        strip_results.append((label_text, strip, label, conf, probs, sample_224, est_bpm, is_noisy))

    # --- ÜST KARAR KARTI ---
    if clean_labels:
        dominant = pd.Series(clean_labels).mode()[0]
    else:
        dominant = "GURULTULU"
    
    overall_quality = len(clean_labels) / len(strips)
    avg_bpm = int(round(float(np.mean(bpm_values)))) if bpm_values else (reported_bpm or 0)
    
    st.markdown(f"""
        <div style="background-color: #1e293b; padding: 1.5rem; border-radius: 15px; border: 1px solid #334155; color: #f8fafc; margin-bottom: 2rem;">
            <h3 style="margin-top:0; color: #f8fafc; text-align:center;">🎯 Genel Analiz Özeti</h3>
            <div style="display: flex; justify-content: space-around; text-align: center; flex-wrap: wrap;">
                <div><small style="color:#cbd5e1">Sinyal Kalitesi</small><br><b>%{(overall_quality*100):.0f} Temiz</b></div>
                <div><small style="color:#cbd5e1">Genel Ritim</small><br><b>{dominant}</b></div>
                <div><small style="color:#cbd5e1">Ortalama Nabız</small><br><b>{avg_bpm} BPM</b></div>
            </div>
            <div style="margin-top:1.5rem; text-align:center;">
                <small style="color:#cbd5e1">Nabız Skalası (40-140 BPM)</small>
                <div style="background-color: #334155; height: 12px; width: 100%; border-radius: 6px; position: relative; margin-top: 5px;">
                    <div style="position: absolute; left: {max(0, min(100, ((avg_bpm-40)/100)*100))}%; width: 6px; height: 20px; background-color: #ef4444; top: -4px; border-radius: 3px;"></div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Sade yönlendirme kutusu
    if overall_quality < 1.0:
        st.info(f"💡 Sinyalde kısmi parazit saptandı. Toplam {len(strips)} şeritten {(len(clean_labels))} tanesi temiz kabul edilip analiz edildi.")
    else:
        st.success("✨ Tüm şeritler yüksek kalitededir ve analiz edilebilir durumda.")

    tab1, tab2, tab3, tab4 = st.tabs(["📋 Şerit Bazlı Analiz", "📊 Özet Tablo & Rapor", "🔍 Ham Görüntü & Teknik", "⚖️ Kanuni & Hukuki Bilgilendirme"])

    with tab1:
        for label_text, strip, label, conf, probs, sample_224, est_bpm, is_noisy in strip_results:
            bpm_status, _ = evaluate_bpm(est_bpm, is_noisy=is_noisy)
            bpm_header = " | Parazitli" if is_noisy else (f" | ~{est_bpm} bpm ({bpm_status})" if est_bpm > 0 else "")

            with st.expander(f"{label_text} → {label} (%{conf * 100:.1f}){bpm_header}", expanded=True):
                col_orig, col_res = st.columns([1.2, 1])
                with col_orig:
                    st.image(strip, caption=f"{label_text} — Orijinal Şerit", use_container_width=True)
                    with st.popover("Modelin İncelediği Atım Kesitini Gör (224x224)"):
                        st.image(sample_224, caption="Otomatik Hizalanan Örnek Atım", use_container_width=True)
                with col_res:
                    show_status_box(label, conf, bpm=est_bpm)
                    
                    st.markdown("<br>**Sınıf Benzerlikleri:**", unsafe_allow_html=True)
                    # High contrast probability bars
                    for cls_name, prob_val in probs.items():
                        desc = get_class_description(cls_name)
                        st.markdown(f'<div style="color:#cbd5e1; font-size:0.85rem; display:flex; justify-content:space-between;"><span>{cls_name}</span><span>%{prob_val*100:.1f}</span></div>', unsafe_allow_html=True)
                        st.progress(prob_val)
                        st.markdown(f'<div style="color:#64748b; font-size:0.75rem; margin-bottom:10px;">{desc}</div>', unsafe_allow_html=True)

            summary_rows.append({
                "Bölüm": label_text,
                "Durum": label,
                "Açıklama": get_class_description(label),
                "Güven (%)": round(conf * 100, 1),
                "Tahmini Nabız": "—" if is_noisy or est_bpm <= 0 else f"~{est_bpm} bpm",
                "Nabız Yorumu": bpm_status,
            })

    with tab2:
        st.markdown("**Detaylı Karşılaştırma Tablosu**")
        df_sum = pd.DataFrame(summary_rows)
        st.dataframe(df_sum, hide_index=True, use_container_width=True)
        csv = df_sum.to_csv(index=False).encode('utf-8')
        st.download_button("CSV Rapor İndir", data=csv, file_name="ekg_analiz_raporu.csv", mime='text/csv')

    with tab3:
        st.markdown("**Teknik Bilgiler**")
        st.write(f"- **Bölümleme Metodu:** {method}")
        st.image(img_rgb, caption="Ham Görüntü / Sayfa Tamamı", use_container_width=True)

    with tab4:
        st.markdown("""
            <div style="background-color: #1e293b; padding: 1.5rem; border-radius: 15px; border: 1px solid #334155; margin-bottom: 1rem;">
                <h4 style="color:#f8fafc; margin-top:0;">🛡️ Tıbbi Sorumluluk Reddi (Medical Disclaimer)</h4>
                <p style="color:#cbd5e1; font-size:0.9rem;">Bu platform kesinlikle bir tıbbi teşhis, tanı veya tedavi cihazı değildir. Sunulan analizler yalnızca araştırma, akademik eğitim ve ön bilgilendirme amacı taşımaktadır. Klinik kararlar sadece uzman hekimler tarafından verilmelidir.</p>
            </div>
            <div style="background-color: #1e293b; padding: 1.5rem; border-radius: 15px; border: 1px solid #334155; margin-bottom: 1rem;">
                <h4 style="color:#f8fafc; margin-top:0;">🔐 KVKK ve Anonim Veri İşleme</h4>
                <p style="color:#cbd5e1; font-size:0.9rem;">Yüklenen kayıtlar kullanıcının açık rızası olmaksızın sisteme kaydedilmez. Onay verildiğinde, kişisel kimlik bilgilerinden arındırılmış halde yalnızca yapay zekâ modelinin doğruluğunu artırmak adına anonim olarak işlenmektedir.</p>
            </div>
            <div style="background-color: #1e293b; padding: 1.5rem; border-radius: 15px; border: 1px solid #334155;">
                <h4 style="color:#f8fafc; margin-top:0;">📜 Kullanım Koşulları ve Fikri Mülkiyet</h4>
                <p style="color:#cbd5e1; font-size:0.9rem;">Sistemin ürettiği çıktılar resmi rapor niteliği taşımaz. Tüm algoritmalar ve marka hakları ScanVitals mülkiyetindedir.</p>
            </div>
        """, unsafe_allow_html=True)


def main():
    st.set_page_config(page_title="ScanVitals", page_icon="🫀", layout="centered")
    
    if "lang" not in st.session_state:
        st.session_state.lang = "tr"

    # Language Switcher in Header
    col_logo, col_lang = st.columns([0.8, 0.2])
    with col_lang:
        st.session_state.lang = st.radio("🌐 Language / Dil", ["tr", "en"], horizontal=True, label_visibility="collapsed")

    # Custom CSS for Dark Theme Contrast & Mobile Responsiveness
    st.markdown("""
        <style>
        .stApp { background-color: #0f172a !important; }
        [data-testid="stHeader"] { background-color: rgba(0,0,0,0)!important; }
        .main { background-color: #0f172a !important; color: #f8fafc !important; }
        div[data-testid="stMetricValue"] { font-size: 2rem !important; color: #f8fafc !important; }
        div[data-testid="stMetricLabel"] { color: #cbd5e1 !important; }
        .stExpander { border: 1px solid #334155 !important; border-radius: 12px !important; background-color: #1e293b !important; box-shadow: 0 4px 6px rgba(0,0,0,0.3); margin-bottom: 1rem !important; }
        .stExpander [data-testid="stExpanderParams"] { color: #f8fafc !important; }
        .hero-card { background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); color: #f8fafc; padding: 2rem; border-radius: 20px; text-align: center; margin-bottom: 2rem; border: 1px solid #334155; box-shadow: 0 10px 20px rgba(0,0,0,0.4); }
        .step-card { background-color: #1e293b; color: #f8fafc; padding: 1.5rem; border-radius: 15px; border-left: 5px solid #3b82f6; margin-bottom: 1rem; border: 1px solid #334155; }
        label, .stMarkdown p, span { color: #f8fafc !important; }
        </style>
    """, unsafe_allow_html=True)

    # TOP NAVIGATION TABS
    tab_main, tab_legal = st.tabs([t("nav_center"), t("nav_legal")])

    with tab_legal:
        st.markdown(f"""
            <div style="background-color: #1e293b; padding: 1.5rem; border-radius: 15px; border: 1px solid #334155; margin-bottom: 1rem;">
                <h4 style="color:#f8fafc; margin-top:0;">{t('legal_disclaimer')}</h4>
                <p style="color:#cbd5e1; font-size:0.9rem;">{t('legal_disclaimer_txt')}</p>
            </div>
            <div style="background-color: #1e293b; padding: 1.5rem; border-radius: 15px; border: 1px solid #334155; margin-bottom: 1rem;">
                <h4 style="color:#f8fafc; margin-top:0;">{t('legal_kvkk')}</h4>
                <p style="color:#cbd5e1; font-size:0.9rem;">Türkiye TİTCK, 6698 sayılı KVKK Madde 6, EU MDR, FDA, GDPR Madde 9 ve EU AI Act kapsamındaki bilgilendirmeler çerçevesinde; yüklenen kayıtlar kullanıcının açık rızası olmaksızın sisteme kaydedilmez. Onay verildiğinde, kişisel kimlik bilgileri içermeden yalnızca yapay zekâ model iyileştirmesi amacıyla anonim işlenmektedir.</p>
            </div>
            <div style="background-color: #1e293b; padding: 1.5rem; border-radius: 15px; border: 1px solid #334155;">
                <h4 style="color:#f8fafc; margin-top:0;">{t('legal_terms')}</h4>
                <p style="color:#cbd5e1; font-size:0.9rem;">{t('legal_terms_txt')}</p>
            </div>
        """, unsafe_allow_html=True)

    with tab_main:
        st.markdown(f'''
            <div class="hero-card">
                <h1 style="color:white; margin-bottom:0.5rem;">{t('app_name')}</h1>
                <p style="color:#cbd5e1; font-size:1.1rem;">{t('subtitle')}</p>
                <div style="margin-top:1.5rem; display:flex; justify-content:center; align-items:center; gap:10px; flex-wrap:wrap;">
                    <span style="background-color:#3b82f6; color:white; padding:4px 12px; border-radius:20px; font-size:0.8rem; font-weight:bold;">{t('mod_active')}</span>
                    <span style="color:#64748b; font-size:0.8rem;">{t('mod_future')}</span>
                </div>
            </div>
        ''', unsafe_allow_html=True)

        if not os.path.exists(MODEL_PATH):
            st.error(f"Model dosyası bulunamadı: {MODEL_PATH}")
            st.stop()

        model = load_model()

        uploaded = st.file_uploader(t("upload_lbl"), type=["jpg", "jpeg", "png", "pdf"])
        
        if uploaded is None:
            st.markdown(f"### {t('welcome_title')}")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f'<div class="step-card">{t("step1")}</div>', unsafe_allow_html=True)
            with col2:
                st.markdown(f'<div class="step-card">{t("step2")}</div>', unsafe_allow_html=True)
            with col3:
                st.markdown(f'<div class="step-card">{t("step3")}</div>', unsafe_allow_html=True)
            
            st.markdown("---")
            st.markdown(f"### {t('ritms_title')}")
            grid_cols = st.columns(2)
            descriptions = [
                ("✅ NORMAL", "Düzenli kalp atımı ve sağlıklı dalga formu."),
                ("⚠️ DUZENSİZ", "Atımlar arası süre değişkenliği mevcut."),
                ("⚡ EFOR", "Hızlı nabız ve yüksek aktivite profili."),
                ("🚩 ISKEMI", "Kan akışı yetersizliğini işaret eden formlar."),
                ("🚫 GURULTULU", "Parazitli veri, güvenilir değil.")
            ]
            for idx, (title, desc) in enumerate(descriptions):
                with grid_cols[idx % 2]:
                    st.markdown(f"**{title}**: {desc}")
            return

        try:
            pages, is_pdf, reported_bpms = load_uploaded_file(uploaded)
        except Exception as e:
            st.error(f"Dosya okunamadı: {e}")
            return

        for i, (page_img, rep_bpm) in enumerate(zip(pages, reported_bpms), start=1):
            st.divider()
            valid, reason = is_valid_ecg_image(page_img)
            if not valid:
                st.error(f"**Kayıt Dışı / Geçersiz Görsel (Sayfa {i}):** {reason}")
                continue

            if is_multi_strip_report(page_img, is_pdf):
                show_strip_analysis(model, page_img, i, reported_bpm=rep_bpm)
                with st.expander(f"Sayfa {i} — Tam Sayfa Ham Görüntü & Genel Bakış"):
                    show_page_analysis(model, page_img, i)
            else:
                show_page_analysis(model, page_img, i)

        st.divider()
        
        st.markdown(f"### {t('pool_title')}")
        with st.container():
            st.markdown('<div style="background-color: #1e293b; padding: 1.5rem; border-radius: 15px; border: 1px solid #334155;">', unsafe_allow_html=True)
            st.markdown(f'<p style="color:#cbd5e1; font-size:0.9rem; margin-bottom:1rem;">{t("pool_info")}</p>', unsafe_allow_html=True)
            
            col_pool1, col_pool2 = st.columns([3, 1])
            with col_pool1:
                agree = st.checkbox(t("pool_consent"))
            with col_pool2:
                initial_pred = "NORMAL"
                if pages:
                    res = model.predict(Image.fromarray(pages[0]), verbose=False)[0]
                    initial_pred = res.names[int(np.argmax(res.probs.data.cpu().numpy()))]
                
                opt_list = [t("accept_model")] + list(CLASS_DESCRIPTIONS.keys())
                selected_opt = st.selectbox(t("pool_select"), opt_list, index=0)
                final_class = selected_opt if selected_opt != t("accept_model") else initial_pred

            st.markdown(f'<p style="color:#64748b; font-size:0.75rem; margin-top:-0.5rem; margin-bottom:1rem;">{t("pool_hint")}</p>', unsafe_allow_html=True)
            
            if st.button(t("btn_save")):
                if agree:
                    save_path = os.path.join("v5_egitim_havuzu", final_class)
                    os.makedirs(save_path, exist_ok=True)
                    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
                    ext = ".jpg" if not is_pdf else ".pdf"
                    filename = f"rec_{timestamp}{ext}"
                    
                    if is_pdf:
                        with open(os.path.join(save_path, filename), "wb") as f:
                            f.write(uploaded.getvalue())
                    else:
                        img_out = Image.fromarray(pages[0])
                        img_out.save(os.path.join(save_path, filename))
                    
                    st.success(t("saved_msg"))
                    subprocess.Popen(["python", "auto_retrain_v5.py"])
                    total_records = sum([len(files) for r, d in os.walk("v5_egitim_havuzu") for files in [d]])
                    st.info(t("pool_cnt").format(total_records))
                else:
                    st.warning("Lütfen önce izin kutucuğunu onaylayın.")
            st.markdown('</div>', unsafe_allow_html=True)

    st.divider()
    st.markdown(f'<div style="text-align:center; color:#64748b; font-size:0.8rem; padding:2rem 0;">{t("footer")}</div>', unsafe_allow_html=True)
    st.info(t("med_warning"))



if __name__ == "__main__":
    main()