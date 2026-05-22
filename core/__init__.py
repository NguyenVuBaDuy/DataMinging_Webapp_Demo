"""
core/ - Package chứa tất cả module nội bộ của ứng dụng.

Modules:
    - config: Hằng số, đường dẫn, nhãn tiếng Việt
    - model_loader: Load model/scaler/config (cached)
    - styles: Custom CSS
    - ui_sidebar: Sidebar UI
    - ui_input_form: Form nhập liệu 12 features
"""

from core.config import BASE_DIR, CONFIG_DIR, FEATURE_LABELS_VI
from core.model_loader import load_all_resources
from core.styles import inject_custom_css
from core.ui_sidebar import render_sidebar
from core.ui_input_form import render_input_form, render_input_summary
