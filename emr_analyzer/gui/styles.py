"""QSS stylesheets for EMR Analyzer."""

MAIN_STYLESHEET = """
/* --- Global --- */
QMainWindow {
    background-color: #f5f6fa;
}
QWidget {
    font-size: 13px;
}

/* --- Menu Bar --- */
QMenuBar {
    background-color: #2c3e50;
    color: #ecf0f1;
    padding: 2px 0px;
}
QMenuBar::item:selected {
    background-color: #34495e;
}
QMenu {
    background-color: #ffffff;
    border: 1px solid #dcdde1;
    padding: 4px 0px;
}
QMenu::item {
    padding: 6px 24px;
}
QMenu::item:selected {
    background-color: #3498db;
    color: white;
}

/* --- Toolbar --- */
QToolBar {
    background-color: #ecf0f1;
    border-bottom: 1px solid #dcdde1;
    padding: 4px;
    spacing: 4px;
}
QToolButton {
    padding: 6px 12px;
    border: 1px solid transparent;
    border-radius: 4px;
    background-color: transparent;
}
QToolButton:hover {
    background-color: #dcdde1;
    border-color: #bdc3c7;
}
QToolButton:pressed {
    background-color: #bdc3c7;
}

/* --- Tabs --- */
QTabWidget::pane {
    border: 1px solid #dcdde1;
    background-color: #ffffff;
    border-radius: 0px 0px 4px 4px;
}
QTabBar::tab {
    background-color: #ecf0f1;
    border: 1px solid #dcdde1;
    padding: 8px 20px;
    margin-right: 2px;
    border-radius: 4px 4px 0px 0px;
}
QTabBar::tab:selected {
    background-color: #ffffff;
    border-bottom-color: #ffffff;
    font-weight: bold;
}
QTabBar::tab:hover:!selected {
    background-color: #dcdde1;
}

/* --- Buttons --- */
QPushButton {
    padding: 8px 16px;
    border: 1px solid #bdc3c7;
    border-radius: 4px;
    background-color: #ecf0f1;
    min-height: 22px;
}
QPushButton:hover {
    background-color: #dcdde1;
}
QPushButton:pressed {
    background-color: #bdc3c7;
}
QPushButton:default {
    background-color: #3498db;
    color: white;
    border-color: #2980b9;
}
QPushButton:default:hover {
    background-color: #2980b9;
}
QPushButton#dangerButton {
    background-color: #e74c3c;
    color: white;
    border-color: #c0392b;
}
QPushButton#dangerButton:hover {
    background-color: #c0392b;
}
QPushButton#successButton {
    background-color: #2ecc71;
    color: white;
    border-color: #27ae60;
}
QPushButton#successButton:hover {
    background-color: #27ae60;
}

/* --- Input Fields --- */
QLineEdit, QTextEdit, QPlainTextEdit {
    border: 1px solid #bdc3c7;
    border-radius: 4px;
    padding: 6px;
    background-color: #ffffff;
}
QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus {
    border-color: #3498db;
}
QComboBox {
    border: 1px solid #bdc3c7;
    border-radius: 4px;
    padding: 6px;
    background-color: #ffffff;
    min-height: 22px;
}
QComboBox:focus {
    border-color: #3498db;
}
QComboBox::drop-down {
    border: none;
    padding-right: 8px;
}

/* --- Tables --- */
QTableView, QTableWidget {
    border: 1px solid #dcdde1;
    gridline-color: #ecf0f1;
    selection-background-color: #3498db;
    selection-color: white;
    alternate-background-color: #f8f9fa;
}
QTableView::item, QTableWidget::item {
    padding: 4px 8px;
}
QHeaderView::section {
    background-color: #ecf0f1;
    border: 1px solid #dcdde1;
    padding: 6px 8px;
    font-weight: bold;
}

/* --- Tree View --- */
QTreeView {
    border: 1px solid #dcdde1;
    alternate-background-color: #f8f9fa;
}
QTreeView::item {
    padding: 4px 4px;
}
QTreeView::item:selected {
    background-color: #3498db;
    color: white;
}

/* --- List View --- */
QListWidget {
    border: 1px solid #dcdde1;
    background-color: #ffffff;
}
QListWidget::item {
    padding: 8px 12px;
    border-bottom: 1px solid #f0f0f0;
}
QListWidget::item:selected {
    background-color: #3498db;
    color: white;
}
QListWidget::item:hover:!selected {
    background-color: #f0f4ff;
}

/* --- Scroll Area --- */
QScrollArea {
    border: none;
    background-color: transparent;
}

/* --- Splitter --- */
QSplitter::handle {
    background-color: #dcdde1;
}
QSplitter::handle:horizontal {
    width: 3px;
}
QSplitter::handle:vertical {
    height: 3px;
}

/* --- Progress Bar --- */
QProgressBar {
    border: 1px solid #bdc3c7;
    border-radius: 4px;
    background-color: #ecf0f1;
    text-align: center;
    height: 20px;
}
QProgressBar::chunk {
    background-color: #3498db;
    border-radius: 3px;
}

/* --- Status Bar --- */
QStatusBar {
    background-color: #ecf0f1;
    border-top: 1px solid #dcdde1;
    padding: 2px 8px;
}

/* --- Group Box --- */
QGroupBox {
    font-weight: bold;
    border: 1px solid #dcdde1;
    border-radius: 4px;
    margin-top: 12px;
    padding-top: 16px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0px 4px;
}

/* --- Dialog --- */
QDialog {
    background-color: #f5f6fa;
}

/* --- Labels --- */
QLabel#heading {
    font-size: 16px;
    font-weight: bold;
    color: #2c3e50;
}
QLabel#subheading {
    font-size: 14px;
    font-weight: bold;
    color: #34495e;
}
QLabel#statusOk {
    color: #27ae60;
    font-weight: bold;
}
QLabel#statusError {
    color: #e74c3c;
    font-weight: bold;
}
QLabel#statusWarning {
    color: #f39c12;
    font-weight: bold;
}

/* --- Drop Zone --- */
QWidget#dropZone {
    border: 3px dashed #bdc3c7;
    border-radius: 8px;
    background-color: #f8f9fa;
}
QWidget#dropZone:hover {
    border-color: #3498db;
    background-color: #f0f4ff;
}

/* --- Abnormal row highlight --- */
QTableWidget#labTable {
    /* Rows colored programmatically */
}
"""

DARK_STYLESHEET = """
/* Dark theme variant (future use) */
"""
