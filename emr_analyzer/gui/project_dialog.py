"""Project selection/creation dialog shown at application startup."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QListWidget, QListWidgetItem, QFileDialog, QMessageBox,
    QGroupBox,
)
from PyQt5.QtCore import Qt

from ..config import BASE_DIR, active_workspace

PROJECTS_CONFIG = BASE_DIR / "projects.json"


def _load_projects() -> list[dict]:
    if not PROJECTS_CONFIG.exists():
        return []
    try:
        data = json.loads(PROJECTS_CONFIG.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (OSError, ValueError):
        return []


def _save_projects(projects: list[dict]) -> None:
    PROJECTS_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    PROJECTS_CONFIG.write_text(
        json.dumps(projects, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


class ProjectDialog(QDialog):
    """Startup dialog: open existing project or create a new one."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("EMR Analyzer — Seleziona Progetto")
        self.setMinimumWidth(600)
        self.setMinimumHeight(400)
        self._selected_path: Path | None = None
        self._projects = _load_projects()
        self._setup_ui()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    @property
    def selected_path(self) -> Path | None:
        return self._selected_path

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        title = QLabel("<h2>Progetti EMR Analyzer</h2>")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # Existing projects
        group = QGroupBox("Progetti recenti")
        group_layout = QVBoxLayout(group)

        self._list = QListWidget()
        self._list.setMinimumHeight(150)
        self._list.itemDoubleClicked.connect(self._on_open_selected)
        group_layout.addWidget(self._list)
        layout.addWidget(group)

        # Buttons
        btn_layout = QHBoxLayout()

        self._open_btn = QPushButton("📂 Apri progetto selezionato")
        self._open_btn.clicked.connect(self._on_open_selected)

        new_btn = QPushButton("🆕 Crea nuovo progetto")
        new_btn.setObjectName("successButton")
        new_btn.clicked.connect(self._on_create_new)

        btn_layout.addWidget(self._open_btn)
        btn_layout.addWidget(new_btn)
        layout.addLayout(btn_layout)

        self._refresh_list()

        # Quick open
        quick_layout = QHBoxLayout()
        quick_open_btn = QPushButton("📁 Apri cartella esistente...")
        quick_open_btn.clicked.connect(self._on_quick_open)
        quick_layout.addWidget(quick_open_btn)
        quick_layout.addStretch()
        layout.addLayout(quick_layout)

    def _refresh_list(self):
        self._list.clear()
        self._projects = _load_projects()
        for p in self._projects:
            item = QListWidgetItem(
                f"{p.get('name', 'Senza nome')}  —  {p.get('path', '')}"
            )
            item.setData(Qt.UserRole, p.get("path", ""))
            self._list.addItem(item)
        self._open_btn.setEnabled(len(self._projects) > 0)

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _on_open_selected(self):
        # Check double-click first, then selected
        items = self._list.selectedItems()
        if not items:
            # Try double-clicked item
            item = self._list.currentItem()
            if not item:
                return
        else:
            item = items[0]

        path = item.data(Qt.UserRole)
        if not path:
            return
        project_path = Path(path)
        if not project_path.exists():
            QMessageBox.warning(
                self, "Progetto non trovato",
                f"La cartella del progetto non esiste più:\n{path}\n\n"
                f"Potrebbe essere stata spostata o eliminata."
            )
            self._remove_project(path)
            return

        self._selected_path = project_path
        self._touch_project(path)
        self.accept()

    def _on_create_new(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Seleziona cartella per il nuovo progetto",
            str(Path.home()),
        )
        if not folder:
            return

        # Use folder name as project name
        project_path = Path(folder)
        name = project_path.name

        # Create subdirectory structure
        project_path.mkdir(parents=True, exist_ok=True)

        # Add to projects list
        self._add_project(name, str(project_path))
        self._selected_path = project_path
        self.accept()

    def _on_quick_open(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Apri cartella progetto esistente",
            str(Path.home()),
        )
        if not folder:
            return
        project_path = Path(folder)
        name = project_path.name

        # Add to projects list if not already there
        existing = [p for p in self._projects if p.get("path") == folder]
        if not existing:
            self._add_project(name, folder)

        self._selected_path = project_path
        self.accept()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _add_project(self, name: str, path: str):
        self._projects.append({
            "name": name,
            "path": path,
            "created_at": datetime.now().isoformat(),
            "last_opened": datetime.now().isoformat(),
        })
        _save_projects(self._projects)

    def _remove_project(self, path: str):
        self._projects = [
            p for p in self._projects if p.get("path") != path
        ]
        _save_projects(self._projects)
        self._refresh_list()

    def _touch_project(self, path: str):
        for p in self._projects:
            if p.get("path") == path:
                p["last_opened"] = datetime.now().isoformat()
        _save_projects(self._projects)
