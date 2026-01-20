# -*- coding: utf-8 -*-
"""
進階搜尋與取代對話框
"""
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QLineEdit, 
    QGroupBox, QHBoxLayout, QRadioButton, QCheckBox, QPushButton
)
from lib.locales import load_locale, tr as locale_tr



class AdvancedFindReplaceDialog(QDialog):
    def tr(self, key: str) -> str:
        lang = "zh_tw"
        if self.parent() and hasattr(self.parent(), "settings"):
            lang = self.parent().settings.get("ui_language", "zh_tw")
        load_locale(lang)
        return locale_tr(key)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(self.tr("title_find_replace"))
        self.setMinimumWidth(400)
        self.layout = QVBoxLayout(self)
        form = QFormLayout()
        self.find_edit = QLineEdit()
        self.replace_edit = QLineEdit()
        form.addRow(self.tr("label_find"), self.find_edit)
        form.addRow(self.tr("label_replace"), self.replace_edit)
        self.layout.addLayout(form)
        self.grp_scope = QGroupBox(self.tr("grp_scope"))
        scope_layout = QHBoxLayout()
        self.rb_current = QRadioButton(self.tr("rb_scope_current"))
        self.rb_all = QRadioButton(self.tr("rb_scope_all"))
        self.rb_current.setChecked(True)
        scope_layout.addWidget(self.rb_current)
        scope_layout.addWidget(self.rb_all)
        self.grp_scope.setLayout(scope_layout)
        self.layout.addWidget(self.grp_scope)
        self.grp_mode = QGroupBox(self.tr("grp_mode"))
        mode_layout = QHBoxLayout()
        self.chk_case = QCheckBox(self.tr("chk_case"))
        self.chk_regex = QCheckBox(self.tr("chk_regex"))
        mode_layout.addWidget(self.chk_case)
        mode_layout.addWidget(self.chk_regex)
        self.grp_mode.setLayout(mode_layout)
        self.layout.addWidget(self.grp_mode)
        btn_layout = QHBoxLayout()
        self.btn_replace = QPushButton(self.tr("btn_replace_action"))
        self.btn_cancel = QPushButton(self.tr("setting_cancel"))
        btn_layout.addWidget(self.btn_replace)
        btn_layout.addWidget(self.btn_cancel)
        self.layout.addLayout(btn_layout)
        self.btn_replace.clicked.connect(self.accept)
        self.btn_cancel.clicked.connect(self.reject)

    def get_settings(self):
        return {
            'find': self.find_edit.text(),
            'replace': self.replace_edit.text(),
            'scope_all': self.rb_all.isChecked(),
            'case_sensitive': self.chk_case.isChecked(),
            'regex': self.chk_regex.isChecked()
        }
