# -*- coding: utf-8 -*-
"""
進階搜尋與取代對話框
"""
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QLineEdit, 
    QGroupBox, QHBoxLayout, QRadioButton, QCheckBox, QPushButton
)


class AdvancedFindReplaceDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Advanced Find & Replace")
        self.setMinimumWidth(400)
        self.layout = QVBoxLayout(self)
        form = QFormLayout()
        self.find_edit = QLineEdit()
        self.replace_edit = QLineEdit()
        form.addRow("Find:", self.find_edit)
        form.addRow("Replace:", self.replace_edit)
        self.layout.addLayout(form)
        self.grp_scope = QGroupBox("Scope")
        scope_layout = QHBoxLayout()
        self.rb_current = QRadioButton("Current Image Only")
        self.rb_all = QRadioButton("All Images")
        self.rb_current.setChecked(True)
        scope_layout.addWidget(self.rb_current)
        scope_layout.addWidget(self.rb_all)
        self.grp_scope.setLayout(scope_layout)
        self.layout.addWidget(self.grp_scope)
        self.grp_mode = QGroupBox("Mode")
        mode_layout = QHBoxLayout()
        self.chk_case = QCheckBox("Case Sensitive")
        self.chk_regex = QCheckBox("Regular Expression")
        mode_layout.addWidget(self.chk_case)
        mode_layout.addWidget(self.chk_regex)
        self.grp_mode.setLayout(mode_layout)
        self.layout.addWidget(self.grp_mode)
        btn_layout = QHBoxLayout()
        self.btn_replace = QPushButton("Replace")
        self.btn_cancel = QPushButton("Cancel")
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
