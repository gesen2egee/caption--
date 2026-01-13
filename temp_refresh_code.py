    def refresh_file_list(self):
        if not self.root_dir_path or not os.path.exists(self.root_dir_path):
            return
        
        dir_path = self.root_dir_path
        # Keep track of current file to restore selection
        current_path = self.current_image_path
        
        self.image_files = []
        valid_exts = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')
        ignore_dirs = {"no_used", "unmask"}

        # Copied logic from open_directory scan
        try:
            for entry in os.scandir(dir_path):
                if entry.is_file() and entry.name.lower().endswith(valid_exts):
                    if any(part.lower() in ignore_dirs for part in Path(entry.path).parts):
                        continue
                    self.image_files.append(entry.path)
        except Exception:
            pass

        try:
            for entry in os.scandir(dir_path):
                if entry.is_dir():
                    if entry.name.lower() in ignore_dirs:
                        continue
                    try:
                        for sub in os.scandir(entry.path):
                            if sub.is_file() and sub.name.lower().endswith(valid_exts):
                                if any(part.lower() in ignore_dirs for part in Path(sub.path).parts):
                                    continue
                                self.image_files.append(sub.path)
                    except Exception:
                        pass
        except Exception:
            pass

        self.image_files = natsorted(self.image_files)

        if not self.image_files:
            self.image_label.clear()
            self.txt_edit.clear()
            self.img_info_label.setText("No Images Found")
            self.current_index = -1
            self.current_image_path = None
            return

        # Restore index
        if current_path and current_path in self.image_files:
            self.current_index = self.image_files.index(current_path)
        else:
            # If current file gone, try to stay at same index or 0
            if self.current_index >= len(self.image_files):
                self.current_index = len(self.image_files) - 1
            if self.current_index < 0:
                self.current_index = 0
        
        self.load_image()
        self.statusBar().showMessage(f"已重新整理列表: 共 {len(self.image_files)} 張圖片", 3000)
