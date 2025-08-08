import gradio as gr
import pandas as pd
import yaml
from ktem.app import BasePage
from ktem.utils.file import YAMLNoDateSafeLoader

from .manager import IndexManager


# UGLY way to restart gradio server by updating atime
def update_current_module_atime():
    import os
    import time

    # Define the file path
    file_path = __file__
    print("Updating atime for", file_path)

    # Get the current time
    current_time = time.time()
    # Set the modified time (and access time) to the current time
    os.utime(file_path, (current_time, current_time))


def format_description(cls):
    user_settings = cls.get_admin_settings()
    params_lines = ["| Tên | Mặc định | Mô tả |", "| --- | --- | --- |"]
    for key, value in user_settings.items():
        params_lines.append(
            f"| {key} | {value.get('value', '')} | {value.get('info', '')} |"
        )
    return f"{cls.__doc__}\n\n" + "\n".join(params_lines)


class IndexManagement(BasePage):
    def __init__(self, app):
        self._app = app
        self.manager: IndexManager = app.index_manager
        self.spec_desc_default = (
            "# Mô tả đặc tả\n\nChọn một chỉ mục để xem mô tả đặc tả."
        )
        self.on_building_ui()

    def on_building_ui(self):
        with gr.Tab(label="Xem"):
            self.index_list = gr.DataFrame(
                headers=["id", "tên", "loại chỉ mục"],
                interactive=False,
            )

            with gr.Column(visible=False) as self._selected_panel:
                self.selected_index_id = gr.Number(value=-1, visible=False)
                with gr.Row():
                    with gr.Column():
                        self.edit_name = gr.Textbox(
                            label="Tên chỉ mục",
                        )
                        self.edit_spec = gr.Textbox(
                            label="Cấu hình chỉ mục",
                            info="Cấu hình quản trị của Chỉ mục ở định dạng YAML",
                            lines=10,
                        )

                        gr.Markdown(
                            "QUAN TRỌNG: Việc thay đổi hoặc xóa chỉ mục sẽ yêu cầu "
                            "khởi động lại hệ thống. Một số cài đặt cấu hình sẽ yêu cầu "
                            "xây dựng lại chỉ mục để chỉ mục hoạt động đúng cách."
                        )
                        with gr.Row():
                            self.btn_edit_save = gr.Button(
                                "Lưu", min_width=10, variant="primary"
                            )
                            self.btn_delete = gr.Button(
                                "Xóa", min_width=10, variant="stop"
                            )
                            with gr.Row(visible=False) as self._delete_confirm:
                                self.btn_delete_yes = gr.Button(
                                    "Xác nhận Xóa",
                                    variant="stop",
                                    min_width=10,
                                )
                                self.btn_delete_no = gr.Button("Hủy", min_width=10)
                            self.btn_close = gr.Button("Đóng", min_width=10)

                    with gr.Column():
                        self.edit_spec_desc = gr.Markdown("# Mô tả đặc tả")

        with gr.Tab(label="Thêm"):
            with gr.Row():
                with gr.Column(scale=2):
                    self.name = gr.Textbox(
                        label="Tên chỉ mục",
                        info="Phải là duy nhất và không được để trống.",
                    )
                    self.index_type = gr.Dropdown(label="Loại chỉ mục")
                    self.spec = gr.Textbox(
                        label="Đặc tả",
                        info="Đặc tả của chỉ mục ở định dạng YAML.",
                    )
                    gr.Markdown(
                        "<mark>Lưu ý</mark>: "
                        "Sau khi tạo chỉ mục, vui lòng khởi động lại ứng dụng"
                    )
                    self.btn_new = gr.Button("Thêm", variant="primary")

                with gr.Column(scale=3):
                    self.spec_desc = gr.Markdown(self.spec_desc_default)

    def _on_app_created(self):
        """Called when the app is created"""
        self._app.app.load(
            self.list_indices,
            inputs=[],
            outputs=[self.index_list],
        )
        self._app.app.load(
            lambda: gr.update(
                choices=[
                    (key.split(".")[-1], key) for key in self.manager.index_types.keys()
                ]
            ),
            outputs=[self.index_type],
        )

    def on_register_events(self):
        self.index_type.select(
            self.on_index_type_change,
            inputs=[self.index_type],
            outputs=[self.spec, self.spec_desc],
        )
        self.btn_new.click(
            self.create_index,
            inputs=[self.name, self.index_type, self.spec],
            outputs=None,
        ).success(self.list_indices, inputs=[], outputs=[self.index_list]).success(
            lambda: ("", None, "", self.spec_desc_default),
            outputs=[
                self.name,
                self.index_type,
                self.spec,
                self.spec_desc,
            ],
        ).success(
            update_current_module_atime
        )
        self.index_list.select(
            self.select_index,
            inputs=self.index_list,
            outputs=[self.selected_index_id],
            show_progress="hidden",
        )

        self.selected_index_id.change(
            self.on_selected_index_change,
            inputs=[self.selected_index_id],
            outputs=[
                self._selected_panel,
                # edit section
                self.edit_spec,
                self.edit_spec_desc,
                self.edit_name,
            ],
            show_progress="hidden",
        )
        self.btn_delete.click(
            lambda: (
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=True),
            ),
            inputs=[],
            outputs=[
                self.btn_edit_save,
                self.btn_delete,
                self.btn_close,
                self._delete_confirm,
            ],
            show_progress="hidden",
        )
        self.btn_delete_yes.click(
            self.delete_index,
            inputs=[self.selected_index_id],
            outputs=[self.selected_index_id],
            show_progress="hidden",
        ).then(self.list_indices, inputs=[], outputs=[self.index_list],).success(
            update_current_module_atime
        )
        self.btn_delete_no.click(
            lambda: (
                gr.update(visible=True),
                gr.update(visible=True),
                gr.update(visible=True),
                gr.update(visible=False),
            ),
            inputs=[],
            outputs=[
                self.btn_edit_save,
                self.btn_delete,
                self.btn_close,
                self._delete_confirm,
            ],
            show_progress="hidden",
        )
        self.btn_edit_save.click(
            self.update_index,
            inputs=[
                self.selected_index_id,
                self.edit_name,
                self.edit_spec,
            ],
            show_progress="hidden",
        ).then(
            self.list_indices,
            inputs=[],
            outputs=[self.index_list],
        )
        self.btn_close.click(
            lambda: -1,
            outputs=[self.selected_index_id],
        )

    def on_index_type_change(self, index_type: str):
        """Update the spec description and pre-fill the default values

        Args:
            index_type: the name of the index type, this is usually the class name

        Returns:
            A tuple of the default spec and the description
        """
        index_type_cls = self.manager.index_types[index_type]
        required: dict = {
            key: value.get("value", None)
            for key, value in index_type_cls.get_admin_settings().items()
        }

        return yaml.dump(required, sort_keys=False), format_description(index_type_cls)

    def create_index(self, name: str, index_type: str, config: str):
        """Create the index

        Args:
            name: the name of the index
            index_type: the type of the index
            config: the expected config of the index
        """
        try:
            self.manager.build_index(
                name=name,
                config=yaml.load(config, Loader=YAMLNoDateSafeLoader),
                index_type=index_type,
            )
            gr.Info(f'Tạo chỉ mục "{name}" thành công. Vui lòng khởi động lại ứng dụng!')
        except Exception as e:
            raise gr.Error(f"Không thể tạo mô hình Embedding {name}: {e}")

    def list_indices(self):
        """List the indices constructed by the user"""
        items = []
        for item in self.manager.indices:
            record = {}
            record["id"] = item.id
            record["tên"] = item.name
            record["loại chỉ mục"] = item.__class__.__name__
            items.append(record)

        if items:
            indices_list = pd.DataFrame.from_records(items)
        else:
            indices_list = pd.DataFrame.from_records(
                [{"id": "-", "tên": "-", "loại chỉ mục": "-"}]
            )

        return indices_list

    def select_index(self, index_list, ev: gr.SelectData) -> int:
        """Return the index id"""
        if ev.value == "-" and ev.index[0] == 0:
            gr.Info("Chưa có chỉ mục nào được tạo. Vui lòng tạo một cái trước!")
            return -1

        if not ev.selected:
            return -1

        return int(index_list["id"][ev.index[0]])

    def on_selected_index_change(self, selected_index_id: int):
        """Show the relevant index as user selects it on the UI

        Args:
            selected_index_id: the id of the selected index
        """
        if selected_index_id == -1:
            _selected_panel = gr.update(visible=False)
            edit_spec = gr.update(value="")
            edit_spec_desc = gr.update(value="")
            edit_name = gr.update(value="")
        else:
            _selected_panel = gr.update(visible=True)
            index = self.manager.info()[selected_index_id]
            edit_spec = yaml.dump(index.config)
            edit_spec_desc = format_description(index.__class__)
            edit_name = index.name

        return (
            _selected_panel,
            edit_spec,
            edit_spec_desc,
            edit_name,
        )

    def update_index(self, selected_index_id: int, name: str, config: str):
        try:
            spec = yaml.load(config, Loader=YAMLNoDateSafeLoader)
            self.manager.update_index(selected_index_id, name, spec)
            gr.Info(f'Cập nhật chỉ mục "{name}" thành công. Vui lòng khởi động lại ứng dụng!')
        except Exception as e:
            raise gr.Error(f'Không thể lưu chỉ mục "{name}": {e}')

    def delete_index(self, selected_index_id):
        try:
            self.manager.delete_index(selected_index_id)
            gr.Info("Xóa chỉ mục thành công. Vui lòng khởi động lại ứng dụng!")
        except Exception as e:
            gr.Warning(f"Không thể xóa chỉ mục: {e}")
            return selected_index_id

        return -1
    