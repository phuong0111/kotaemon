from textwrap import dedent

import gradio as gr
from ktem.app import BasePage


class HintPage(BasePage):
    def __init__(self, app):
        self._app = app
        self.on_building_ui()

    def on_building_ui(self):
        with gr.Accordion(label="Gợi ý", open=False):
            gr.Markdown(
                dedent(
                    """
                - Bạn có thể chọn bất kỳ văn bản nào từ câu trả lời chat để **làm nổi bật các trích dẫn liên quan** trên panel bên phải.
                - **Trích dẫn** có thể được xem trên cả trình xem PDF và văn bản thô.
                - Bạn có thể điều chỉnh định dạng trích dẫn và sử dụng lý luận nâng cao (CoT) trong menu **Cài đặt Chat**.
                - Muốn **khám phá thêm**? Kiểm tra phần **Trợ giúp** để tạo không gian riêng của bạn.
            """  # noqa
                )
            )
            