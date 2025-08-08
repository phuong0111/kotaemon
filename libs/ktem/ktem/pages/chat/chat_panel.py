import gradio as gr
from ktem.app import BasePage
from theflow.settings import settings as flowsettings

KH_DEMO_MODE = getattr(flowsettings, "KH_DEMO_MODE", False)

if not KH_DEMO_MODE:
    PLACEHOLDER_TEXT = (
        "Đây là bắt đầu của một cuộc trò chuyện mới.\n"
        "Hãy bắt đầu bằng cách tải lên một tệp hoặc URL web. "
        "Truy cập tab Tệp để có thêm tùy chọn (ví dụ: GraphRAG)."
    )
else:
    PLACEHOLDER_TEXT = (
        "Chào mừng đến với Kotaemon Demo. "
        "Hãy bắt đầu bằng cách duyệt các cuộc trò chuyện đã được tải sẵn để làm quen.\n"
        "Kiểm tra phần Gợi ý để có thêm mẹo hữu ích."
    )


class ChatPanel(BasePage):
    def __init__(self, app):
        self._app = app
        self.on_building_ui()

    def on_building_ui(self):
        self.chatbot = gr.Chatbot(
            label=self._app.app_name,
            placeholder=PLACEHOLDER_TEXT,
            show_label=False,
            elem_id="main-chat-bot",
            show_copy_button=True,
            likeable=True,
            bubble_full_width=False,
        )
        with gr.Row():
            self.text_input = gr.MultimodalTextbox(
                interactive=True,
                scale=20,
                file_count="multiple",
                placeholder=(
                    "Nhập tin nhắn, tìm kiếm @web, hoặc gắn thẻ tệp bằng @tên_tệp"
                ),
                container=False,
                show_label=False,
                elem_id="chat-input",
            )

    def submit_msg(self, chat_input, chat_history):
        """Submit a message to the chatbot"""
        return "", chat_history + [(chat_input, None)]
    