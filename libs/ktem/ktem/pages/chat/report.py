from typing import Optional

import gradio as gr
from ktem.app import BasePage
from ktem.db.models import IssueReport, engine
from sqlmodel import Session


class ReportIssue(BasePage):
    def __init__(self, app):
        self._app = app
        self.on_building_ui()

    def on_building_ui(self):
        with gr.Accordion(label="Phản hồi", open=False, elem_id="report-accordion"):
            self.correctness = gr.Radio(
                choices=[
                    ("Câu trả lời là chính xác", "correct"),
                    ("Câu trả lời không chính xác", "incorrect"),
                ],
                label="Tính chính xác:",
            )
            self.issues = gr.CheckboxGroup(
                choices=[
                    ("Câu trả lời có tính xúc phạm", "offensive"),
                    ("Bằng chứng không chính xác", "wrong-evidence"),
                ],
                label="Vấn đề khác:",
            )
            self.more_detail = gr.Textbox(
                placeholder=(
                    "Chi tiết thêm (ví dụ: sai như thế nào, câu trả lời đúng là gì, v.v...)"
                ),
                container=False,
                lines=3,
            )
            gr.Markdown(
                "Điều này sẽ gửi cuộc trò chuyện hiện tại và cài đặt người dùng "
                "để hỗ trợ việc điều tra"
            )
            self.report_btn = gr.Button("Báo cáo")

    def report(
        self,
        correctness: str,
        issues: list[str],
        more_detail: str,
        conv_id: str,
        chat_history: list,
        settings: dict,
        user_id: Optional[int],
        info_panel: str,
        chat_state: dict,
        *selecteds,
    ):
        selecteds_ = {}
        for index in self._app.index_manager.indices:
            if index.selector is not None:
                if isinstance(index.selector, int):
                    selecteds_[str(index.id)] = selecteds[index.selector]
                elif isinstance(index.selector, tuple):
                    selecteds_[str(index.id)] = [selecteds[_] for _ in index.selector]
                else:
                    print(f"Unknown selector type: {index.selector}")

        with Session(engine) as session:
            issue = IssueReport(
                issues={
                    "correctness": correctness,
                    "issues": issues,
                    "more_detail": more_detail,
                },
                chat={
                    "conv_id": conv_id,
                    "chat_history": chat_history,
                    "info_panel": info_panel,
                    "chat_state": chat_state,
                    "selecteds": selecteds_,
                },
                settings=settings,
                user=user_id,
            )
            session.add(issue)
            session.commit()
        gr.Info("Cảm ơn bạn đã phản hồi")
        