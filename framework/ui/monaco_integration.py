from streamlit_code_editor import code_editor

class CodeMirrorEditor:
    def render(self, content):
        response = code_editor(
            content,
            height=600,
            language="python",
            theme="default",
            key="codemirror-editor"
        )
        return response.get("text", content)

