from streamlit_ace import st_ace

class AceEditor:
    def render(self, content):
        return st_ace(
            value=content,
            language="python",
            theme="monokai",
            key="ace_editor",
            font_size=14,
            tab_size=4,
            show_gutter=True,
            show_print_margin=False,
            wrap=True,
            auto_update=True,
            readonly=False,
            min_lines=20,
            height=500
        )

