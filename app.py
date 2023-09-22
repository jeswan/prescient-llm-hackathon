import lightning as L
import lightning.app.frontend as frontend
import streamlit as st


def your_streamlit_app(lightning_app_state):
    st.header("LLM Hacakathon", divider="rainbow")

    image = st.file_uploader(
        label="Upload an image below to recieve a text description",
        type=["png", "jpg", "jpeg", "webp", "tif", "tiff"],
    )

    if image is not None:
        st.image(image)

        question = st.text_area("Question")

        clicked = st.button("Ask")

        if clicked:
            # make request with (question, image)

            answer = "foo"

            st.markdown(answer)


def on_predict(file):
    bytes_data = file.getvalue()

    return bytes_data


class LitStreamlit(L.LightningFlow):
    def __init__(self):
        super().__init__()

        self.prediction = ""

    def configure_layout(self):
        return frontend.StreamlitFrontend(render_fn=your_streamlit_app)

    def run(self, prediction: str):
        self.prediction = prediction


class LitApp(L.LightningFlow):
    def __init__(self):
        super().__init__()

        self.lit_streamlit = LitStreamlit()

    def run(self):
        self.lit_streamlit.run("")

    def configure_layout(self):
        header = {
            "name": "header",
            "body": "LLM Hacakathon",
            "content": self.lit_streamlit,
        }

        return [header]


app = L.LightningApp(LitApp())
