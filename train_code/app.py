import gradio as gr

from app_util import ChatDetDemo

header = '''
<div align=center>
<h1 style="font-weight: 900; margin-bottom: 7px;">
🤖 ChatDET: Contextual Object Detection with Large Language Models
</h1>
</div>
'''

abstract = '''
🤗 This is the official Gradio demo for <b>ChatDET: Contextual Object Detection with Large Language Models</b>.

🆒 Our goal is to promote object detection with better `context understanding` and enable `interactive feedback`
through `human language vocabulary`, all made possible by using large language models!

🤝 This demo is still under construction. Your comments or suggestions are welcome!
'''

footer = r'''
🦁 **Github Repo**
We would be grateful if you consider star our <a href="https://github.com/">github repo</a>

📝 **Citation**
We would be grateful if you consider citing our work if you find it useful:
```bibtex
@article{
}
```

📋 **License**
This project is licensed under
<a rel="license" href="https://github.com/sczhou/CodeFormer/blob/master/LICENSE">S-Lab License 1.0</a>.
Redistribution and use for non-commercial purposes should follow this license.

📧 **Contact**
If you have any questions, please feel free to contact Yuhang Zang <b>(zang0012@ntu.edu.sg)</b>.
'''

css = '''
h1#title {
  text-align: center;
}
'''


qa_samples = [
    ["demo/confusing.jpg", "What is unusual about this image?"],
]

captioning_samples = [
    ["demo/confusing.jpg"], ["demo/pikachu.png"], ["demo/100652400.jpg"]
]

grounding_samples = [
    ["demo/COCO_train2014_000000519404.jpg", "woman in white shirt"],
    ["demo/COCO_train2014_000000519404.jpg", "woman in black shirt"]
]

detection_samples = [
    ["demo/confusing.jpg", "person car"],
    ["demo/000000397133.jpg", "person bottle cup knife spoon bowl broccoli carrot oven sink"],
    # ["demo/000000037777.jpg", "banana orange chair oven sink refrigerator"]
]

model_ovcoco = ChatDetDemo('exps/public/ovcoco/checkpoint.pth')
model_grounding = ChatDetDemo('exps/public/refcocog/checkpoint.pth')
model_qa = ChatDetDemo('exps/public/flickr/checkpoint.pth')


def inference_fn_select(image_input, text_input, task_button, history=[]):
    if task_button == 'Question Answering' or task_button == 'Captioning':
        return model_qa.forward(image_input, text_input, task_button, history)
    elif task_button == 'Grounding':
        return model_grounding.forward(image_input, text_input, task_button, history)
    else:
        return model_ovcoco.forward(image_input, text_input, task_button, history, threshold=0.2)


def set_qa_samples(example: list) -> dict:
    return gr.Image.update(example[0]), gr.Textbox.update(example[1]), 'Question Answering'


def set_captioning_samples(example: list) -> dict:
    return gr.Image.update(example[0]), gr.Textbox.update(''), 'Captioning'


def set_grounding_samples(example: list) -> dict:
    return gr.Image.update(example[0]), gr.Textbox.update(example[1]), 'Grounding'


def set_detection_samples(example: list) -> dict:
    return gr.Image.update(example[0]), gr.Textbox.update(example[1]), 'Detection'


def inference_fn(image_input, text_input, task_button, history=[]):
    if task_button == "Question Answering":
        history.append(text_input)
        prompt = " ".join(history)
        output = [prompt]
        history += output

        chat = [
            (history[i], history[i + 1]) for i in range(0, len(history) - 1, 2)
        ]
    else:
        prompt = text_input
        history = []
        chat = []
    return image_input, chat, history


with gr.Blocks(css=css, theme=gr.themes.Soft()) as demo:
    gr.Markdown(header)
    gr.Markdown(abstract)
    state = gr.State([])

    with gr.Row():
        with gr.Column(scale=0.5, min_width=500):
            image_input = gr.Image(type="pil", interactive=True, label="Upload an image 📁").style(height=250)
        with gr.Column(scale=0.5, min_width=500):
            chat_input = gr.Textbox(label="Type your text prompt ⤵️")
            task_button = gr.Radio(label="Task type", interactive=True,
                                   choices=['Question Answering', 'Captioning', 'Grounding', 'Detection'],
                                   value='Question Answering')
            with gr.Row():
                submit_button = gr.Button(value="🏃 Run", interactive=True, variant="primary")
                clear_button = gr.Button(value="🔄 Clear", interactive=True)

    with gr.Row():
        with gr.Column(scale=0.5, min_width=500):
            image_output = gr.Image(type='pil', interactive=False, label="Detection output")
        with gr.Column(scale=0.5, min_width=500):
            chat_output = gr.Chatbot(label="Text output").style(height=300)

    with gr.Row():
        qa_examples = gr.Dataset(
            label='Question Answering Examples',
            components=[image_input, chat_input],
            samples=qa_samples,
        )
        captioning_examples = gr.Dataset(
            label='Captioning Examples',
            components=[image_input, ],
            samples=captioning_samples,
        )

    with gr.Row():
        grounding_examples = gr.Dataset(
            label='Grounding Examples',
            components=[image_input, chat_input],
            samples=grounding_samples,
        )
        detection_examples = gr.Dataset(
            label='Detection Examples',
            components=[image_input, chat_input],
            samples=detection_samples,
        )

    submit_button.click(
        inference_fn_select,
        # model.forward,
        [image_input, chat_input, task_button, state],
        [image_output, chat_output, state],
    )
    clear_button.click(
        lambda: (None, None, "", [], [], 'Question Answering'),
        [],
        [image_input, image_output, chat_input, chat_output, state, task_button],
        queue=False,
    )
    image_input.change(
        lambda: (None, "", []),
        [],
        [image_output, chat_output, state],
        queue=False,
    )
    qa_examples.click(
        fn=set_qa_samples,
        inputs=[qa_examples],
        outputs=[image_input, chat_input, task_button],
    )
    captioning_examples.click(
        fn=set_captioning_samples,
        inputs=[captioning_examples],
        outputs=[image_input, chat_input, task_button],
    )
    grounding_examples.click(
        fn=set_grounding_samples,
        inputs=[grounding_examples],
        outputs=[image_input, chat_input, task_button],
    )
    detection_examples.click(
        fn=set_detection_samples,
        inputs=[detection_examples],
        outputs=[image_input, chat_input, task_button],
    )

    gr.Markdown(footer)

demo.launch(enable_queue=True, server_name='172.21.25.96', server_port=19902)
