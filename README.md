<p align="center">
  <a href="#"><img src="assets/bangla_clip_1.PNG" alt="bangla clip"></a>
</p>
<p align="center">
    <em>A dead-simple image search and image-text matching system for Bangla using CLIP (Contrastive Language–Image Pre-training)</em>
</p>


---

#### Installation

* `python >= 3.9`
* `pip install -r requirements.txt`
* Download the model weights and place inside the `models` folder.

### bangla-image-search
The model consists of an EfficientNet / ResNet image encoder and a BERT text encoder and was trained on multiple datasets from Bangla image-text domain. To run the `app`,

```console
streamlit run app.py
```
---

### Demo

<p align="center">
  <a href="#"><img src="assets/bangla_clip_2.PNG" alt="bangla clip"></a>
</p>
<p align="center">
    <em>Live Demo: </em> <a href="https://huggingface.co/spaces/zabir-nabil/bangla-clip">HuggingFace Space</>
</p>


### Training CLIP for Bangla

 * *Training Code:* [bangla-CLIP](https://github.com/zabir-nabil/bangla-CLIP)
 * *Article:* [medium](https://medium.com/@furcifer/a-dead-simple-image-search-engine-for-bangla-using-clip-contrastive-language-image-pre-training-a1f7f50b8419)


