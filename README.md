# Comic OCR
An OCR system fine-tuned for comics and manga


## Usage

```
pip install comic-ocr
```


```python
import comic_ocr

# paragraphs: List[comic_ocr.typing.Paragraph]
paragraphs = comic_ocr.read_paragraphs('../example/xkcd_100.png')

...
paragraphs[0].text # 'Just wrote a paper disproving gravity.'
paragraphs[0].location # Rect(size=(168, 40), at=(15, 32))
```
