# sota-neural-translation

|            | Seq2Seq LSTM | Seq2Seq GRU w/Attn | Transformer |
|------------|--------------|--------------------|-------------|
| CE Train   | ??           | 1.81               | 3.28        |
| CE Val     | ??           | 3.20               | 3.31        |
| CE Test    | ??           | 3.25               | 3.287       |
| Perplexity | ??           | 25.774             | 26.751      |
| BLEU       | 17?          | 29.3               | 10.5        |
| Parameters | 50M          | 20M                | 50M         |			

---

###### Original:
`ein mann in einem blauen hemd steht auf einer leiter und putzt ein fenster .`

###### Translation:
`a man in a blue shirt is standing on a ladder and cleaning a window .`

---

###### Seq2Seq LSTM:
`????`

###### Seq2Seq GRU w/ Attn:
`a man in a blue shirt is standing on a ladder cleaning a window .`

###### Transformer:
`a man in a white shirt and hat is sitting on a bench .`
