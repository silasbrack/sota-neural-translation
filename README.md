# sota-neural-translation

|            | Seq2Seq LSTM | Seq2Seq GRU w/Attn | Transformer |
|------------|--------------|--------------------|-------------|
| CE Train   | 2.19         | 1.81               | 3.28        |
| CE Val     | 2.71         | 3.20               | 3.31        |
| CE Test    | 2.84         | 3.25               | 3.287       |
| Perplexity | 17.116       | 25.774             | 26.751      |
| BLEU       | 21.7         | 29.3               | 10.5        |
| BLEU WMT14 | 0.13         | 0.21               | 2.8         |
| Parameters | 35M          | 20M                | 50M         |			

---

###### Original:
`eine gruppe von menschen steht vor einem iglu .`

###### Translation:
`a group of people stands in front of an igloo .`

---

###### Seq2Seq LSTM:
`????`

###### Seq2Seq GRU w/ Attn:
`a group of people standing in front of a theater .`

###### Transformer:
`a group of people are standing in front of a large building .`
